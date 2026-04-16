"""
Main Management UI Application.

NiceGUI-based web interface for managing MCP tools.

Run with:
    python -m mcp_ui
    # OR
    python -m mcp_ui.management_ui

Port is read from config/ports.json (reserved.management_ui) or
via MCP_UI_PORT environment variable.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from collections.abc import Callable

from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from nicegui import ui, app as nicegui_app

# Set up structured logging
from .logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()


def _load_ports_config() -> dict:
    """
    Load the central ports configuration.

    Returns:
        Dictionary with ports configuration
    """
    possible_paths = [
        Path(__file__).parent.parent / "config" / "ports.json",
        Path(__file__).parent.parent / "ports.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with Path(path).open() as f:
                    return json.load(f)
            except Exception:
                pass

    raise ValueError(
        "ports.json not found. Please create config/ports.json with "
        "'reserved.management_ui' port."
    )


def _get_ui_port_from_config() -> int | None:
    """Get the UI port from ports.json."""
    try:
        ports_config = _load_ports_config()
        return ports_config.get("reserved", {}).get("management_ui")
    except Exception:
        return None


def _get_default_ui_port() -> int:
    """Get default UI port from config or environment."""
    # First check environment variable
    env_port = os.environ.get("MCP_UI_PORT")
    if env_port:
        return int(env_port)

    # Then check ports.json
    port = _get_ui_port_from_config()
    if port:
        return port

    # No fallback - require configuration
    raise ValueError(
        "UI port not configured. Set MCP_UI_PORT environment variable or "
        "create config/ports.json with reserved.management_ui port."
    )


def _get_ui_theme() -> str:
    """Get UI theme from environment or config."""
    env_theme = os.environ.get("MCP_UI_THEME")
    if env_theme:
        return env_theme
    try:
        ports_config = _load_ports_config()
        return ports_config.get("ui", {}).get("theme", "dark")
    except Exception:
        return "dark"


def _get_storage_secret() -> str:
    """Get storage secret from environment variable, or use default for development.

    Note: The secret is used for session cookie signing (required by app.storage.user).
    The actual user data is stored as plain JSON in .nicegui/storage-user-*.json files.
    """
    secret = os.environ.get("MCP_UI_SECRET")
    if secret:
        return secret

    # Check secrets file
    secrets_file = Path.home() / ".mcp_ui" / "secrets"
    if secrets_file.exists():
        try:
            with Path(secrets_file).open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("=", 1)
                        if len(parts) == 2 and parts[0].strip() == "MCP_UI_SECRET":
                            return parts[1].strip()
        except Exception:
            pass

    # Development default (data is plain JSON anyway)
    return "dev-only-secret"


# =============================================================================
# Configuration
# =============================================================================

ENV_USERNAME = "MCP_UI_USERNAME"
ENV_PASSWORD = "MCP_UI_PASSWORD"

# Default credentials (for development only)
passwords = {"admin": "admin"}


# =============================================================================
# Imports
# =============================================================================

from .api_client import get_client
from .state import get_state
from .components import ToolList, ToolCard, show_success, show_error, show_global_tool_settings
from .components.auth_box import AuthBox
from .components.env_var_editor import parse_env_vars_from_api


# =============================================================================
# Global State
# =============================================================================

_api_client = None


def get_api_client():
    """Get or create the API client."""
    global _api_client
    if _api_client is None:
        _api_client = get_client()
    return _api_client


# =============================================================================
# Page Functions
# =============================================================================

@ui.page("/login")
def login(redirect_to: str = "/") -> RedirectResponse | None:
    """Login page route."""
    logger.debug(f"Login page called, redirect_to={redirect_to}")

    def try_login() -> None:
        """Try to log in with the provided credentials."""
        stored_username = os.environ.get(ENV_USERNAME)
        stored_password = os.environ.get(ENV_PASSWORD)

        # Use defaults if no credentials configured
        if stored_username is None and stored_password is None:
            stored_username = "admin"
            stored_password = "admin"

        if username.value == stored_username and password.value == stored_password:
            nicegui_app.storage.user.update({"username": username.value, "authenticated": True})
            show_success("Login successful!")
            ui.navigate.to(redirect_to)
        else:
            show_error("Wrong username or password")

    # If already authenticated, redirect to main page
    if nicegui_app.storage.user.get("authenticated", False):
        return RedirectResponse("/")

    with ui.card().classes("absolute-center"):
        ui.label("Management UI Login").classes("text-h5 mb-4")
        username = ui.input("Username").on("keydown.enter", try_login).classes("w-full mb-2")
        password = ui.input("Password", password=True, password_toggle_button=True).on(
            "keydown.enter", try_login
        ).classes("w-full mb-4")
        ui.button("Log in", on_click=try_login).classes("w-full")

    return None


@ui.page("/")
async def main_page() -> None:
    """Main management page route."""
    logger.info("page: main_page loaded")

    # Check authentication inside the page function
    if not nicegui_app.storage.user.get("authenticated", False):
        ui.navigate.to("/login?redirect_to=/")
        return

    def logout() -> None:
        nicegui_app.storage.user.clear()
        ui.navigate.to("/login")

    state = get_state()

    # Set up theme
    theme = _get_ui_theme()
    if theme == "dark":
        ui.dark_mode().enable()

    # === HEADER ===
    with ui.header().classes("w-full p-4 bg-primary"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(
                f"MCP Tools Management - {nicegui_app.storage.user.get('username', 'User')}"
            ).classes("text-h5 text-white")

            with ui.row():
                status_icon = ui.icon("circle").classes(
                    "text-green-400" if state.connection_status == "connected" else "text-red-400"
                )

                ui.button(
                    "Settings",
                    icon="settings",
                    on_click=lambda: _open_tool_settings(state),
                ).props("flat color=white")

                ui.button("Logout", icon="logout", on_click=logout).props("flat color=white")

    # === CONTENT ROW with refreshable ===
    @ui.refreshable
    async def content_area():
        """Refreshable content area that updates when tool is selected."""
        with ui.row().classes("w-full h-[calc(100vh-64px)]"):
            # Left sidebar
            with ui.column().classes("w-64 p-4 bg-gray-100 dark:bg-gray-800 overflow-auto"):
                await _render_sidebar(state, content_area.refresh)

            # Main content
            with ui.column().classes("flex-1 p-4 overflow-auto"):
                await _render_content(state)

    # Initial load: fetch tools before first render, then refresh UI
    if not state.tools and not state.loading_tools:
        await _refresh_tools()
        content_area.refresh()

    await content_area()


async def _open_tool_settings(state) -> None:
    """Open the global tool settings dialog."""
    from .components.tool_settings import _load_tools_config, _save_tools_config
    from launcher.tools_config import discover_tools_from_server
    from launcher.env_manager import load_auth_config

    servers = [tool.name for tool in state.tools]

    # Discover tools for servers that have empty tool lists in config
    config = _load_tools_config()
    needs_update = False
    for tool_info in state.tools:
        server_name = tool_info.name
        if not config.get("tools", {}).get(server_name) and tool_info.mcp_port:
            mcp_url = f"http://localhost:{tool_info.mcp_port}/mcp"
            auth_cfg = load_auth_config(server_name)
            api_key = auth_cfg.get("api_key")
            discovered = await discover_tools_from_server(mcp_url, api_key=api_key)
            if discovered:
                if "tools" not in config:
                    config["tools"] = {}
                config["tools"][server_name] = discovered
                needs_update = True

    if needs_update:
        _save_tools_config(config)

    await show_global_tool_settings(servers)


async def _render_sidebar(state, content_refresh: Callable = None, initial_load: bool = True) -> None:
    """Render the sidebar with tool list."""
    logger.debug("_render_sidebar called")

    async def _fetch_tool_detail(tool_name: str) -> None:
        """Background task to fetch tool detail, extensions, and env vars in parallel."""
        client = get_api_client()
        state = get_state()

        # Fetch tool detail, extensions, and env vars concurrently
        tool_task = asyncio.create_task(client.get_tool(tool_name))
        ext_task = asyncio.create_task(client.get_extensions(tool_name))
        env_task = asyncio.create_task(client.get_tool_env(tool_name))

        # Show tool detail as soon as it's ready
        tool_response = await tool_task
        if tool_response.success:
            state.selected_tool_detail = tool_response.data
            if content_refresh:
                content_refresh()

        # Wait for extensions and attach
        ext_response = await ext_task
        if ext_response.success and state.selected_tool_detail:
            state.selected_tool_detail.extensions = ext_response.data

        # Wait for env vars and attach to state
        env_response = await env_task
        if env_response.success and env_response.data:
            env_vars = parse_env_vars_from_api(env_response.data)
            state.env_variables = env_vars
            state.env_cache[tool_name] = env_vars
        else:
            state.env_variables = state.env_cache.get(tool_name, [])

        if not tool_response.success:
            state.selected_tool_detail = None
            show_error(f"Error loading tool: {tool_response.error}")
        elif state.selected_tool_detail:
            # Cache for instant re-selection
            state.tool_detail_cache[tool_name] = state.selected_tool_detail

        state.loading_detail = False
        # Refresh UI with final state
        if content_refresh:
            content_refresh()

    def on_select(tool_name: str) -> None:
        state = get_state()
        state.select_tool(tool_name)
        # Check cache for previously fetched detail
        cached = state.tool_detail_cache.get(tool_name)
        if cached:
            state.selected_tool_detail = cached
            state.loading_detail = False
            if content_refresh:
                content_refresh()
            return
        # Immediately show loading state
        state.loading_detail = True
        state.selected_tool_detail = None
        if content_refresh:
            content_refresh()
        # Fetch in background
        asyncio.create_task(_fetch_tool_detail(tool_name))

    async def on_refresh() -> None:
        await _refresh_tools()
        if content_refresh:
            content_refresh()

    ToolList(
        tools=state.tools,
        selected_tool=state.selected_tool,
        on_select=on_select,
        on_refresh=on_refresh,
        loading=state.loading_tools,
    )
    logger.debug("_render_sidebar completed")


async def _render_content(state) -> None:
    """Render the main content area."""
    logger.debug("_render_content called")

    # Use cached tool detail from background fetch
    selected_tool_detail = state.selected_tool_detail

    # Always fetch env vars directly to guarantee they're available for rendering.
    # Background fetch in _fetch_tool_detail updates the cache, but we can't rely
    # on it completing before this render fires.
    # Always fetch env vars for current tool to get latest values (e.g. from .env changes)
    env_variables = state.env_cache.get(selected_tool_detail.name, []) if selected_tool_detail else []
    if selected_tool_detail:
        client = get_api_client()
        env_response = await client.get_tool_env(selected_tool_detail.name)
        if env_response.success and env_response.data:
            env_variables = parse_env_vars_from_api(env_response.data)
            state.env_variables = env_variables
            state.env_cache[selected_tool_detail.name] = env_variables
        elif not env_variables:
            env_variables = state.env_cache.get(selected_tool_detail.name, [])

    # Fetch auth config for the tool
    tool_auth: dict = state.tool_auth.get(selected_tool_detail.name, {}) if selected_tool_detail else {}
    if selected_tool_detail:
        client = get_api_client()
        auth_response = await client.get_tool_auth(selected_tool_detail.name)
        if auth_response.success and auth_response.data:
            tool_auth = auth_response.data.get("api_key", {})
            state.tool_auth[selected_tool_detail.name] = tool_auth

    async def on_query(ext_name: str, params: dict[str, Any] | None = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.query_extension(state.selected_tool, ext_name, params)
            if response.success:
                # Update the extension's data in place and refresh UI
                for ext in state.selected_tool_detail.extensions:
                    if ext.name == ext_name:
                        ext.data = response.data
                        break
                show_success("Query successful")
                if content_refresh:
                    content_refresh()
            else:
                show_error(f"Query failed: {response.error}")
        finally:
            state.loading_detail = False

    async def on_execute(ext_name: str, params: dict[str, Any] | None = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.execute_extension(state.selected_tool, ext_name, params)
            if response.success:
                show_success("Action executed successfully")
            else:
                show_error(f"Execution failed: {response.error}")
        finally:
            state.loading_detail = False

    async def on_env_update(tool_name: str, var_name: str, value: str) -> None:
        """Handle environment variable update."""
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.update_tool_env(tool_name, {var_name: value})
            if response.success:
                show_success(f"Updated {var_name}")
                # Refresh env vars
                env_response = await client.get_tool_env(tool_name)
                if env_response.success and env_response.data:
                    from .components.env_var_editor import parse_env_vars_from_api
                    state.env_variables = parse_env_vars_from_api(env_response.data)
                    state.env_cache[tool_name] = state.env_variables
                if content_refresh:
                    content_refresh()
            else:
                show_error(f"Failed to update {var_name}: {response.error}")
        finally:
            state.loading_detail = False

    async def on_env_delete(tool_name: str, var_name: str) -> None:
        """Handle environment variable deletion."""
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.delete_tool_env(tool_name, var_name)
            if response.success:
                show_success(f"Removed {var_name}")
                # Refresh env vars
                env_response = await client.get_tool_env(tool_name)
                if env_response.success and env_response.data:
                    from .components.env_var_editor import parse_env_vars_from_api
                    state.env_variables = parse_env_vars_from_api(env_response.data)
                    state.env_cache[tool_name] = state.env_variables
                if content_refresh:
                    content_refresh()
            else:
                show_error(f"Failed to delete {var_name}: {response.error}")
        finally:
            state.loading_detail = False

    async def on_auth_update(tool_name: str, new_api_key: str) -> None:
        """Handle auth key update."""
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.update_tool_auth(tool_name, new_api_key)
            if response.success:
                show_success("Authorization key updated - restart tool to take effect")
                # Refresh auth config
                auth_response = await client.get_tool_auth(tool_name)
                if auth_response.success and auth_response.data:
                    tool_auth = auth_response.data.get("api_key", {})
                    state.tool_auth[tool_name] = tool_auth
                if content_refresh:
                    content_refresh()
            else:
                show_error(f"Failed to update auth key: {response.error}")
        finally:
            state.loading_detail = False

    # Render auth section
    if selected_tool_detail:
        AuthBox(
            tool_name=selected_tool_detail.name,
            is_set=tool_auth.get("is_set", False),
            value_masked=tool_auth.get("value_masked"),
            on_update=on_auth_update,
        )

    ToolCard(
        tool=selected_tool_detail,
        on_query=on_query,
        on_execute=on_execute,
        loading=state.loading_detail,
        env_variables=env_variables,
        on_env_update=on_env_update,
        on_env_delete=on_env_delete,
    )
    logger.debug("_render_content completed")


async def _refresh_tools() -> None:
    """Refresh the tools list from the API."""
    logger.info("action: refresh_tools started")

    state = get_state()
    state.loading_tools = True
    state.last_error = None
    state.tool_detail_cache.clear()

    try:
        client = get_api_client()
        response = await client.get_tools()

        if response.success:
            state.set_tools(response.data)
            state.connection_status = "connected"
            logger.info(f"action: refresh_tools success count={len(response.data)}")
        else:
            state.set_error(response.error)
            show_error(f"Connection error: {response.error}")

    except Exception as e:
        state.connection_status = "error"
        state.last_error = str(e)
        show_error(f"Unexpected error: {e}")
    finally:
        state.loading_tools = False

    logger.debug(f"_refresh_tools completed, tools: {len(state.tools)}")
    ui.update()


# =============================================================================
# Uvicorn Support - Use ui.run_with() to attach NiceGUI to FastAPI app
# =============================================================================

logger.info("Initializing NiceGUI with FastAPI")

# Create a FastAPI app and attach NiceGUI to it
from fastapi import FastAPI

fastapi_app = FastAPI(title="MCP Tools Management UI")

# Initialize NiceGUI with the FastAPI app
ui.run_with(fastapi_app, storage_secret=_get_storage_secret())

# Export the FastAPI app for uvicorn
app = fastapi_app

logger.info("NiceGUI initialization complete")


def run_ui() -> None:
    """Run the UI server."""
    import signal
    import os
    import sys

    port = _get_default_ui_port()
    theme = _get_ui_theme()
    storage_secret = _get_storage_secret()

    logger.info(f"Starting UI server on port {port}, theme={theme}")

    run_kwargs = {
        "port": port,
        "title": "MCP Tool Manager",
        "dark": theme == "dark",
        "reload": False,
        "show": False,
    }
    if storage_secret:
        run_kwargs["storage_secret"] = storage_secret

    # NiceGUI/uvicorn install their own signal handlers.
    # Use os._exit(0) to bypass NiceGUI's cleanup and ensure exit code 0.
    def _sigint_handler(signum, frame):
        logger.info("Received SIGINT, shutting down")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Install handler before NiceGUI starts
    signal.signal(signal.SIGINT, _sigint_handler)

    # Also handle SIGTERM
    def _sigterm_handler(signum, frame):
        logger.info("Received SIGTERM, shutting down")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    ui.run(**run_kwargs)


if __name__ in {"__main__", "__mp_main__"}:
    run_ui()
