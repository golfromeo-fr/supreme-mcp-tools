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

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

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
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass

    raise ValueError(
        "ports.json not found. Please create config/ports.json with "
        "'reserved.management_ui' port."
    )


def _get_ui_port_from_config() -> Optional[int]:
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
            with open(secrets_file) as f:
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

from .auth import is_auth_enabled, verify_credentials
from .api_client import get_client, close_client
from .models import ToolInfo, ToolDetail, ExtensionType
from .state import get_state
from .components import ToolList, ToolCard, show_success, show_error
from .logging_config import generate_trace_id, set_trace_id


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


async def _render_sidebar(state, content_refresh: callable = None, initial_load: bool = True) -> None:
    """Render the sidebar with tool list."""
    logger.debug("_render_sidebar called")

    def on_select(tool_name: str) -> None:
        state = get_state()
        state.select_tool(tool_name)
        if content_refresh:
            content_refresh()

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

    selected_tool_detail: Optional[ToolDetail] = None
    extensions_error: Optional[str] = None
    if state.selected_tool:
        # Fetch tool detail
        client = get_api_client()
        response = await client.get_tool(state.selected_tool)

        if response.success:
            selected_tool_detail = response.data
            # Fetch extensions separately
            ext_response = await client.get_extensions(state.selected_tool)
            if ext_response.success:
                selected_tool_detail.extensions = ext_response.data
            else:
                extensions_error = ext_response.error
                logger.warning(f"Failed to load extensions for {state.selected_tool}: {extensions_error}")
                show_error(f"Extensions unavailable: {extensions_error}")
        else:
            show_error(f"Error loading tool: {response.error}")

    async def on_query(ext_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.query_extension(state.selected_tool, ext_name, params)
            if response.success:
                show_success("Query successful")
            else:
                show_error(f"Query failed: {response.error}")
        finally:
            state.loading_detail = False

    async def on_mutate(ext_name: str, values: Dict[str, Any]) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.loading_detail = True
        try:
            client = get_api_client()
            response = await client.mutate_extension(state.selected_tool, ext_name, values)
            if response.success:
                show_success("Configuration updated")
            else:
                show_error(f"Update failed: {response.error}")
        finally:
            state.loading_detail = False

    async def on_execute(ext_name: str, params: Optional[Dict[str, Any]] = None) -> None:
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

    ToolCard(
        tool=selected_tool_detail,
        on_query=on_query,
        on_mutate=on_mutate,
        on_execute=on_execute,
        loading=state.loading_detail,
    )
    logger.debug("_render_content completed")


async def _refresh_tools() -> None:
    """Refresh the tools list from the API."""
    logger.info("action: refresh_tools started")

    state = get_state()
    state.loading_tools = True
    state.last_error = None

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
    import uvicorn

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

    ui.run(**run_kwargs)


if __name__ in {"__main__", "__mp_main__"}:
    run_ui()
