"""
Main Management UI Application.

NiceGUI-based web interface for managing MCP tools.

Run with:
    python -m mcp_ui
    # OR
    python -m mcp_ui.management_ui

Port is read from config/ports.json (reserved.management_ui) or
via MCP_UI_PORT environment variable.

Layout (2026-08 overhaul): ui.drawer navigation with the live tool list,
per-tool tab panels (Overview / Extensions / Env Vars / Auth), targeted
panel refreshes instead of full-page teardown, and a 10s status poll.
"""

import asyncio
import hmac
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
setup_logging(os.environ.get("MCP_UI_LOG_LEVEL", "INFO"))
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


_storage_secret: str | None = None


def _get_storage_secret() -> str:
    """Get storage secret from environment, secrets file, or generate one.

    The secret signs session cookies (required by app.storage.user). It is
    resolved once per process and cached so both the run_with() and ui.run()
    call sites see the same value.

    Security: there is deliberately NO hardcoded fallback. Without a configured
    secret, an ephemeral random one is generated — sessions survive only until
    restart, and cookies signed with a leaked hardcoded default could be
    forged to bypass login.
    """
    global _storage_secret
    if _storage_secret:
        return _storage_secret

    secret = os.environ.get("MCP_UI_SECRET")
    if secret:
        _storage_secret = secret
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
                            _storage_secret = parts[1].strip()
                            return _storage_secret
        except Exception:
            pass

    # Ephemeral per-process secret: logins reset on restart, nothing forgeable.
    import secrets as _secrets

    _storage_secret = _secrets.token_hex(32)
    logger.warning(
        "MCP_UI_SECRET is not set — using an ephemeral in-memory secret. "
        "All logins are invalidated on restart. Set MCP_UI_SECRET for persistence."
    )
    return _storage_secret


# =============================================================================
# Configuration
# =============================================================================

ENV_USERNAME = "MCP_UI_USERNAME"
ENV_PASSWORD = "MCP_UI_PASSWORD"

POLL_INTERVAL_SECONDS = 10.0


def _safe_redirect_target(target: str) -> str:
    """Restrict post-login redirects to same-site absolute paths."""
    return target if target.startswith("/") and not target.startswith("//") else "/"


# =============================================================================
# Imports
# =============================================================================

from .api_client import get_client
from .state import get_state
from .components import ToolList, show_success, show_error, show_global_tool_settings
from .components.tool_card import ToolOverview, render_empty_state, render_loading_state
from .components.data_sources_box import DataSourcesBox
from .components.actions_box import ActionsBox
from .components.auth_box import AuthBox
from .components.env_var_editor import EnvVarEditor, parse_env_vars_from_api
from .components.loading import loading_spinner


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


def verify_credentials(username: str, password: str) -> bool:
    """Check credentials against env config with per-credential defaults.

    Either MCP_UI_USERNAME or MCP_UI_PASSWORD may be overridden independently;
    unset values fall back to admin/admin. Comparison is constant-time.
    """
    stored_username = os.environ.get(ENV_USERNAME) or "admin"
    stored_password = os.environ.get(ENV_PASSWORD) or "admin"
    user_ok = hmac.compare_digest((username or "").encode(), stored_username.encode())
    pass_ok = hmac.compare_digest((password or "").encode(), stored_password.encode())
    return user_ok and pass_ok


# =============================================================================
# Page Functions
# =============================================================================

@ui.page("/login")
def login(redirect_to: str = "/") -> RedirectResponse | None:
    """Login page route."""
    logger.debug(f"Login page called, redirect_to={redirect_to}")

    def try_login() -> None:
        """Try to log in with the provided credentials."""
        if verify_credentials(username.value, password.value):
            nicegui_app.storage.user.update({"username": username.value, "authenticated": True})
            show_success("Login successful!")
            ui.navigate.to(_safe_redirect_target(redirect_to))
        else:
            show_error("Wrong username or password")

    # If already authenticated, redirect to main page
    if nicegui_app.storage.user.get("authenticated", False):
        return RedirectResponse("/")

    # Center via flex, NOT Quasar's absolute-center: absolutely-positioned
    # + transformed containers break hit-testing for client events (buttons
    # render but their click events never reach the server).
    with ui.column().classes("w-full h-screen items-center justify-center"):
        with ui.card():
            ui.label("Management UI Login").classes("text-h5 mb-4")
            username = ui.input("Username").on("keydown.enter", try_login).classes("w-full mb-2")
            password = ui.input("Password", password=True, password_toggle_button=True).on(
                "keydown.enter", try_login
            ).classes("w-full mb-4")
            ui.button("Log in", on_click=try_login).classes("w-full")

    return None


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
            mcp_url = f"http://127.0.0.1:{tool_info.mcp_port}/mcp"
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
    dark_handle = ui.dark_mode()
    if theme == "dark":
        dark_handle.enable()

    # Design tokens: consistent radii and a subtle drawer border in both modes
    ui.add_css("""
        .q-card { border-radius: 12px; }
        .q-drawer { border-right: 1px solid rgba(127, 127, 127, 0.2); }
    """)

    # === Region rebuilds: container.clear() + rebuild into the container ===
    # The @ui.refreshable pattern re-rendered into the ambient slot stack and
    # appended duplicate layouts whenever the page function re-ran (reconnect)
    # or handlers refreshed from outside the original slot. Explicit
    # containers are slot-safe by construction. The containers themselves are
    # created in the layout section below; these closures resolve them (and
    # the handlers) at call time.

    def rebuild_status_chip():
        chip_container.clear()
        with chip_container:
            connected = state.connection_status == "connected"
            ui.badge(
                state.connection_status,
                color="green" if connected else "red",
            ).classes("text-xs")

    def rebuild_error_banner():
        banner_container.clear()
        with banner_container:
            if state.last_error:
                show_error(
                    f"Connection error: {state.last_error}",
                    on_dismiss=lambda: (state.set_error(None), rebuild_error_banner()),
                    on_retry=on_refresh_tools,
                )

    def rebuild_sidebar():
        sidebar_container.clear()
        with sidebar_container:
            ToolList(
                tools=state.tools,
                selected_tool=state.selected_tool,
                on_select=on_select,
                on_refresh=on_refresh_tools,
                loading=state.loading_tools,
            )

    async def rebuild_content():
        content_container.clear()
        with content_container:
            await _render_content_area(state, handlers)

    def schedule_content_rebuild() -> None:
        """Content rebuild is async (env/auth fetches); schedule from sync handlers."""
        asyncio.create_task(rebuild_content())

    # === Handlers ===

    async def fetch_tool_detail(tool_name: str) -> None:
        """Fetch detail, extensions, and env vars for the selected tool.

        Every await is a chance for the user to select a different tool — a
        stale fetch must not write its results into shared state (UI-9 race).
        """
        client = get_api_client()
        tool_task = asyncio.create_task(client.get_tool(tool_name))
        ext_task = asyncio.create_task(client.get_extensions(tool_name))
        env_task = asyncio.create_task(client.get_tool_env(tool_name))

        tool_response = await tool_task
        if get_state().selected_tool != tool_name:
            return
        if tool_response.success:
            state.selected_tool_detail = tool_response.data
            await rebuild_content()
        else:
            state.selected_tool_detail = None
            show_error(f"Error loading tool: {tool_response.error}")

        ext_response = await ext_task
        if get_state().selected_tool != tool_name:
            return
        if ext_response.success and state.selected_tool_detail:
            state.selected_tool_detail.extensions = ext_response.data
            await rebuild_content()

        env_response = await env_task
        if get_state().selected_tool != tool_name:
            return
        if env_response.success and env_response.data:
            env_vars = parse_env_vars_from_api(env_response.data)
            state.env_variables = env_vars
            state.env_cache[tool_name] = env_vars
        else:
            state.env_variables = state.env_cache.get(tool_name, [])

        if state.selected_tool_detail:
            state.tool_detail_cache[tool_name] = state.selected_tool_detail
        state.loading_detail = False
        await rebuild_content()

    def on_select(tool_name: str) -> None:
        state = get_state()
        if state.selected_tool == tool_name:
            return
        state.active_tab = "overview"
        state.select_tool(tool_name)
        cached = state.tool_detail_cache.get(tool_name)
        if cached:
            state.selected_tool_detail = cached
            state.loading_detail = False
        else:
            state.loading_detail = True
            state.selected_tool_detail = None
            state.env_variables = state.env_cache.get(tool_name, [])
        rebuild_sidebar()
        schedule_content_rebuild()
        if not cached:
            asyncio.create_task(fetch_tool_detail(tool_name))

    async def on_refresh_tools() -> None:
        await _refresh_tools()
        rebuild_sidebar()
        await rebuild_content()

    async def on_query(ext_name: str, params: dict[str, Any] | None = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        try:
            response = await get_api_client().query_extension(
                state.selected_tool, ext_name, params
            )
            if response.success:
                for ext in (
                    state.selected_tool_detail.extensions if state.selected_tool_detail else []
                ):
                    if ext.name == ext_name:
                        ext.data = response.data
                        break
                show_success("Query successful")
                schedule_content_rebuild()
            else:
                show_error(f"Query failed: {response.error}")
        except Exception as e:
            show_error(f"Query failed: {e}")

    async def on_execute(ext_name: str, params: dict[str, Any] | None = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        try:
            response = await get_api_client().execute_extension(
                state.selected_tool, ext_name, params
            )
            if response.success:
                show_success("Action executed successfully")
            else:
                show_error(f"Execution failed: {response.error}")
        except Exception as e:
            show_error(f"Execution failed: {e}")

    async def on_env_update(tool_name: str, var_name: str, value: str) -> None:
        """Handle environment variable update."""
        try:
            response = await get_api_client().update_tool_env(tool_name, {var_name: value})
            if response.success:
                show_success(f"Updated {var_name}")
                state.env_cache.pop(tool_name, None)  # force re-fetch at next render
                schedule_content_rebuild()
            else:
                show_error(f"Failed to update {var_name}: {response.error}")
        except Exception as e:
            show_error(f"Failed to update {var_name}: {e}")

    async def on_env_delete(tool_name: str, var_name: str) -> None:
        """Handle environment variable deletion."""
        try:
            response = await get_api_client().delete_tool_env(tool_name, var_name)
            if response.success:
                show_success(f"Removed {var_name}")
                state.env_cache.pop(tool_name, None)
                schedule_content_rebuild()
            else:
                show_error(f"Failed to delete {var_name}: {response.error}")
        except Exception as e:
            show_error(f"Failed to delete {var_name}: {e}")

    async def on_auth_update(tool_name: str, new_api_key: str) -> None:
        """Handle auth key update."""
        try:
            response = await get_api_client().update_tool_auth(tool_name, new_api_key)
            if response.success:
                show_success("Authorization key updated - restart tool to take effect")
                state.tool_auth.pop(tool_name, None)
                schedule_content_rebuild()
            else:
                show_error(f"Failed to update auth key: {response.error}")
        except Exception as e:
            show_error(f"Failed to update auth key: {e}")

    handlers = {
        "on_query": on_query,
        "on_execute": on_execute,
        "on_env_update": on_env_update,
        "on_env_delete": on_env_delete,
        "on_auth_update": on_auth_update,
    }

    async def poll_status() -> None:
        """Periodic status poll: updates badges in place, never rebuilds panels."""
        state = get_state()
        response = await get_api_client().get_tools()
        if response.success:
            old_status = {t.name: t.status for t in state.tools}
            state.set_tools(response.data)
            state.connection_status = "connected"
            new_status = {t.name: t.status for t in state.tools}
            rebuild_status_chip()
            rebuild_sidebar()
            if new_status != old_status:
                # A selected tool changed state — refresh the visible detail.
                schedule_content_rebuild()
        else:
            state.connection_status = "disconnected"
            rebuild_status_chip()

    # === HEADER ===
    with ui.header().classes("w-full p-3"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(
                f"MCP Tools Management — {nicegui_app.storage.user.get('username', 'User')}"
            ).classes("text-h6")
            with ui.row().classes("items-center gap-2"):
                chip_container = ui.row().classes("items-center")
                ui.button(
                    icon="dark_mode",
                    on_click=dark_handle.toggle,
                ).props("flat round").tooltip("Toggle dark mode")
                ui.button(
                    "Settings",
                    icon="settings",
                    on_click=lambda: _open_tool_settings(state),
                ).props("flat")
                ui.button("Logout", icon="logout", on_click=logout).props("flat")

    # === DRAWER: tool navigation ===
    # behavior=desktop pins the drawer open on narrow viewports too — Quasar's
    # default switches to an overlay below 1024px, hiding the navigation
    # (behavior change vs NiceGUI 3.9, where the drawer showed at any width).
    with ui.drawer(side="left").props("bordered behavior=desktop"):
        sidebar_container = ui.column().classes("w-full p-3 gap-2")

    # === CONTENT ===
    with ui.column().classes("w-full p-4 gap-2"):
        banner_container = ui.column().classes("w-full")
        content_container = ui.column().classes("w-full")

    rebuild_status_chip()
    rebuild_error_banner()
    rebuild_sidebar()
    await rebuild_content()

    # Initial load: fetch tools before first render, then rebuild regions
    if not state.tools and not state.loading_tools:
        await _refresh_tools()
        rebuild_sidebar()
        await rebuild_content()

    # Live status: poll without tearing down the user's open panels
    ui.timer(POLL_INTERVAL_SECONDS, poll_status)


async def _render_content_area(state, handlers: dict) -> None:
    """Render the detail area for the selected tool: tabs or empty state.

    ``handlers`` carries the page-level callbacks (on_query, on_execute,
    on_env_update, on_env_delete, on_auth_update) — these live as closures in
    main_page and must be passed down; referencing them directly here raises
    NameError at render time.
    """
    if state.selected_tool is None:
        render_empty_state()
        return

    detail = state.selected_tool_detail
    with ui.tabs(value=state.active_tab, on_change=lambda e: setattr(state, "active_tab", e.value)).classes("w-full") as tabs:
        ui.tab("overview", icon="dashboard", label="Overview")
        ui.tab("extensions", icon="extension", label="Extensions")
        ui.tab("env", icon="tune", label="Env Vars")
        ui.tab("auth", icon="key", label="Auth")

    with ui.tab_panels(tabs, value=state.active_tab).classes("w-full"):
        with ui.tab_panel("overview"):
            await _render_overview_tab(state, detail)
        with ui.tab_panel("extensions"):
            _render_extensions_tab(state, detail, handlers)
        with ui.tab_panel("env"):
            await _render_env_tab(state, detail, handlers)
        with ui.tab_panel("auth"):
            await _render_auth_tab(state, detail, handlers)


async def _render_overview_tab(state, detail) -> None:
    """Overview tab: identity, status, special panels."""
    if detail is None:
        if state.loading_detail:
            render_loading_state(f"Loading {state.selected_tool}...")
        else:
            show_error(f"Could not load details for {state.selected_tool}")
        return

    ToolOverview(tool=detail)


def _render_extensions_tab(state, detail, handlers: dict) -> None:
    """Extensions tab: data sources and actions, one expansion level each."""
    if detail is None:
        if state.loading_detail:
            loading_spinner(f"Loading {state.selected_tool} extensions...")
        else:
            ui.label("Tool details not loaded.").classes("text-grey")
        return

    if not detail.extensions:
        with ui.card().classes("w-full"):
            ui.label("No extensions registered").classes("text-h6 mb-2")
            ui.label(
                "This tool does not register any management extensions through the "
                "launcher's extension registry. Tool calls still work through the "
                "MCP endpoint — this panel only shows launcher-level extensions."
            ).classes("text-grey text-sm")
        return

    data_sources = [e for e in detail.extensions if e.type.value == "data_source"]
    actions = [e for e in detail.extensions if e.type.value == "action"]
    other = [
        e for e in detail.extensions
        if e.type.value not in ("data_source", "action")
    ]

    with ui.column().classes("w-full gap-3"):
        if data_sources:
            DataSourcesBox(extensions=data_sources, on_query=handlers.get("on_query"))
        if actions:
            ActionsBox(extensions=actions, on_execute=handlers.get("on_execute"))
        if other:
            ui.label(f"{len(other)} non-interactive extension(s) "
                     f"(event/stream) not shown here.").classes("text-caption text-grey")


async def _render_env_tab(state, detail, handlers: dict) -> None:
    """Env Vars tab: always fetch fresh values at render time."""
    if detail is None:
        ui.label("Select a tool to see its environment variables.").classes("text-grey")
        return

    env_variables = state.env_cache.get(detail.name, [])
    client = get_api_client()
    env_response = await client.get_tool_env(detail.name)
    if env_response.success and env_response.data:
        env_variables = parse_env_vars_from_api(env_response.data)
        state.env_variables = env_variables
        state.env_cache[detail.name] = env_variables

    EnvVarEditor(
        tool_name=detail.name,
        variables=env_variables,
        on_update=handlers.get("on_env_update"),
        on_delete=handlers.get("on_env_delete"),
    )


async def _render_auth_tab(state, detail, handlers: dict) -> None:
    """Auth tab: fetch current auth config at render time."""
    if detail is None:
        ui.label("Select a tool to see its authorization settings.").classes("text-grey")
        return

    tool_auth: dict = {}
    client = get_api_client()
    auth_response = await client.get_tool_auth(detail.name)
    if auth_response.success and auth_response.data:
        tool_auth = auth_response.data.get("api_key", {})
        state.tool_auth[detail.name] = tool_auth

    AuthBox(
        tool_name=detail.name,
        is_set=tool_auth.get("is_set", False),
        value_masked=tool_auth.get("value_masked"),
        on_update=handlers.get("on_auth_update"),
    )


# =============================================================================
# Entry point
# =============================================================================
# NOTE: there is deliberately NO ui.run_with(fastapi_app) at module scope.
# The earlier dual initialization (run_with at import + ui.run at startup)
# broke client event routing: pages rendered and sockets connected, but
# button events were dispatched into the wrong app instance and silently
# dropped. Standalone ui.run() in run_ui() is the single initialization.

logger.info("mcp_ui initialized (standalone ui.run() entry point)")


def run_ui() -> None:
    """Run the UI server."""
    import signal
    import os
    import sys

    port = _get_default_ui_port()
    theme = _get_ui_theme()
    storage_secret = _get_storage_secret()

    host = os.environ.get("MCP_UI_HOST", "127.0.0.1")
    if not os.environ.get(ENV_USERNAME) and not os.environ.get(ENV_PASSWORD):
        logger.warning(
            "MCP_UI_USERNAME/MCP_UI_PASSWORD are unset — default admin/admin "
            f"credentials are active. UI is bound to {host}; set credentials "
            "before exposing it beyond localhost."
        )
    elif not os.environ.get(ENV_USERNAME) or not os.environ.get(ENV_PASSWORD):
        logger.warning(
            "Only one of MCP_UI_USERNAME/MCP_UI_PASSWORD is set — the other "
            "falls back to 'admin'."
        )

    logger.info(f"Starting UI server on port {port}, theme={theme}")

    run_kwargs = {
        "port": port,
        "host": os.environ.get("MCP_UI_HOST", "127.0.0.1"),
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
