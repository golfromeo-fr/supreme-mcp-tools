"""
Main Management UI Application.

NiceGUI-based web interface for managing MCP tools.

Run with:
    python -m mcp_ui.management_ui
    # OR
    uvicorn mcp_ui:app --host 0.0.0.0 --port 8400
    # OR with custom port via environment variable:
    # MCP_UI_PORT=8400 uvicorn mcp_ui:app --host 0.0.0.0 --port 8400
"""

import os
import logging
from typing import Optional, Dict, Any

from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from nicegui import ui, app as nicegui_app

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _get_ui_port_from_config() -> int:
    """Get the UI port from ports.json."""
    try:
        from launcher.launcher_config import load_ports_config
        ports_config = load_ports_config()
        return ports_config.get("reserved", {}).get("management_ui")
    except Exception:
        return None


# Default UI port from ports.json, or environment variable, or error
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


# =============================================================================
# Configuration
# =============================================================================

ENV_USERNAME = "MCP_UI_USERNAME"
ENV_PASSWORD = "MCP_UI_PASSWORD"

# in reality users passwords would obviously need to be hashed
passwords = {'admin': 'admin'}  # Default credentials


# =============================================================================
# Imports
# =============================================================================

from .auth import is_auth_enabled, verify_credentials
from .api_client import ManagementAPIClient, APIError
from .models import ToolInfo, ToolDetail, ExtensionType
from .state import get_state, AppState
from .components import ToolList, ToolCard


# =============================================================================
# Global State
# =============================================================================

_api_client: Optional[ManagementAPIClient] = None


def get_api_client() -> ManagementAPIClient:
    """Get or create the API client."""
    global _api_client
    if _api_client is None:
        _api_client = ManagementAPIClient()
    return _api_client


# =============================================================================
# Page Functions
# =============================================================================

@ui.page('/login')
def login(redirect_to: str = '/') -> RedirectResponse | None:
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
            nicegui_app.storage.user.update({'username': username.value, 'authenticated': True})
            ui.notify('Login successful!', type='positive')
            ui.navigate.to(redirect_to)
        else:
            ui.notify('Wrong username or password', color='negative')
    
    # If already authenticated, redirect to main page
    if nicegui_app.storage.user.get('authenticated', False):
        return RedirectResponse('/')
    
    with ui.card().classes('absolute-center'):
        ui.label('Management UI Login').classes('text-h5 mb-4')
        username = ui.input('Username').on('keydown.enter', try_login).classes('w-full mb-2')
        password = ui.input('Password', password=True, password_toggle_button=True).on('keydown.enter', try_login).classes('w-full mb-4')
        ui.button('Log in', on_click=try_login).classes('w-full')
    
    return None


@ui.page('/')
async def main_page() -> None:
    """Main management page route."""
    logger.debug("Main page called")
    
    # Check authentication inside the page function
    if not nicegui_app.storage.user.get('authenticated', False):
        ui.navigate.to(f'/login?redirect_to=/')
        return
    
    def logout() -> None:
        nicegui_app.storage.user.clear()
        ui.navigate.to('/login')
    
    state = get_state()
    logger.debug(f"State: {state}")
    
    # Set up theme
    theme = os.environ.get('MCP_UI_THEME', 'dark')
    if theme == 'dark':
        ui.dark_mode().enable()
    
    # === HEADER ===
    with ui.header().classes('w-full p-4 bg-primary'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label(f'MCP Tools Management - {nicegui_app.storage.user.get("username", "User")}').classes('text-h5 text-white')
            
            with ui.row():
                status_icon = ui.icon('circle').classes(
                    'text-green-400' if state.connection_status == 'connected' else 'text-red-400'
                )
                
                ui.button(
                    'Logout',
                    icon='logout',
                    on_click=logout
                ).props('flat color=white')
    
    # === CONTENT ROW with refreshable ===
    @ui.refreshable
    async def content_area():
        """Refreshable content area that updates when tool is selected."""
        with ui.row().classes('w-full h-[calc(100vh-64px)]'):
            # Left sidebar
            with ui.column().classes('w-64 p-4 bg-gray-100 dark:bg-gray-800 overflow-auto'):
                await _render_sidebar(state, content_area.refresh)
            
            # Main content
            with ui.column().classes('flex-1 p-4 overflow-auto'):
                await _render_content(state)
    
    await content_area()


async def _render_sidebar(state: AppState, content_refresh: callable = None, initial_load: bool = True) -> None:
    """Render the sidebar with tool list."""
    logger.debug("_render_sidebar called")
    
    # DEFERRING API CALL: Don't block page render with initial API call
    # Instead, trigger refresh after page loads using a timer
    # This prevents NiceGUI's 3-second timeout from cancelling the request
    if initial_load and not state.tools and not state.tools_loading:
        # Schedule the refresh to happen after page renders
        async def schedule_refresh():
            await _refresh_tools()
        ui.timer(0.1, schedule_refresh, once=True)
    
    def on_select(tool_name: str) -> None:
        state = get_state()
        state.selected_tool = tool_name
        # Refresh the content area to show tool details
        if content_refresh:
            content_refresh()
    
    async def on_refresh() -> None:
        await _refresh_tools()
        ui.notify('Tools refreshed', type='positive')
        if content_refresh:
            content_refresh()
    
    ToolList(
        tools=state.tools,
        selected_tool=state.selected_tool,
        on_select=on_select,
        on_refresh=on_refresh,
        loading=state.tools_loading
    )
    logger.debug("_render_sidebar completed")


async def _render_content(state: AppState) -> None:
    """Render the main content area."""
    logger.debug("_render_content called")
    
    selected_tool_detail: Optional[ToolDetail] = None
    if state.selected_tool:
        # Fetch extensions if not already cached for this tool
        if state.selected_tool not in state.extensions:
            try:
                extensions = await get_api_client().get_extensions(state.selected_tool)
                state.extensions[state.selected_tool] = extensions
            except APIError:
                state.extensions[state.selected_tool] = []
        
        try:
            selected_tool_detail = await get_api_client().get_tool(state.selected_tool)
            # Populate extensions from state (API doesn't return extensions in tool detail)
            if selected_tool_detail:
                selected_tool_detail.extensions = state.extensions.get(state.selected_tool, [])
        except APIError as e:
            ui.notify(f"Error loading tool: {e}", type='negative')
    
    async def on_query(ext_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.extensions_loading = True
        try:
            await get_api_client().query_extension(state.selected_tool, ext_name, params)
            ui.notify("Query successful", type='positive')
            ui.navigate.to('/')
        except APIError as e:
            ui.notify(f"Query failed: {e}", type='negative')
        finally:
            state.extensions_loading = False
    
    async def on_mutate(ext_name: str, values: Dict[str, Any]) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.mutation_loading = True
        try:
            await get_api_client().mutate_extension(state.selected_tool, ext_name, values)
            ui.notify("Configuration updated", type='positive')
            ui.navigate.to('/')
        except APIError as e:
            ui.notify(f"Update failed: {e}", type='negative')
        finally:
            state.mutation_loading = False
    
    async def on_execute(ext_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        state = get_state()
        if not state.selected_tool:
            return
        state.extensions_loading = True
        try:
            await get_api_client().execute_extension(state.selected_tool, ext_name, params)
            ui.notify("Action executed successfully", type='positive')
            ui.navigate.to('/')
        except APIError as e:
            ui.notify(f"Execution failed: {e}", type='negative')
        finally:
            state.extensions_loading = False
    
    ToolCard(
        tool=selected_tool_detail,
        on_query=on_query,
        on_mutate=on_mutate,
        on_execute=on_execute,
        loading=state.extensions_loading
    )
    logger.debug("_render_content completed")


async def _refresh_tools() -> None:
    """Refresh the tools list from the API."""
    logger.debug("_refresh_tools called")
    state = get_state()
    state.tools_loading = True
    state.error_message = None
    
    try:
        tools = await get_api_client().get_tools()
        state.tools = tools
        state.connection_status = 'connected'
        logger.debug(f"_refresh_tools got {len(tools)} tools")
        
        # NOTE: Extensions are now fetched on-demand when a tool is selected,
        # not during initial load. This prevents blocking the page render.
        # See _render_content() for on-demand extension fetching.
    except APIError as e:
        state.connection_status = 'error'
        state.error_message = str(e)
        ui.notify(f"Connection error: {e}", type='negative')
    except Exception as e:
        state.connection_status = 'error'
        ui.notify(f"Unexpected error: {e}", type='negative')
    finally:
        state.tools_loading = False
    logger.debug(f"_refresh_tools completed, tools: {len(state.tools)}")
    
    # Trigger UI refresh to update the tool list after loading completes
    # This is needed when refresh is triggered by the deferred timer
    ui.update()


# =============================================================================
# Uvicorn Support - Use ui.run_with() to attach NiceGUI to FastAPI app
# =============================================================================

logger.debug("Initializing NiceGUI with FastAPI")

# Create a FastAPI app and attach NiceGUI to it
from fastapi import FastAPI
fastapi_app = FastAPI(title='MCP Tools Management UI')

# Initialize NiceGUI with the FastAPI app
ui.run_with(
    fastapi_app,
    storage_secret=os.environ.get('MCP_UI_SECRET', 'change-this-secret-in-production')
)

# Export the FastAPI app for uvicorn
app = fastapi_app

logger.debug("NiceGUI initialization complete")


if __name__ in {'__main__', '__mp_main__'}:
    import uvicorn
    # Get port from config (ports.json or environment variable)
    port = _get_default_ui_port()
    uvicorn.run('mcp_ui.management_ui:app', host='0.0.0.0', port=port, log_level='info')
