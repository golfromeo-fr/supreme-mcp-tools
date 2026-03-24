"""
Tool List Component.

Renders the sidebar tool list with status indicators.
"""

from nicegui import ui
from typing import List, Callable, Optional

from ..models import ToolInfo, ToolStatus


def ToolList(
    tools: List[ToolInfo],
    selected_tool: Optional[str] = None,
    on_select: Optional[Callable[[str], None]] = None,
    on_refresh: Optional[Callable[[], None]] = None,
    loading: bool = False
) -> None:
    """
    Render the tool list sidebar.
    
    Args:
        tools: List of ToolInfo objects to display.
        selected_tool: Name of currently selected tool.
        on_select: Callback when a tool is selected.
        on_refresh: Callback when refresh button is clicked.
        loading: Whether to show loading spinner.
    """
    with ui.column().classes('w-full'):
        # Header with refresh button
        with ui.row().classes('w-full justify-between items-center mb-2'):
            ui.label('Tools').classes('text-h6')
            if on_refresh:
                btn = ui.button(
                    icon='refresh',
                    on_click=on_refresh
                ).props('flat dense')
                btn.enabled = not loading
        
        if loading:
            ui.spinner().classes('mx-auto')
            return
        
        # Tool list
        if not tools:
            ui.label('No tools registered').classes('text-grey text-center p-4')
            return
        
        with ui.list().classes('w-full'):
            for tool in tools:
                _tool_list_item(tool, tool.name == selected_tool, on_select)


def _tool_list_item(
    tool: ToolInfo,
    is_selected: bool,
    on_select: Optional[Callable[[str], None]]
) -> None:
    """Render a single tool list item."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Capture tool.name in a local variable to avoid closure issue
    tool_name = tool.name
    
    def handle_click():
        logger.debug(f"Tool clicked: {tool_name}")
        if on_select:
            logger.debug(f"Calling on_select with: {tool_name}")
            on_select(tool_name)
    
    # Use a button with flat styling for reliable click handling
    button_classes = 'w-full justify-between'
    if is_selected:
        button_classes += ' bg-blue-100 dark:bg-blue-900'
    
    with ui.button(
        on_click=handle_click
    ).props('flat').classes(button_classes):
        ui.label(tool.name).classes('font-medium')
        _status_badge(tool.status)


def _status_badge(status: ToolStatus) -> None:
    """Render status badge with color coding."""
    color_map = {
        ToolStatus.RUNNING: 'green',
        ToolStatus.STOPPED: 'grey',
        ToolStatus.ERROR: 'red',
        ToolStatus.UNKNOWN: 'orange',
        # Health-based statuses from ServiceRegistry
        ToolStatus.HEALTHY: 'green',
        ToolStatus.DEGRADED: 'yellow',
        ToolStatus.UNHEALTHY: 'red'
    }
    ui.badge(status.value, color=color_map.get(status, 'grey'))
