"""
Tool List Component.

Renders the sidebar tool list with status indicators and icons.
"""

from nicegui import ui
from typing import List, Callable, Optional

from ..models import ToolInfo, ToolStatus
from ..logging_config import get_logger

logger = get_logger(__name__)

# Status color mapping
_STATUS_COLORS: dict[ToolStatus, str] = {
    ToolStatus.RUNNING: "green",
    ToolStatus.HEALTHY: "green",
    ToolStatus.STOPPED: "grey",
    ToolStatus.ERROR: "red",
    ToolStatus.UNHEALTHY: "red",
    ToolStatus.DEGRADED: "yellow",
    ToolStatus.UNKNOWN: "orange",
}


def _status_badge(status: ToolStatus) -> None:
    """Render status badge with color coding."""
    color = _STATUS_COLORS.get(status, "grey")
    ui.badge(status.value, color=color)


def _get_tool_icon(status: ToolStatus) -> str:
    """Get icon name for tool based on status."""
    if status == ToolStatus.RUNNING or status == ToolStatus.HEALTHY:
        return "check_circle"
    elif status == ToolStatus.ERROR or status == ToolStatus.UNHEALTHY:
        return "error"
    elif status == ToolStatus.DEGRADED:
        return "warning"
    elif status == ToolStatus.STOPPED:
        return "stop_circle"
    else:
        return "help_circle"


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
    with ui.column().classes("w-full"):
        # Header with refresh button
        with ui.row().classes("w-full justify-between items-center mb-2"):
            ui.label("Tools").classes("text-h6")
            if on_refresh:
                btn = ui.button(
                    icon="refresh",
                    on_click=on_refresh
                ).props("flat dense")
                btn.enabled = not loading

        if loading:
            ui.spinner().classes("mx-auto")
            return

        # Tool list
        if not tools:
            ui.label("No tools registered").classes("text-grey text-center p-4")
            return

        for tool in tools:
            _tool_list_item(tool, tool.name == selected_tool, on_select)


def _tool_list_item(
    tool: ToolInfo,
    is_selected: bool,
    on_select: Optional[Callable[[str], None]]
) -> None:
    """Render a single tool list item."""
    # Capture tool.name in a local variable to avoid closure issue
    tool_name = tool.name

    def handle_click() -> None:
        logger.info(f"button_click: select_tool tool={tool_name}")
        if on_select:
            on_select(tool_name)

    # Button styling based on selection
    button_classes = "w-full justify-between items-center"
    if is_selected:
        button_classes += " bg-blue-100 dark:bg-blue-900 border-l-4 border-blue-500"

    with ui.button(
        on_click=handle_click
    ).props("flat no-caps").classes(button_classes):
        with ui.row().classes("items-center gap-2"):
            ui.icon(_get_tool_icon(tool.status)).classes("text-gray-500")
            ui.label(tool.name).classes("font-medium")

        _status_badge(tool.status)
