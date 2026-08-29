"""
Tool Overview Component.

Renders the Overview tab: tool identity, status, endpoint info, and the
tool-specific panels (ragmcp collections, memorymcp explorer).
"""

from nicegui import ui

from .constants import STATUS_COLORS
from ..models import ToolDetail, ToolStatus, Extension
from ..logging_config import get_logger

logger = get_logger(__name__)


def ToolOverview(tool: ToolDetail) -> None:
    """
    Render the tool overview: header, status, info, special panels.

    Args:
        tool: ToolDetail object with full information.
    """
    logger.debug(f"component: ToolOverview tool={tool.name}")

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full justify-between items-center mb-2"):
            ui.label(tool.name).classes("text-h5 font-bold")
            _status_badge(tool.status)
        _render_tool_info(tool)

    # Tool-specific panels when their extension data is available
    if tool.name == "ragmcp":
        _render_rag_collections_panel(tool.extensions)
    elif tool.name == "memorymcp":
        _render_memory_explorer_panel(tool.extensions)


def render_empty_state() -> None:
    """Render empty state when no tool is selected."""
    with ui.column().classes("w-full items-center justify-center p-8"):
        ui.icon("info", size="48px").classes("text-gray-400 mb-2")
        ui.label("Select a tool from the sidebar").classes("text-gray-500")


def render_loading_state(message: str = "Loading tool details...") -> None:
    """Render loading state while tool data is being fetched."""
    with ui.column().classes("w-full items-center justify-center p-8"):
        ui.spinner("dots", size="48px").classes("text-primary mb-4")
        ui.label(message).classes("text-gray-500")


def _status_badge(status: ToolStatus) -> None:
    """Render status badge with color coding."""
    color = STATUS_COLORS.get(status, "grey")
    ui.badge(status.value.upper(), color=color)


def _render_tool_info(tool: ToolDetail) -> None:
    """Render tool information section."""
    with ui.column().classes("w-full gap-2"):
        if tool.mcp_port:
            with ui.row().classes("items-center gap-2"):
                ui.icon("link", size="sm").classes("text-blue-500")
                ui.label("MCP Endpoint:").classes("text-sm text-gray-500")
                ui.label(f"http://127.0.0.1:{tool.mcp_port}/mcp").classes(
                    "text-sm text-blue-500 font-mono"
                )

        if tool.management_url:
            with ui.row().classes("items-center gap-2"):
                ui.icon("settings", size="sm").classes("text-gray-500")
                ui.label("Management:").classes("text-sm text-gray-500")
                ui.label(tool.management_url).classes("text-sm font-mono")

        if tool.extension_count > 0:
            with ui.row().classes("items-center gap-2"):
                ui.icon("extension", size="sm").classes("text-purple-500")
                ui.label("Extensions:").classes("text-sm text-gray-500")
                summary = tool.get_extension_summary()
                parts = []
                if summary["data_sources"]:
                    parts.append(f"{summary['data_sources']} data sources")
                if summary["actions"]:
                    parts.append(f"{summary['actions']} actions")
                if summary["events"]:
                    parts.append(f"{summary['events']} events")
                if summary["streams"]:
                    parts.append(f"{summary['streams']} streams")
                ui.label(", ".join(parts)).classes("text-sm")


def _render_rag_collections_panel(extensions: list[Extension]) -> None:
    """Render ragmcp-specific collections panel when collection data exists."""
    from .rag_collections_panel import RagCollectionsPanel

    list_collections_data = None
    check_indexing_progress_data = None

    for ext in extensions:
        if ext.name == "list_collections" and ext.data:
            list_collections_data = ext.data
        elif ext.name == "check_indexing_progress" and ext.data:
            check_indexing_progress_data = ext.data

    if list_collections_data:
        RagCollectionsPanel(
            collections_data=list_collections_data,
            indexing_progress=check_indexing_progress_data
        )


def _render_memory_explorer_panel(extensions: list[Extension]) -> None:
    """Render memorymcp-specific memory explorer panel when stats exist."""
    from .memory_explorer_panel import MemoryExplorerPanel

    memory_data = None
    for ext in extensions:
        if ext.name == "memory_stats" and ext.data:
            memory_data = ext.data

    MemoryExplorerPanel(memory_data=memory_data)
