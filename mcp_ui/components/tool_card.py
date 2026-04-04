"""
Tool Card Component.

Renders tool detail card with extensions in a two-column layout.
"""

from nicegui import ui
from typing import List, Dict, Any, Callable, Optional

from ..models import ToolDetail, ToolStatus, Extension, ExtensionType, EnvVariable
from ..logging_config import get_logger
from .data_sources_box import DataSourcesBox
from .mutators_box import MutatorsBox
from .actions_box import ActionsBox
from .env_var_editor import EnvVarEditor

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


def ToolCard(
    tool: Optional[ToolDetail],
    on_query: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_mutate: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_execute: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    loading: bool = False,
    env_variables: Optional[List[EnvVariable]] = None,
    on_env_update: Optional[Callable[[str, str, str], None]] = None,
    on_env_delete: Optional[Callable[[str, str], None]] = None,
    current_mutator_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Render tool detail card with extensions.

    Args:
        tool: ToolDetail object with full information.
        on_query: Callback when querying a data source.
        on_mutate: Callback when submitting a mutator.
        on_execute: Callback when executing an action.
        loading: Whether to show loading spinner.
        env_variables: List of environment variables.
        on_env_update: Callback when updating an env var.
        on_env_delete: Callback when deleting an env var.
        current_mutator_values: Current values for mutators (e.g. api_key_info).
    """
    logger.debug(f"component: ToolCard tool={tool.name if tool else None} loading={loading}")

    if tool is None:
        if loading:
            _render_loading_state()
        else:
            _render_empty_state()
        return

    with ui.card().classes("w-full") as card:
        # Header with name and status
        with ui.row().classes("w-full justify-between items-center mb-4"):
            ui.label(tool.name).classes("text-h5 font-bold")
            _status_badge(tool.status)

        if loading:
            with ui.row().classes("w-full justify-center items-center gap-4 p-4"):
                ui.spinner()
                ui.label(f"Loading {tool.name}...").classes("text-grey")
            return card

        # Tool info section
        _render_tool_info(tool)

        # Show extensions or placeholder message
        if not tool.extensions:
            with ui.card().classes("w-full mt-4 bg-gray-100 dark:bg-gray-800"):
                ui.label("No Extensions Registered").classes("text-h6 mb-2")
                ui.label(
                    "This tool does not have any extensions registered through the Management API."
                ).classes("text-grey text-sm")
            return card

        # Extensions in collapsible sections
        _render_extensions(
            tool,
            on_query,
            on_mutate,
            on_execute,
            current_mutator_values,
            env_variables,
            tool.name,
            on_env_update,
            on_env_delete,
        )


def _render_empty_state() -> None:
    """Render empty state when no tool is selected."""
    with ui.column().classes("w-full items-center justify-center p-8"):
        ui.icon("info", size="48px").classes("text-gray-400 mb-2")
        ui.label("Select a tool from the sidebar").classes("text-gray-500")


def _render_loading_state() -> None:
    """Render loading state while tool data is being fetched."""
    with ui.column().classes("w-full items-center justify-center p-8"):
        ui.spinner("dots", size="48px").classes("text-primary mb-4")
        ui.label("Loading tool details...").classes("text-gray-500")


def _status_badge(status: ToolStatus) -> None:
    """Render status badge with color coding."""
    color = _STATUS_COLORS.get(status, "grey")
    ui.badge(status.value.upper(), color=color)


def _render_tool_info(tool: ToolDetail) -> None:
    """Render tool information section."""
    with ui.column().classes("w-full gap-2 mb-4"):
        ui.label("Tool Information").classes("text-subtitle1 font-medium mb-2")

        # MCP Port
        if tool.mcp_port:
            with ui.row().classes("items-center gap-2"):
                ui.icon("link", size="sm").classes("text-blue-500")
                ui.label("MCP Endpoint:").classes("text-sm text-gray-500")
                ui.label(f"http://localhost:{tool.mcp_port}/mcp").classes("text-sm text-blue-500 font-mono")

        # Management URL
        if tool.management_url:
            with ui.row().classes("items-center gap-2"):
                ui.icon("settings", size="sm").classes("text-gray-500")
                ui.label("Management:").classes("text-sm text-gray-500")
                ui.label(tool.management_url).classes("text-sm font-mono")

        # Extension count
        if tool.extension_count > 0:
            with ui.row().classes("items-center gap-2"):
                ui.icon("extension", size="sm").classes("text-purple-500")
                ui.label("Extensions:").classes("text-sm text-gray-500")
                summary = tool.get_extension_summary()
                parts = []
                if summary["data_sources"]:
                    parts.append(f"{summary['data_sources']} data sources")
                if summary["mutators"]:
                    parts.append(f"{summary['mutators']} mutators")
                if summary["actions"]:
                    parts.append(f"{summary['actions']} actions")
                ui.label(", ".join(parts)).classes("text-sm")


def _render_extensions(
    tool: ToolDetail,
    on_query: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_mutate: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_execute: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    current_mutator_values: Optional[Dict[str, Dict[str, Any]]] = None,
    env_variables: Optional[List[EnvVariable]] = None,
    tool_name: Optional[str] = None,
    on_env_update: Optional[Callable[[str, str, str], None]] = None,
    on_env_delete: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Render collapsible extension sections."""
    # Separate extensions by type
    data_sources = [e for e in tool.extensions if e.type == ExtensionType.DATA_SOURCE]
    mutators = [e for e in tool.extensions if e.type == ExtensionType.MUTATOR]
    actions = [e for e in tool.extensions if e.type == ExtensionType.ACTION]

    with ui.column().classes("w-full gap-4"):
        # Data Sources (READ) - wrap in expansion for grouping
        if data_sources:
            with ui.expansion("Data Sources (READ)", icon="storage", value=False).classes("w-full"):
                DataSourcesBox(
                    extensions=data_sources,
                    on_query=on_query,
                    on_refresh=on_query
                )

        # Mutators (WRITE) - wrap in expansion for grouping
        if mutators:
            with ui.expansion("Configuration (WRITE)", icon="edit", value=False).classes("w-full"):
                MutatorsBox(
                    extensions=mutators,
                    on_submit=on_mutate,
                    current_values=current_mutator_values,
                    env_variables=env_variables,
                    tool_name=tool_name,
                    on_env_update=on_env_update,
                    on_env_delete=on_env_delete,
                )

        # Actions - wrap in expansion for grouping
        if actions:
            with ui.expansion("Actions", icon="bolt", value=False).classes("w-full"):
                ActionsBox(
                    extensions=actions,
                    on_execute=on_execute
                )
