"""
Tool Card Component.

Renders tool detail card with extensions.
"""

from nicegui import ui
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime

from ..models import ToolDetail, ToolStatus, Extension, ExtensionType
from .data_sources_box import DataSourcesBox
from .mutators_box import MutatorsBox
from .actions_box import ActionsBox


def ToolCard(
    tool: Optional[ToolDetail],
    on_query: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_mutate: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    on_execute: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    loading: bool = False
) -> None:
    """
    Render tool detail card with extensions.
    
    Args:
        tool: ToolDetail object with full information.
        on_query: Callback when querying a data source.
        on_mutate: Callback when submitting a mutator.
        on_execute: Callback when executing an action.
        loading: Whether to show loading spinner.
    """
    if tool is None:
        _render_empty_state()
        return
    
    with ui.card().classes('w-full'):
        # Tool header
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label(tool.name).classes('text-h5')
            _status_badge(tool.status)
        
        if loading:
            ui.spinner().classes('mx-auto')
            return
        
        # Always show tool info section
        _render_tool_info(tool)
        
        # Show extensions or placeholder message
        if not tool.extensions:
            with ui.card().classes('w-full mt-4 bg-gray-100 dark:bg-gray-800'):
                ui.label('No Extensions Registered').classes('text-h6 mb-2')
                ui.label(
                    'This tool does not have any extensions registered through the Management API. '
                    'Extensions must be registered by the tool process to appear here.'
                ).classes('text-grey text-sm')
                ui.label(
                    'The tool is still operational as an MCP server.'
                ).classes('text-grey text-sm mt-2')
            return
        
        # Separate extensions by type
        data_sources = [e for e in tool.extensions if e.type == ExtensionType.DATA_SOURCE]
        mutators = [e for e in tool.extensions if e.type == ExtensionType.MUTATOR]
        actions = [e for e in tool.extensions if e.type == ExtensionType.ACTION]
        
        # Two-column grid for data sources and mutators
        if data_sources or mutators:
            with ui.grid().classes('w-full grid-cols-2 gap-4'):
                if data_sources:
                    DataSourcesBox(data_sources, on_query=on_query)
                
                if mutators:
                    MutatorsBox(mutators, on_submit=on_mutate, loading=loading)
        
        # Actions (full width)
        if actions:
            ActionsBox(actions, on_execute=on_execute, loading=loading)


def _render_empty_state() -> None:
    """Render empty state when no tool is selected."""
    ui.label('Select a tool from the sidebar').classes(
        'text-grey text-center p-8 text-h6'
    )


def _status_badge(status: ToolStatus) -> None:
    """Render status badge."""
    color_map = {
        ToolStatus.RUNNING: 'green',
        ToolStatus.STOPPED: 'grey',
        ToolStatus.ERROR: 'red',
        ToolStatus.UNKNOWN: 'orange',
        ToolStatus.HEALTHY: 'green',
        ToolStatus.DEGRADED: 'yellow',
        ToolStatus.UNHEALTHY: 'red',
    }
    ui.badge(status.value, color=color_map.get(status, 'grey'))


def _render_tool_info(tool: ToolDetail) -> None:
    """Render tool information section."""
    with ui.card().classes('w-full mb-4 bg-gray-50 dark:bg-gray-700'):
        ui.label('Tool Information').classes('text-h6 mb-2')
        
        # MCP Endpoint
        if tool.mcp_port:
            with ui.row().classes('items-center gap-2 mb-1'):
                ui.icon('link', size='sm').classes('text-blue-500')
                ui.label('MCP Endpoint:').classes('text-sm text-gray-500')
                ui.label(f'http://localhost:{tool.mcp_port}/mcp').classes('text-sm text-blue-500')
        
        # Management URL
        if tool.management_url:
            with ui.row().classes('items-center gap-2 mb-1'):
                ui.icon('settings', size='sm').classes('text-gray-500')
                ui.label('Management:').classes('text-sm text-gray-500')
                ui.label(tool.management_url).classes('text-sm')
        
        # Capabilities
        if tool.capabilities:
            with ui.column().classes('gap-1 mt-2'):
                ui.label('Capabilities:').classes('text-sm font-medium')
                for key, value in tool.capabilities.items():
                    with ui.row().classes('items-center gap-1'):
                        ui.label(f'{key}:').classes('text-xs text-gray-500')
                        ui.label(str(value)).classes('text-xs')
