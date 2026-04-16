"""
Data Sources Box Component.

Renders read-only data sources panel with inline metrics display.
"""

from nicegui import ui
from typing import Any
from collections.abc import Callable

from ..models import Extension


def _get_extension_status(ext: Extension) -> tuple[str, str]:
    """
    Get the status and note for an extension.

    Returns (status_label, status_note) tuple.
    status_label is: "✅ Working", "⚠ Warning", or "❌ Not implemented"
    """
    if not ext.data:
        return "❓ Not queried", "Query to see data"

    # Check for cache_stats - if all hits/misses are 0, cache not implemented
    if ext.name == "cache_stats":
        hits = ext.data.get("hits", 0)
        misses = ext.data.get("misses", 0)
        if hits == 0 and misses == 0:
            return "⚠ No cache", "Cache not implemented (hits=0, misses=0)"

    # Check for request_stats - if no requests recorded
    if ext.name == "request_stats":
        total = ext.data.get("total_requests", 0)
        if total == 0:
            return "⚠ No requests", "No requests recorded yet"

    # Check for api_response_times - if min/max/avg are all 0
    if ext.name == "api_response_times":
        min_t = ext.data.get("min_time_ms", 0)
        max_t = ext.data.get("max_time_ms", 0)
        avg_t = ext.data.get("avg_time_ms", 0)
        if min_t == 0 and max_t == 0 and avg_t == 0:
            return "⚠ No timing", "No timing data (all values 0)"

    # All other extensions with data are considered implemented
    return "✅ Working", ""


def DataSourcesBox(
    extensions: list[Extension],
    on_query: Callable[[str], None] | None = None,
    on_refresh: Callable[[str], None] | None = None
) -> None:
    """
    Render read-only data sources box with inline metrics.

    Args:
        extensions: List of Extension objects (data sources).
        on_query: Callback when querying a data source.
        on_refresh: Callback when refreshing a data source.
    """
    with ui.card().classes('w-full'):
        ui.label('Data Sources').classes('text-h6 mb-2')

        if not extensions:
            ui.label('No data sources available').classes('text-grey')
            return

        for ext in extensions:
            # Build summary text for the expansion header
            summary = _build_summary(ext)
            status_label, status_note = _get_extension_status(ext)

            with ui.expansion(f"{summary} [{status_label}]", icon='storage').classes('w-full'):
                # Show status note below header
                if status_note:
                    ui.label(status_note).classes('text-caption mb-2')

                if ext.description:
                    ui.label(ext.description).classes('text-grey mb-2')

                if ext.data:
                    # Show inline summary cards for key metrics (pass category from metadata)
                    category = ext.metadata.get('category') if ext.metadata else None
                    _inline_metrics(ext.data, category=category)
                    # Show full data table
                    ui.separator()
                    _data_table(ext.data)
                else:
                    ui.label('No data available').classes('text-grey')

                with ui.row().classes('gap-2 mt-2'):
                    if on_query:
                        ui.button(
                            'Query',
                            icon='refresh',
                            on_click=lambda e=ext.name: on_query(e) if on_query else None
                        ).props('flat dense')
                    if on_refresh:
                        ui.button(
                            'Refresh',
                            icon='refresh',
                            on_click=lambda e=ext.name: on_refresh(e) if on_refresh else None
                        ).props('flat dense')


def _build_summary(ext: Extension) -> str:
    """
    Build a summary string for the expansion header based on available data.

    Args:
        ext: Extension object with data.

    Returns:
        Summary string for display in header.
    """
    if not ext.data:
        return ext.name

    # Special handling for collections category - show count of collections
    if hasattr(ext, 'metadata') and ext.metadata and ext.metadata.get('category') == 'collections':
        if 'total' in ext.data:
            return f"{ext.name} ({ext.data['total']} collections)"
        return ext.name

    # Look for common metric keys to display inline
    summary_parts = [ext.name]

    # Try to find total/count metrics
    if 'total' in ext.data:
        summary_parts.append(f"total: {ext.data['total']}")
    elif 'total_tool_calls' in ext.data:
        summary_parts.append(f"{ext.data['total_tool_calls']} calls")
    elif 'count' in ext.data:
        summary_parts.append(f"count: {ext.data['count']}")
    elif 'avg_time_ms' in ext.data:
        summary_parts.append(f"avg: {ext.data['avg_time_ms']}ms")

    return f"{summary_parts[0]} ({', '.join(summary_parts[1:])})" if len(summary_parts) > 1 else summary_parts[0]


def _render_collections_data(data: dict[str, Any]) -> None:
    """
    Render collections data in a user-friendly table format.

    Args:
        data: Dictionary with collection names as keys and stats as values.
    """
    # Filter out 'total' from collections display
    collections = {k: v for k, v in data.items() if k != 'total'}

    if not collections:
        ui.label('No collections indexed').classes('text-grey')
        return

    # Render as a table with Collection | Stats columns
    rows = [{'collection': k, 'stats': v} for k, v in collections.items()]
    ui.table(
        columns=[
            {'name': 'collection', 'label': 'Collection', 'field': 'collection'},
            {'name': 'stats', 'label': 'Stats', 'field': 'stats'}
        ],
        rows=rows,
        row_key='collection'
    ).classes('w-full').props('flat dense')


def _inline_metrics(data: dict[str, Any], category: str = None) -> None:
    """
    Render key metrics as inline chips/badges for quick visibility.

    Args:
        data: Dictionary of metric data.
        category: Optional category to determine rendering style.
    """
    # Special rendering for collections
    if category == 'collections':
        _render_collections_data(data)
        return

    # Show key metrics as chips in a row
    with ui.row().classes('gap-2 flex-wrap'):
        for key, value in data.items():
            if value is not None:
                ui.badge(
                    f"{key}: {value}",
                    color=_get_metric_color(key)
                ).classes('text-caption')


def _get_metric_color(key: str) -> str:
    """
    Get a color for a metric badge based on its name.
    
    Args:
        key: Metric key name.
    
    Returns:
        Color string for the badge.
    """
    key_lower = key.lower()
    if 'count' in key_lower or 'total' in key_lower:
        return 'primary'
    elif 'time' in key_lower or 'ms' in key_lower:
        return 'secondary'
    elif 'rate' in key_lower or 'percent' in key_lower:
        return 'accent'
    elif 'min' in key_lower or 'max' in key_lower:
        return 'info'
    elif 'error' in key_lower or 'fail' in key_lower:
        return 'negative'
    elif 'success' in key_lower:
        return 'positive'
    else:
        return 'grey'


def _data_table(data: dict[str, Any]) -> None:
    """Render data as a key-value table."""
    rows = [{'key': k, 'value': str(v)} for k, v in data.items()]
    
    if not rows:
        ui.label('No data').classes('text-grey')
        return
    
    ui.table(
        columns=[
            {'name': 'key', 'label': 'Property', 'field': 'key'},
            {'name': 'value', 'label': 'Value', 'field': 'value'}
        ],
        rows=rows,
        row_key='key'
    ).classes('w-full').props('flat dense')
