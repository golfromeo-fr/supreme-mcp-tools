"""
Data Sources Box Component.

Renders read-only data sources panel with inline metrics display.
"""

import json

from nicegui import ui
from typing import Any
from collections.abc import Callable

from ..models import Extension


def _get_extension_status(ext: Extension) -> tuple[str, str]:
    """
    Get the status and note for an extension.

    Returns (status_label, status_note) tuple. status_label is a plain-text
    short state (emoji-free; badges handle color in the UI).
    """
    if not ext.data:
        return "Not queried", "Query to see data"

    # Check for cache_stats - if all hits/misses are 0, cache not implemented
    if ext.name == "cache_stats":
        hits = ext.data.get("hits", 0)
        misses = ext.data.get("misses", 0)
        if hits == 0 and misses == 0:
            return "No cache", "Cache not implemented (hits=0, misses=0)"

    # Check for request_stats - if no requests recorded
    if ext.name == "request_stats":
        total = ext.data.get("total_requests", 0)
        if total == 0:
            return "No requests", "No requests recorded yet"

    # Check for api_response_times - if min/max/avg are all 0
    if ext.name == "api_response_times":
        min_t = ext.data.get("min_time_ms", 0)
        max_t = ext.data.get("max_time_ms", 0)
        avg_t = ext.data.get("avg_time_ms", 0)
        if min_t == 0 and max_t == 0 and avg_t == 0:
            return "No timing", "No timing data (all values 0)"

    # All other extensions with data are considered implemented
    return "Working", ""


def DataSourcesBox(
    extensions: list[Extension],
    on_query: Callable[[str], None] | None = None,
    on_refresh: Callable[[str], None] | None = None
) -> None:
    """
    Render read-only data sources with inline metrics.

    Renders directly into the enclosing container (the Extensions tab) —
    no outer card or section header. One expansion per data source.

    Args:
        extensions: List of Extension objects (data sources).
        on_query: Callback when querying a data source.
        on_refresh: Deprecated duplicate of on_query; ignored.
    """
    if not extensions:
        ui.label('No data sources available').classes('text-grey')
        return

    for ext in extensions:
        summary = _build_summary(ext)
        _status_label, status_note = _get_extension_status(ext)

        with ui.expansion(f"{summary} — {ext.name}", icon='storage').classes('w-full'):
            if status_note:
                ui.label(status_note).classes('text-caption mb-2')

            if ext.description:
                ui.label(ext.description).classes('text-grey mb-2')

            if ext.data:
                # Show inline summary cards for key metrics (pass category from metadata)
                category = ext.metadata.get('category') if ext.metadata else None
                _inline_metrics(ext.data, category=category)
                # Bar charts for nested numeric series (e.g. by_tool, by_endpoint)
                _maybe_charts(ext.data)
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


def _maybe_charts(data: dict[str, Any], max_charts: int = 2) -> None:
    """
    Render bar charts for nested numeric dict series (e.g. by_tool, by_endpoint).

    Only dicts with at least two numeric values are charted, capped at
    ``max_charts`` so a data source with many series stays readable.
    """
    charted = 0
    for key, value in data.items():
        if charted >= max_charts:
            break
        if not isinstance(value, dict) or len(value) < 2:
            continue
        numeric = {
            k: v for k, v in value.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if len(numeric) < 2:
            continue
        ui.label(key).classes("text-caption text-grey mt-1")
        ui.echart({
            "xAxis": {
                "type": "category",
                "data": list(numeric.keys()),
                "axisLabel": {"rotate": 30, "fontSize": 10},
            },
            "yAxis": {"type": "value"},
            "series": [{"type": "bar", "data": list(numeric.values())}],
            "tooltip": {"trigger": "axis"},
            "grid": {"containLabel": True},
        }).classes("w-full h-48")
        charted += 1


def _format_scalar(value: Any) -> str:
    """Render a value as display text; pretty-print nested structures."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _data_table(data: dict[str, Any]) -> None:
    """Render data as a key-value table; nested structures as JSON blocks."""
    rows = [
        {'key': k, 'value': _format_scalar(v)}
        for k, v in data.items()
        if not isinstance(v, (dict, list)) or not v
    ]

    if rows:
        ui.table(
            columns=[
                {'name': 'key', 'label': 'Property', 'field': 'key'},
                {'name': 'value', 'label': 'Value', 'field': 'value'}
            ],
            rows=rows,
            row_key='key'
        ).classes('w-full').props('flat dense')

    # Non-empty nested structures: pretty JSON in collapsible code blocks
    # (Quasar table cells don't render newlines, so indent=2 would collapse).
    nested = {
        k: v for k, v in data.items()
        if isinstance(v, (dict, list)) and v
    }
    for key, value in nested.items():
        with ui.expansion(key, icon="data_object").classes("w-full"):
            ui.code(json.dumps(value, indent=2, default=str)).classes("w-full")
