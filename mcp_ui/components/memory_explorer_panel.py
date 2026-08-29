"""
Memory Explorer Panel.

Dedicated panel for memorymcp showing memories, search, and filters.
Shows at the top of the memorymcp tool detail view.
"""

from nicegui import ui
from typing import Any
from collections.abc import Callable


def MemoryExplorerPanel(
    memory_data: dict[str, Any] | None = None,
    on_search: Callable[[str, dict[str, Any]], None] | None = None,
    on_refresh: Callable[[], None] | None = None
) -> None:
    """
    Render memory explorer panel at the top of memorymcp tool detail view.

    Args:
        memory_data: Optional pre-fetched memory data/stats.
        on_search: Callback when search is submitted.
        on_refresh: Callback when refresh is clicked.
    """
    with ui.card().classes('w-full mb-4'):
        ui.label('Memory Explorer').classes('text-h6 mb-3')

        # Search + filters are only interactive when a callback is wired;
        # without one they used to render as dead controls.
        if on_search:
            # Search bar
            with ui.row().classes('w-full gap-2 mb-3'):
                search_input = ui.input(
                    'Search memories...',
                    placeholder='Query memories with recency weighting...'
                ).classes('flex-grow')
                ui.button(
                    'Search',
                    icon='search',
                    on_click=lambda: _do_search(search_input.value, on_search)
                ).props('color=primary')

            # Quick filter chips
            with ui.row().classes('items-center gap-2 mb-3'):
                ui.label('Quick filters:').classes('text-sm text-grey')
                _render_filter_chip('code_pattern', 'Code Pattern', on_search)
                _render_filter_chip('trick', 'Trick', on_search)
                _render_filter_chip('lesson', 'Lesson', on_search)
                _render_filter_chip('plan', 'Plan', on_search)

        # Memory types reference
        with ui.expansion('Memory Types Reference', icon='info_outline').classes('w-full mb-3'):
            _render_types_reference()

        # Memory metrics/stats if available
        if memory_data:
            _render_memory_stats(memory_data)

        # Refresh button
        if on_refresh:
            with ui.row().classes('mt-3'):
                ui.button(
                    'Refresh',
                    icon='refresh',
                    on_click=on_refresh
                ).props('flat dense')


def _render_filter_chip(type_value: str, label: str, on_search: Callable | None) -> None:
    """Render a clickable filter chip."""
    ui.chip(
        label,
        color='primary',
        on_click=lambda: on_search and on_search('', {'memory_type': type_value})
    ).classes('cursor-pointer')


def _render_types_reference() -> None:
    """Render memory types with descriptions."""
    types = [
        ('code_pattern', 'Useful coding idiom or pattern discovered'),
        ('architectural_decision', 'Why a particular approach was chosen'),
        ('trick', 'Clever workaround or unexpected solution'),
        ('plan', 'Project direction or roadmap'),
        ('lesson', 'Something that went wrong to avoid'),
        ('concept', 'General understanding or knowledge'),
    ]

    with ui.column().classes('gap-2'):
        for type_name, description in types:
            with ui.row().classes('items-center gap-2'):
                ui.label(f'{type_name}:').classes('font-bold text-sm')
                ui.label(description).classes('text-sm text-grey')


def _render_memory_stats(data: dict[str, Any]) -> None:
    """Render memory statistics."""
    with ui.card().classes('w-full bg-gray-100 dark:bg-gray-800 p-3'):
        ui.label('Memory Stats').classes('text-sm font-bold mb-2')

        total = data.get('total_memories', 0)
        by_type = data.get('by_type', {})

        with ui.row().classes('items-center gap-4'):
            ui.label(f'Total: {total}').classes('text-sm')
            ui.label(f'Types: {len(by_type)}').classes('text-sm')

        # Type breakdown
        if by_type:
            with ui.row().classes('flex-wrap gap-2 mt-2'):
                for mtype, count in by_type.items():
                    ui.chip(f'{mtype}: {count}').props('size=sm')


def _do_search(query: str, on_search: Callable | None) -> None:
    """Execute search with query and filters."""


def _do_search(query: str, on_search: Callable | None) -> None:
    """Execute search with query and filters."""
    if on_search:
        on_search(query, {})
