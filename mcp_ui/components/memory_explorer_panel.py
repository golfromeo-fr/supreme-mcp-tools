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
    if on_search:
        on_search(query, {})


def MemoryUpsertForm(
    on_submit: Callable[[dict[str, Any]], None] | None = None
) -> None:
    """
    Render a form to quickly store a new memory.

    Args:
        on_submit: Callback when submit is clicked with memory data.
    """
    memory_text = None
    memory_type = None
    memory_tags = None

    with ui.card().classes('w-full'):
        ui.label('Quick Store Memory').classes('text-h6 mb-3')

        with ui.input('What did you learn?', placeholder='Describe the memory...'
                     ).classes('w-full mb-2') as text_input:
            memory_text = text_input

        with ui.select(
            'Memory Type',
            options=[
                ('code_pattern', 'Code Pattern'),
                ('trick', 'Trick'),
                ('lesson', 'Lesson'),
                ('plan', 'Plan'),
                ('architectural_decision', 'Architectural Decision'),
                ('concept', 'Concept'),
            ],
            value='concept'
        ).classes('w-full mb-2') as type_select:
            memory_type = type_select

        with ui.input('Tags (comma-separated)', placeholder='pattern, api, auth'
                     ).classes('w-full mb-3') as tags_input:
            memory_tags = tags_input

        ui.button(
            'Store Memory',
            icon='save',
            on_click=lambda: _submit_memory(
                text_input.value,
                type_select.value,
                tags_input.value,
                on_submit
            )
        ).props('color=primary')


def _submit_memory(
    text: str,
    mtype: str,
    tags_str: str,
    on_submit: Callable | None
) -> None:
    """Submit a new memory."""
    if not text:
        ui.notify('Please enter memory text', type='warning')
        return

    tags = [t.strip() for t in tags_str.split(',')] if tags_str else []

    if on_submit:
        on_submit({
            'text': text,
            'memory_type': mtype,
            'tags': tags,
        })

    ui.notify('Memory stored!', type='positive')


def MemoryDetailView(
    memory: dict[str, Any],
    on_close: Callable[[], None] | None = None
) -> None:
    """
    Render detailed view of a single memory.

    Args:
        memory: Memory data dictionary.
        on_close: Callback when close button is clicked.
    """
    with ui.dialog().classes('w-full') as dialog, ui.card().classes('w-full'):
        ui.label('Memory Detail').classes('text-h6 mb-3')

        # Type and metadata
        with ui.row().classes('items-center gap-2 mb-2'):
            ui.chip(memory.get('memory_type', 'unknown')).props('color=primary')
            if memory.get('tags'):
                for tag in memory.get('tags', []):
                    ui.chip(tag).props('size=sm')

        # Text content
        ui.label('Content:').classes('text-sm font-bold')
        ui.label(memory.get('text', '')).classes('text-sm mb-3')

        # Metadata
        with ui.row().classes('gap-4 mb-2'):
            ui.label(f"Created: {memory.get('created_at', 'unknown')}").classes('text-xs text-grey')
            ui.label(f"Accessed: {memory.get('last_accessed', 'never')}").classes('text-xs text-grey')
            ui.label(f"Usage: {memory.get('usage_count', 0)}x").classes('text-xs text-grey')

        # Source if available
        if memory.get('source'):
            ui.label(f"Source: {memory.get('source')}").classes('text-xs text-grey mb-2')

        # Provenance history
        provenance = memory.get('provenance', {})
        if provenance and isinstance(provenance, dict):
            history = provenance.get('history', [])
            if history:
                with ui.expansion('Provenance History', icon='history').classes('w-full mt-2'):
                    for entry in history:
                        ui.label(f"{entry.get('timestamp', '')} - {entry.get('source', '')}").classes('text-xs')

        # Close button
        with ui.row().classes('justify-end mt-3'):
            ui.button('Close', on_click=dialog.close).props('flat')