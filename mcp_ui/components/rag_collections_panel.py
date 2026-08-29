"""
RAG MCP Collections Panel.

Dedicated panel for ragmcp showing indexed collections with stats.
Shows in the ragmcp Overview tab.
"""

import re

from nicegui import ui
from typing import Any
from collections.abc import Callable


def RagCollectionsPanel(
    collections_data: dict[str, Any],
    indexing_progress: dict[str, Any] = None,
    on_refresh: Callable[[], None] | None = None
) -> None:
    """
    Render ragmcp collections panel at the top of tool detail view.

    Args:
        collections_data: Dictionary with collection names as keys and stats as values.
                          Expected format: {'fastapi-code': '124 chunks @ 1536d', ...}
        indexing_progress: Optional indexing progress data.
        on_refresh: Callback when refresh is clicked.
    """
    with ui.card().classes('w-full mb-4'):
        ui.label('Indexed Collections').classes('text-h6 mb-3')

        if not collections_data:
            ui.label('No collections indexed yet').classes('text-grey')
            return

        # Filter out 'total' from collections display
        collections = {k: v for k, v in collections_data.items() if k != 'total'}
        total = collections_data.get('total', len(collections))

        # Show indexing progress if available
        if indexing_progress:
            _render_indexing_progress(indexing_progress)

        # Show collections count
        with ui.row().classes('items-center gap-2 mb-3'):
            ui.icon('folder', size='sm').classes('text-primary')
            ui.label(f'{total} collection(s) indexed').classes('text-sm')

        if not collections:
            ui.label('No collections indexed yet').classes('text-grey')
            return

        _render_collection_rows(collections)

        if on_refresh:
            with ui.row().classes('mt-3'):
                ui.button(
                    'Refresh',
                    icon='refresh',
                    on_click=on_refresh
                ).props('flat dense')


def _chunks_of(stats: Any) -> int | None:
    """Extract the chunk count from a '124 chunks @ 1536d' style stats string."""
    match = re.search(r"(\d+)\s+chunks", str(stats))
    return int(match.group(1)) if match else None


def _render_collection_rows(collections: dict[str, Any]) -> None:
    """Render collections as labeled progress bars sized against the largest."""
    counts = {k: _chunks_of(v) for k, v in collections.items()}
    max_count = max((c for c in counts.values() if c is not None), default=None)

    with ui.column().classes('w-full gap-2'):
        for name, stats in collections.items():
            count = counts[name]
            with ui.column().classes('w-full gap-1'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label(name).classes('font-mono text-sm')
                    ui.label(str(stats)).classes('text-caption text-grey')
                if count is not None and max_count:
                    ui.linear_progress(value=count / max_count, show_value=False).classes('w-full')


def _render_indexing_progress(progress: dict[str, Any]) -> None:
    """Render indexing progress if there's an active indexing job."""
    status = progress.get('status', '')

    if status == 'running':
        with ui.card().classes('w-full bg-blue-50 dark:bg-blue-900 p-3 mb-3'):
            with ui.row().classes('items-center gap-2'):
                ui.spinner(size='sm')
                ui.label('Indexing in progress...').classes('text-sm')
            if 'progress' in progress and progress['progress']:
                ui.label(progress['progress']).classes('text-xs mt-1')
    elif status == 'completed_stopped' or status == 'completed':
        # Show last indexing info
        with ui.card().classes('w-full bg-green-50 dark:bg-green-900 p-3 mb-3'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('check_circle', size='sm').classes('text-green')
                ui.label('Indexing finished').classes('text-sm')
            if 'workspace' in progress:
                ui.label(f"Workspace: {progress['workspace']}").classes('text-xs mt-1')
            if 'collection' in progress:
                ui.label(f"Collection: {progress['collection']}").classes('text-xs')
            if 'runtime' in progress:
                ui.label(f"Duration: {progress['runtime']}").classes('text-xs')
    elif status == 'incomplete':
        # Show warning that indexing didn't complete
        with ui.card().classes('w-full bg-yellow-50 dark:bg-yellow-900 p-3 mb-3'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('warning', size='sm').classes('text-yellow')
                ui.label('Indexing incomplete').classes('text-sm')
            if 'progress' in progress and progress['progress']:
                ui.label(f"Last progress: {progress['progress']}").classes('text-xs mt-1')
            if 'workspace' in progress:
                ui.label(f"Workspace: {progress['workspace']}").classes('text-xs')
            if 'runtime' in progress:
                ui.label(f"Duration: {progress['runtime']}").classes('text-xs')
    elif status == 'no_active_indexing':
        pass  # Don't show anything
    else:
        # Unknown status
        with ui.card().classes('w-full bg-gray-100 dark:bg-gray-800 p-3 mb-3'):
            ui.label(f'Indexing status: {status}').classes('text-xs')
