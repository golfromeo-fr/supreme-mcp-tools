"""
DB Connection Presets Panel.

Dedicated panel for databasemcp showing the numbered .env connection presets
(connection_presets data source, masked) with per-preset Connect /
Disconnect actions. Shows in the databasemcp Overview tab.
"""

from nicegui import ui
from typing import Any, Callable


def PresetsPanel(
    presets: list[dict[str, Any]],
    on_connect: Callable[[str], Any] | None = None,
    on_disconnect: Callable[[str], Any] | None = None,
    on_refresh: Callable[[], Any] | None = None,
) -> None:
    """
    Render the databasemcp connection presets panel.

    Args:
        presets: Rows from the connection_presets data source:
                 {number, dialect, name, desc, url (masked), autoconnect, connected}.
        on_connect: Callback with the preset number/alias.
        on_disconnect: Callback with the connection name.
        on_refresh: Callback when Refresh is clicked.
    """
    with ui.card().classes('w-full mb-4'):
        with ui.row().classes('w-full justify-between items-center mb-2'):
            ui.label('DB Connection Presets').classes('text-h6')
            if on_refresh:
                ui.button('Refresh', icon='refresh', on_click=on_refresh).props('flat dense')

        if not presets:
            ui.label(
                'No DB_PRESET_<NN> entries in .env '
                '(or the server predates the presets feature)'
            ).classes('text-grey text-sm')
            return

        for preset in presets:
            _render_preset_row(preset, on_connect, on_disconnect)


def _render_preset_row(
    preset: dict[str, Any],
    on_connect: Callable[[str], Any] | None,
    on_disconnect: Callable[[str], Any] | None,
) -> None:
    connection_name = preset.get('name') or preset.get('number', '?')
    desc = preset.get('desc') or preset.get('url', '')
    url = preset.get('url', '')

    with ui.row().classes('w-full justify-between items-center gap-2 py-1 flex-wrap'):
        with ui.row().classes('items-center gap-2'):
            ui.badge(preset.get('number', '?'), color='primary')
            ui.badge(preset.get('dialect', '?'), color='grey-8')
            if preset.get('autoconnect'):
                ui.badge('auto', color='amber-8').tooltip('DB_PRESET_AUTOCONNECT')
            ui.label(desc).classes('text-sm').tooltip(url)

        with ui.row().classes('items-center gap-2'):
            if preset.get('connected'):
                ui.badge('connected', color='positive')
                if on_disconnect:
                    ui.button(
                        'Disconnect',
                        icon='link_off',
                        on_click=lambda _, n=connection_name: on_disconnect(n),
                    ).props('flat dense')
            elif on_connect:
                ui.button(
                    'Connect',
                    icon='link',
                    on_click=lambda _, p=connection_name: on_connect(p),
                ).props('flat dense')
