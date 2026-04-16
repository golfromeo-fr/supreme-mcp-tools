"""
Actions Box Component.

Renders actions panel with execute buttons.
"""

from nicegui import ui
from typing import Any
from collections.abc import Callable

from ..models import Extension
from .mutators_box import _generate_form


def _get_action_status(ext: Extension) -> tuple[str, str]:
    """
    Get status for an action extension.

    Returns (status_label, status_note) tuple.
    """
    input_props = ext.json_schema.get('input', {}).get('properties', {})

    # If no input properties, action may not need parameters
    if not input_props:
        return "✅ Ready", "No parameters required"

    # Check if it has description - lack of description might mean not fully implemented
    if not ext.description:
        return "⚠️ Unknown", "No description - may not be implemented"

    return "✅ Ready", ""


def ActionsBox(
    extensions: list[Extension],
    on_execute: Callable[[str, dict[str, Any]], None] | None = None,
    loading: bool = False
) -> None:
    """
    Render actions box with execute buttons.

    Args:
        extensions: List of Extension objects (actions).
        on_execute: Callback when executing an action.
        loading: Whether to show loading state.
    """
    with ui.card().classes('w-full'):
        ui.label('Actions').classes('text-h6 mb-2')

        if not extensions:
            ui.label('No actions available').classes('text-grey')
            return

        for ext in extensions:
            status_label, status_note = _get_action_status(ext)

            with ui.expansion(f"{ext.name} [{status_label}]", icon='play_arrow').classes('w-full'):
                # Show status note
                if status_note:
                    ui.label(status_note).classes('text-caption mb-2')

                if ext.description:
                    ui.label(ext.description).classes('text-grey mb-2')

                # Generate form from schema
                form_values = _generate_form(ext.json_schema)

                # Execute button with confirmation
                execute_btn = ui.button(
                    'Execute',
                    icon='play_arrow',
                    color='primary',
                    on_click=lambda e=ext.name, v=form_values: (
                        _confirm_and_execute(e, v, on_execute)
                    )
                )
                if loading:
                    execute_btn.disable()


def _generate_action_form(schema: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    """Generate form for action parameters."""
    return _generate_form(schema)


async def _confirm_and_execute(
    extension_name: str,
    get_values: Callable[[], dict[str, Any]],
    on_execute: Callable[[str, dict[str, Any]], None] | None
) -> None:
    """Show confirmation dialog and execute action."""
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Execute {extension_name}?').classes('text-h6')
        ui.label('This action cannot be undone.').classes('text-grey mb-4')
        
        with ui.row():
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button(
                'Execute',
                color='primary',
                on_click=lambda: [
                    on_execute(extension_name, get_values()) if on_execute else None,
                    dialog.close()
                ]
            )
    
    dialog.open()
