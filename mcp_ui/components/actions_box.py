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

    Returns (status_label, status_note) tuple. status_label is plain text
    (emoji-free; badges handle color in the UI).
    """
    input_props = ext.json_schema.get('input', {}).get('properties', {})

    # If no input properties, action may not need parameters
    if not input_props:
        return "Ready", "No parameters required"

    # Check if it has description - lack of description might mean not fully implemented
    if not ext.description:
        return "Unknown", "No description - may not be implemented"

    return "Ready", ""


def ActionsBox(
    extensions: list[Extension],
    on_execute: Callable[[str, dict[str, Any]], None] | None = None,
    loading: bool = False
) -> None:
    """
    Render actions with execute buttons.

    Renders directly into the enclosing container (the Extensions tab) —
    no outer card or section header. One expansion per action.

    Args:
        extensions: List of Extension objects (actions).
        on_execute: Callback when executing an action.
        loading: Whether to disable execute buttons.
    """
    if not extensions:
        ui.label('No actions available').classes('text-grey')
        return

    for ext in extensions:
        status_label, status_note = _get_action_status(ext)

        with ui.expansion(f"{ext.name} — {status_label}", icon='play_arrow').classes('w-full'):
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

            async def _do_execute() -> None:
                # Close first so the UI unblocks, then await the parent's async
                # handler — as a discarded coroutine this call never ran, so
                # actions appeared to succeed while doing nothing.
                dialog.close()
                if on_execute:
                    await on_execute(extension_name, get_values())

            ui.button(
                'Execute',
                color='primary',
                on_click=_do_execute,
            )
    
    dialog.open()
