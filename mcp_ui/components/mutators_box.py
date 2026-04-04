"""
Mutators Box Component.

Renders editable mutators panel with dynamic forms, plus environment variables
all in one unified Configuration card.
"""

import asyncio
from nicegui import ui
from typing import List, Dict, Any, Callable, Optional

from ..models import Extension, EnvVariable


def _get_mutator_status(ext: Extension) -> tuple[str, str]:
    """
    Get status for a mutator extension.

    Returns (status_label, status_note) tuple.
    """
    input_props = ext.json_schema.get('input', {}).get('properties', {})

    # If no input properties, it's likely a read-only/default mutator
    if not input_props:
        return "ℹ️ Default only", "No configuration parameters"

    # Check if it has description - lack of description might mean not fully implemented
    if not ext.description:
        return "⚠️ Unknown", "No description - may not be implemented"

    return "✅ Configurable", ""


def MutatorsBox(
    extensions: List[Extension],
    on_submit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    loading: bool = False,
    current_values: Optional[Dict[str, Dict[str, Any]]] = None,
    env_variables: Optional[List[EnvVariable]] = None,
    tool_name: Optional[str] = None,
    on_env_update: Optional[Callable[[str, str, str], None]] = None,
    on_env_delete: Optional[Callable[[str, str], None]] = None,
) -> None:
    """
    Render editable mutators box with dynamic forms, plus env vars.

    Args:
        extensions: List of Extension objects (mutators).
        on_submit: Callback when submitting a mutator.
        loading: Whether to show loading state.
        current_values: Current values for mutators (e.g. api_key_info).
        env_variables: List of EnvVariable objects for this tool.
        tool_name: Name of the tool (needed for env var callbacks).
        on_env_update: Callback(tool_name, var_name, new_value) for env var updates.
        on_env_delete: Callback(tool_name, var_name) for env var deletions.
    """
    current_values = current_values or {}
    env_variables = env_variables or []

    with ui.card().classes('w-full'):
        ui.label('Configuration').classes('text-h6 mb-2')

        if not extensions and not env_variables:
            ui.label('No configuration options available').classes('text-grey')
            return

        # --- FEF Mutators ---
        for ext in extensions:
            status_label, status_note = _get_mutator_status(ext)
            current = current_values.get(ext.name, {})

            with ui.expansion(f"{ext.name} [{status_label}]", icon='settings').classes('w-full'):
                # Show current value for api_key
                if ext.name == "api_key" and current:
                    is_set = current.get("is_set", False)
                    value_masked = current.get("value_masked", "(not set)")
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.icon("key", size="sm").classes("text-grey")
                        if is_set:
                            ui.label(f"Current: {value_masked}").classes("text-sm font-mono text-orange-600")
                        else:
                            ui.label("Current: (not set)").classes("text-sm text-grey italic")
                    ui.separator().classes("mb-2")

                # Show status note
                if status_note:
                    ui.label(status_note).classes("text-caption mb-2")

                if ext.description:
                    ui.label(ext.description).classes('text-grey mb-2')

                # Generate form from schema
                form_values = _generate_form(ext.json_schema)

                # Submit button
                submit_btn = ui.button(
                    'Apply Changes',
                    icon='save',
                    on_click=lambda e=ext.name, v=form_values: (
                        on_submit(e, v()) if on_submit else None
                    )
                )
                if loading:
                    submit_btn.disable()

        # --- Environment Variables ---
        if env_variables:
            with ui.expansion(f"Environment Variables [{len(env_variables)}]", icon='key', value=True).classes('w-full'):
                ui.label("Runtime environment variables — update takes effect immediately").classes(
                    "text-caption text-grey mb-3"
                )
                for var in sorted(env_variables, key=lambda v: v.name):
                    _render_env_var_row(tool_name, var, on_env_update, on_env_delete)


def _render_env_var_row(
    tool_name: str,
    var: EnvVariable,
    on_update: Optional[Callable],
    on_delete: Optional[Callable],
) -> None:
    """Render a single env var row with inline editing."""
    with ui.card().classes("w-full") as row_card:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(var.name).classes("font-mono text-sm font-bold")
            if var.required:
                ui.badge("REQUIRED", color="red").classes("text-xs")
            if var.secret:
                ui.icon("lock", size="xs").classes("text-orange-500")
            else:
                ui.icon("lock_open", size="xs").classes("text-grey")
            ui.space()
            value_display = ui.label(
                var.value_masked if var.value_masked else ("(not set)" if not var.is_set else "")
            ).classes("text-sm font-mono")
            if not var.is_set:
                value_display.classes("text-grey italic")
            elif var.secret:
                value_display.classes("text-orange-600")

            edit_btn = ui.button(
                icon="edit",
                on_click=lambda: _toggle_env_edit(
                    row_card, tool_name, var, on_update, on_delete
                )
            ).props("flat round size=sm")
            edit_btn.props('color="blue"')


def _toggle_env_edit(
    card: ui.card,
    tool_name: str,
    var: EnvVariable,
    on_update: Optional[Callable],
    on_delete: Optional[Callable],
) -> None:
    """Replace the card content with an edit form for env var."""
    card.clear()

    with card:
        with ui.column().classes("w-full gap-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(var.name).classes("font-mono text-sm font-bold")
                if var.required:
                    ui.badge("REQUIRED", color="red").classes("text-xs")

            if var.description:
                ui.label(var.description).classes("text-caption text-grey")

            if var.options:
                input_field = ui.select(
                    options=var.options,
                    value=None,
                    with_input=True,
                    label=f"Select or enter value for {var.name}",
                ).classes("w-full")
            elif var.secret:
                input_field = ui.input(
                    f"New value for {var.name}",
                    password=True,
                    password_toggle_button=True,
                ).classes("w-full")
            else:
                input_field = ui.input(
                    f"New value for {var.name}",
                ).classes("w-full")

            if var.default:
                ui.label(f"Default: {var.default}").classes("text-caption text-grey")

            with ui.row().classes("gap-2"):
                def handle_save() -> None:
                    new_value = input_field.value
                    if new_value is not None and on_update:
                        asyncio.create_task(on_update(tool_name, var.name, str(new_value)))
                    card.clear()
                    with card:
                        ui.label(f"Saving {var.name}...").classes("text-grey italic text-sm")

                def handle_cancel() -> None:
                    card.clear()
                    with card:
                        _render_env_var_row(tool_name, var, on_update, on_delete)

                ui.button("Save", icon="save", on_click=handle_save).props("color=positive")
                ui.button("Cancel", on_click=handle_cancel).props("flat")

                if on_delete and not var.required:
                    def handle_delete() -> None:
                        asyncio.create_task(on_delete(tool_name, var.name))
                        card.clear()
                        with card:
                            ui.label(f"Deleting {var.name}...").classes("text-grey italic text-sm")

                    ui.button("Delete", icon="delete", on_click=handle_delete).props(
                        "color=negative flat"
                    )


def _generate_form(schema: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    """
    Generate form fields from JSON schema.
    
    Args:
        schema: JSON schema definition.
    
    Returns:
        A callable that returns current form values.
    """
    inputs: Dict[str, Any] = {}
    properties = schema.get('input', {}).get('properties', {})
    
    if not properties:
        ui.label('No configuration required - using defaults').classes('text-grey text-sm italic mb-2')
    else:
        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get('type', 'string')
            label = prop_def.get('description', prop_name)
            default = prop_def.get('default')
            
            if prop_type == 'integer':
                inputs[prop_name] = ui.number(
                    label,
                    value=default if default is not None else 0,
                    min=prop_def.get('minimum'),
                    max=prop_def.get('maximum')
                ).classes('w-full mb-2')
            
            elif prop_type == 'number':
                inputs[prop_name] = ui.number(
                    label,
                    value=default if default is not None else 0.0,
                    min=prop_def.get('minimum'),
                    max=prop_def.get('maximum'),
                    format='%.2f'
                ).classes('w-full mb-2')
            
            elif prop_type == 'boolean':
                inputs[prop_name] = ui.switch(
                    label,
                    value=default if default is not None else False
                ).classes('w-full mb-2')
            
            elif prop_type == 'array':
                default_value = ','.join(default) if default else ''
                inputs[prop_name] = ui.textarea(
                    label,
                    value=default_value
                ).classes('w-full mb-2')
                ui.label('Comma-separated values').classes('text-grey text-xs mb-2')
            
            else:  # string and others
                inputs[prop_name] = ui.input(
                    label,
                    value=default if default is not None else ''
                ).classes('w-full mb-2')
    
    def get_values() -> Dict[str, Any]:
        return {name: input_widget.value for name, input_widget in inputs.items()}
    
    return get_values
