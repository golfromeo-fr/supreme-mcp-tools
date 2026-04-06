"""
Form generation utilities.

Provides schema-to-form generation for extension inputs.
Used by ActionsBox for action parameter forms.
"""

from typing import Any, Dict, Callable
from nicegui import ui


def _generate_form(schema: Dict[str, Any]) -> Callable[[], Dict[str, Any]]:
    """
    Generate a form from a JSON schema and return a getter function.

    Args:
        schema: JSON schema with 'input' properties

    Returns:
        A callable that returns current form values as a dict
    """
    input_props = schema.get('input', {}).get('properties', {})

    # Track form field references
    field_refs: Dict[str, Any] = {}

    def _create_fields():
        for prop_name, prop_def in input_props.items():
            prop_type = prop_def.get('type', 'string')
            description = prop_def.get('description', '')
            default = prop_def.get('default')

            if description:
                ui.label(description).classes('text-caption text-grey mb-1')

            if prop_type == 'boolean':
                field_refs[prop_name] = ui.switch(
                    f"Enable {prop_name}",
                    value=default if default is not None else False,
                )
            elif prop_type == 'integer' or prop_type == 'number':
                min_val = prop_def.get('minimum')
                max_val = prop_def.get('maximum')
                step = 1 if prop_type == 'integer' else 0.01
                field_refs[prop_name] = ui.number(
                    f"Value for {prop_name}",
                    value=default,
                    min=min_val,
                    max=max_val,
                    step=step,
                )
            elif prop_type == 'string':
                enum_vals = prop_def.get('enum')
                if enum_vals:
                    field_refs[prop_name] = ui.select(
                        options=enum_vals,
                        value=default,
                        label=prop_name,
                    )
                else:
                    field_refs[prop_name] = ui.input(
                        label=prop_name,
                        value=default or '',
                    )
            elif prop_type == 'array':
                field_refs[prop_name] = ui.input(
                    label=prop_name,
                    value=default or '',
                    placeholder='Comma-separated values',
                )
            else:
                field_refs[prop_name] = ui.input(
                    label=prop_name,
                    value=default or '',
                )

    # Build form in a column
    with ui.column().classes('w-full gap-2'):
        _create_fields()

    # Return getter function
    def get_values() -> Dict[str, Any]:
        values = {}
        for prop_name, field in field_refs.items():
            val = field.value
            # Handle select which might have dict value
            if hasattr(field, 'value') and isinstance(val, dict):
                val = val.get('value', '')
            values[prop_name] = val
        return values

    return get_values
