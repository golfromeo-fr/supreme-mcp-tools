"""
Mutators Box Component.

Renders editable mutators panel with dynamic forms.
"""

from nicegui import ui
from typing import List, Dict, Any, Callable, Optional

from ..models import Extension


def MutatorsBox(
    extensions: List[Extension],
    on_submit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    loading: bool = False
) -> None:
    """
    Render editable mutators box with dynamic forms.
    
    Args:
        extensions: List of Extension objects (mutators).
        on_submit: Callback when submitting a mutator.
        loading: Whether to show loading state.
    """
    with ui.card().classes('w-full'):
        ui.label('Configuration').classes('text-h6 mb-2')
        
        if not extensions:
            ui.label('No configuration options available').classes('text-grey')
            return
        
        for ext in extensions:
            with ui.expansion(ext.name, icon='settings').classes('w-full'):
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
