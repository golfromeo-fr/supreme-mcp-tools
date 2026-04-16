"""
Environment Variable Editor Component.

Renders a collapsible panel for viewing and editing environment variables
associated with a tool. Secret values are masked; inline editing with save/cancel.
"""

from collections.abc import Callable

from nicegui import ui

from ..models import EnvVariable
from ..logging_config import get_logger

logger = get_logger(__name__)


def EnvVarEditor(
    tool_name: str,
    variables: list[EnvVariable],
    on_update: Callable[[str, str, str], None] | None = None,
    on_delete: Callable[[str, str], None] | None = None,
) -> None:
    """
    Render environment variable editor for a tool.

    Shows all declared env vars with masked values and inline edit capability.

    Args:
        tool_name: Name of the tool these vars belong to
        variables: List of EnvVariable objects
        on_update: Callback(tool_name, var_name, new_value) when saving a value
        on_delete: Callback(tool_name, var_name) when deleting a variable
    """
    if not variables:
        return

    with ui.expansion("Environment Variables", icon="key").classes("w-full"):
        ui.label("Configure environment variables for this tool. Changes take effect immediately.").classes(
            "text-caption text-grey mb-3"
        )

        with ui.column().classes("w-full gap-3"):
            for var in sorted(variables, key=lambda v: (not v.required, v.name)):
                _render_var_row(tool_name, var, on_update, on_delete)


def _render_var_row(
    tool_name: str,
    var: EnvVariable,
    on_update: Callable | None,
    on_delete: Callable | None,
) -> None:
    """Render a single environment variable row with inline editing."""
    with ui.card().classes("w-full") as row_card:
        with ui.row().classes("w-full items-center gap-3"):
            # Variable name
            ui.label(var.name).classes("font-mono text-sm font-bold")

            # Required/optional badge
            if var.required:
                ui.badge("REQUIRED", color="red").classes("text-xs")
            else:
                ui.badge("optional", color="grey").classes("text-xs")

            # Secret badge
            if var.secret:
                ui.icon("lock", size="xs").classes("text-orange-500")
            else:
                ui.icon("lock_open", size="xs").classes("text-grey")

            ui.space()

            # Current value (masked for secrets, plain for non-secrets)
            value_display = ui.label(
                var.value_masked if var.value_masked else ("(not set)" if not var.is_set else "")
            ).classes("text-sm font-mono")

            if not var.is_set:
                value_display.classes("text-grey italic")
            elif var.secret:
                value_display.classes("text-orange-600")

            # Edit button
            edit_btn = ui.button(icon="edit", on_click=lambda: _toggle_edit(
                row_card, tool_name, var, on_update, on_delete
            )).props("flat round size=sm")
            edit_btn.props('color="blue"')


def _toggle_edit(
    card: ui.card,
    tool_name: str,
    var: EnvVariable,
    on_update: Callable | None,
    on_delete: Callable | None,
) -> None:
    """Replace the card content with an edit form."""
    card.clear()

    with card:
        with ui.column().classes("w-full gap-2"):
            # Header
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(var.name).classes("font-mono text-sm font-bold")
                if var.required:
                    ui.badge("REQUIRED", color="red").classes("text-xs")

            # Description
            if var.description:
                ui.label(var.description).classes("text-caption text-grey")

            # Type-appropriate input field
            var_type = getattr(var, 'type', 'string')
            # Determine initial value: use current raw value if set, otherwise default
            current_val = getattr(var, 'value_raw', '') or var.default or ''
            if var.options:
                input_field = ui.select(
                    options=var.options,
                    value=None,
                    with_input=True,
                    label=f"Select or enter value for {var.name}",
                ).classes("w-full")
            elif var_type == "boolean":
                # Parse boolean from string "true"/"false" or bool
                bool_val = False
                if var.is_set and current_val:
                    bool_val = current_val.lower() == "true"
                elif var.default:
                    bool_val = var.default.lower() == "true"
                input_field = ui.switch(
                    f"Enable {var.name}",
                    value=bool_val,
                ).classes("w-full")
            elif var_type == "integer":
                int_val = None
                if var.is_set and current_val:
                    try:
                        int_val = int(current_val)
                    except ValueError:
                        int_val = int(var.default) if var.default else None
                elif var.default:
                    try:
                        int_val = int(var.default)
                    except ValueError:
                        int_val = None
                input_field = ui.number(
                    f"Value for {var.name}",
                    value=int_val,
                    min=var.minimum,
                    max=var.maximum,
                    format="%d",
                ).classes("w-full")
            elif var_type == "number":
                float_val = None
                if var.is_set and current_val:
                    try:
                        float_val = float(current_val)
                    except ValueError:
                        float_val = float(var.default) if var.default else None
                elif var.default:
                    try:
                        float_val = float(var.default)
                    except ValueError:
                        float_val = None
                input_field = ui.number(
                    f"Value for {var.name}",
                    value=float_val,
                    min=var.minimum,
                    max=var.maximum,
                    format="%.2f",
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

            # Default hint
            if var.default:
                ui.label(f"Default: {var.default}").classes("text-caption text-grey")

            # Action buttons
            with ui.row().classes("gap-2"):
                def handle_save() -> None:
                    var_type = getattr(var, 'type', 'string')
                    if var_type == "boolean":
                        new_value = "true" if input_field.value else "false"
                    elif var_type == "integer":
                        new_value = str(int(input_field.value))
                    elif var_type == "number":
                        new_value = str(float(input_field.value))
                    else:
                        new_value = str(input_field.value) if input_field.value is not None else ""
                    if new_value and on_update:
                        # Call on_update WITHOUT awaiting - it updates state and schedules content_refresh
                        # which must happen in the UI task, not a background task
                        on_update(tool_name, var.name, new_value)
                    # Show saving indicator - parent refresh will replace with fresh data
                    card.clear()
                    with card:
                        ui.label(f"Saving {var.name}...").classes("text-grey italic text-sm")

                def handle_cancel() -> None:
                    card.clear()
                    with card:
                        _render_var_row(tool_name, var, on_update, on_delete)

                ui.button("Save", icon="save", on_click=handle_save).props("color=positive")
                ui.button("Cancel", on_click=handle_cancel).props("flat")

                if on_delete and not var.required:
                    def handle_delete() -> None:
                        logger.info(f"env_delete: tool={tool_name} var={var.name}")
                        on_delete(tool_name, var.name)
                        card.clear()
                        with card:
                            ui.label(f"Deleting {var.name}...").classes("text-grey italic text-sm")

                    ui.button("Delete", icon="delete", on_click=handle_delete).props(
                        "color=negative flat"
                    )


def parse_env_vars_from_api(data: dict) -> list[EnvVariable]:
    """
    Parse environment variables from the management API response.

    Args:
        data: Response dict from GET /api/tools/{name}/env

    Returns:
        List of EnvVariable objects
    """
    variables = data.get("variables", {})
    result = []

    for var_name, var_data in variables.items():
        result.append(EnvVariable(
            name=var_name,
            description=var_data.get("description", ""),
            required=var_data.get("required", False),
            secret=var_data.get("secret", True),
            value_masked=var_data.get("value_masked", ""),
            is_set=var_data.get("is_set", False),
            default=var_data.get("default", ""),
            options=var_data.get("options", []),
            type=var_data.get("type", "string"),
            minimum=var_data.get("minimum"),
            maximum=var_data.get("maximum"),
        ))

    return result
