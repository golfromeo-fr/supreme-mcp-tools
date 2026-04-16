"""
Auth Box Component.

Renders per-tool authorization key management in the web UI.
"""

from collections.abc import Callable

from nicegui import ui

from ..logging_config import get_logger

logger = get_logger(__name__)


def AuthBox(
    tool_name: str,
    is_set: bool,
    value_masked: str | None,
    on_update: Callable[[str, str], None] | None = None,
) -> None:
    """
    Render auth configuration box for a tool.

    Args:
        tool_name: Name of the tool.
        is_set: Whether an auth key is configured.
        value_masked: Masked value of the key (e.g., "****abcd").
        on_update: Callback(tool_name, new_api_key) when updating.
    """
    with ui.card().classes("w-full"):
        ui.label("Authorization").classes("text-h6 mb-2")

        with ui.row().classes("items-center gap-2"):
            if is_set:
                ui.icon("lock", size="sm").classes("text-green-500")
                ui.label("Authorization key is set").classes("text-sm")
                if value_masked:
                    ui.label(f"({value_masked})").classes("text-sm text-grey")
            else:
                ui.icon("lock_open", size="sm").classes("text-grey")
                ui.label("No authorization key - tool is open").classes("text-sm text-grey")

        with ui.row().classes("gap-2 mt-2"):
            ui.button(
                "Update Key",
                icon="key",
                on_click=lambda: _show_update_dialog(tool_name, on_update),
            ).props("flat")

        ui.label(
            "Authorization key protects the tool's extension API (port 81xx). "
            "Changes require tool restart to take effect."
        ).classes("text-caption text-grey mt-2")


def _show_update_dialog(
    tool_name: str,
    on_update: Callable[[str, str], None] | None,
) -> None:
    """Show dialog to update the auth key."""
    new_key = None

    with ui.dialog() as dialog, ui.card():
        ui.label(f"Update Authorization Key for {tool_name}").classes("text-h6 mb-4")
        ui.label("Enter the new authorization key (leave empty to remove):").classes("text-grey mb-2")

        input_field = ui.input(
            "Authorization Key",
            password=True,
            password_toggle_button=True,
        ).classes("w-full")

        def handle_save() -> None:
            new_key = input_field.value
            if on_update:
                on_update(tool_name, new_key or "")
            dialog.close()

        with ui.row().classes("gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", icon="save", on_click=handle_save).props("color=primary")

    dialog.open()
