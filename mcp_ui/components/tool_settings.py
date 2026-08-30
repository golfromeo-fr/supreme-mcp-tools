"""
Tool Settings Component.

UI for managing tool settings including disabled tools.
"""

import fcntl
import json
import os
from pathlib import Path
from nicegui import ui
from collections.abc import Callable

from ..logging_config import get_logger

logger = get_logger(__name__)

# Config file location
TOOLS_CONFIG_FILE = Path.home() / ".config" / "supreme-mcp-tools" / "tools_config.json"

REPO_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"


def get_function_descriptions(server_name: str) -> dict[str, str]:
    """Read function descriptions from the tool's config.json (name -> description)."""
    path = REPO_TOOLS_DIR / server_name / "config.json"
    try:
        cfg = json.loads(path.read_text())
        return {
            t["name"]: t.get("description", "")
            for t in cfg.get("tools", [])
            if isinstance(t, dict) and t.get("name")
        }
    except Exception:
        return {}


async def apply_function_mask(
    server_name: str,
    tool_name: str,
    is_enabled: bool,
    on_done: Callable | None = None,
) -> None:
    """Persist a function mask through the management API and notify.

    Shared by the Function Masks dialog and the per-tool Functions tab.
    ``on_done`` (optional) re-renders the caller's rows after the save.
    """
    from ..management_ui import get_api_client

    client = get_api_client()
    if is_enabled:
        response = await client.enable_tool(server_name, tool_name)
    else:
        response = await client.disable_tool(server_name, tool_name)
    if response.success:
        ui.notify("Mask saved. Changes take effect immediately.", type="info", duration=3)
    else:
        ui.notify(f"Failed to save: {response.error}", type="negative", duration=5)
    if on_done:
        on_done()


def _load_tools_config() -> dict:
    """Load tools configuration from file."""
    if not TOOLS_CONFIG_FILE.exists():
        return {"disabled_tools": {}, "tools": {}, "version": 1}

    try:
        with Path(TOOLS_CONFIG_FILE).open() as f:
            return json.load(f)
    except Exception:
        return {"disabled_tools": {}, "tools": {}, "version": 1}


def _save_tools_config(config: dict) -> None:
    """Save tools configuration atomically under an exclusive lock.

    The launcher and browser sessions can write this file concurrently; a bare
    open('w') truncated the JSON on crash and lost concurrent updates (the same
    pattern fixed in launcher/env_manager.py). Readers of the os.replace()d
    file always see a complete document.
    """
    TOOLS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TOOLS_CONFIG_FILE.open("a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = TOOLS_CONFIG_FILE.with_suffix(".json.tmp")
        try:
            with tmp.open("w") as f:
                json.dump(config, f, indent=2)
            os.replace(tmp, TOOLS_CONFIG_FILE)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def get_server_tools(server_name: str) -> list[str]:
    """Get known tools for a server from config."""
    config = _load_tools_config()
    return config.get("tools", {}).get(server_name, [])


def get_disabled_tools(server_name: str) -> list[str]:
    """Get disabled tools for a server from config."""
    config = _load_tools_config()
    return config.get("disabled_tools", {}).get(server_name, [])


def set_disabled_tools(server_name: str, disabled_list: list[str]) -> None:
    """Set disabled tools for a server in config."""
    config = _load_tools_config()
    if "disabled_tools" not in config:
        config["disabled_tools"] = {}
    config["disabled_tools"][server_name] = disabled_list
    _save_tools_config(config)


def get_all_servers() -> list[str]:
    """Get list of all servers from config."""
    config = _load_tools_config()
    tools = config.get("tools", {})
    return list(tools.keys())


class ToolSettingsDialog:
    """Dialog for managing tool settings including disabled tools."""

    def __init__(
        self,
        server_name: str,
        available_tools: list[str],
        on_save: Callable | None = None,
    ):
        """
        Initialize tool settings dialog.

        Args:
            server_name: Name of the server (e.g., 'webmcp')
            available_tools: List of available tool names
            on_save: Optional callback when settings are saved
        """
        self.server_name = server_name
        self.available_tools = available_tools
        self.on_save = on_save
        self.disabled_tools: list[str] = []
        self._dialog = None
        self._checkboxes: dict[str, ui.checkbox] = {}

    def load_disabled_tools(self) -> list[str]:
        """Load disabled tools from config file."""
        return get_disabled_tools(self.server_name)

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        disabled = self.disabled_tools

        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            self._dialog = dialog

            with ui.row().classes("w-full justify-between items-center mb-4"):
                ui.label(f"Function Masks — {self.server_name}").classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props("flat round")

            if not self.available_tools:
                ui.label("No tools available for this server.").classes("text-grey mb-4")
                ui.label("(Server may be offline or not responding)").classes("text-grey text-caption")
            else:
                ui.label("Enable or disable tools:").classes("text-body2 text-grey mb-4")

                with ui.column().classes("w-full gap-2"):
                    for tool_name in sorted(self.available_tools):
                        is_disabled = tool_name in disabled
                        with ui.row().classes("w-full justify-between items-center"):
                            ui.label(tool_name).classes("font-mono")
                            switch = ui.switch(
                                "Enabled",
                                value=not is_disabled,
                                on_change=lambda e, t=tool_name: self._on_toggle(t, e.value),
                            )
                            self._checkboxes[tool_name] = switch

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Save",
                    on_click=self._on_save,
                    color="primary",
                )

    def _on_toggle(self, tool_name: str, is_enabled: bool) -> None:
        """Handle toggle change."""
        if is_enabled:
            if tool_name in self.disabled_tools:
                self.disabled_tools.remove(tool_name)
        else:
            if tool_name not in self.disabled_tools:
                self.disabled_tools.append(tool_name)

    def _on_save(self) -> None:
        """Handle save button click."""
        set_disabled_tools(self.server_name, self.disabled_tools)
        ui.notify("Settings saved. Changes take effect immediately.", type="positive", duration=3)
        if self.on_save:
            self.on_save()
        self._dialog.close()

    def open(self) -> None:
        """Open the settings dialog."""
        self.disabled_tools = self.load_disabled_tools()
        self._checkboxes = {}
        self._build_ui()
        self._dialog.open()


class GlobalToolSettingsDialog:
    """Global dialog for managing tool settings across all servers."""

    def __init__(self, servers: list[str], on_save: Callable | None = None):
        """
        Initialize global tool settings dialog.

        Args:
            servers: List of server names
            on_save: Optional callback when settings are saved
        """
        self.servers = servers
        self.on_save = on_save
        self.selected_server: str | None = None
        self._dialog = None
        self._filter_text = ""
        self._rows_container = None
        self._count_label = None
        self._all_tools: list[str] = []
        self._current_server: str | None = None

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            self._dialog = dialog

            with ui.row().classes("w-full justify-between items-center mb-4"):
                ui.label("Function Masks — enable or disable tools per server").classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props("flat round")

            with ui.row().classes("w-full gap-4 mb-4"):
                ui.label("Server:").classes("text-body1")
                server_select = ui.select(
                    options=self.servers,
                    value=self.servers[0] if self.servers else None,
                    on_change=lambda e: self._on_server_change(e.value),
                ).classes("w-48")

            self._settings_container = ui.column().classes("w-full")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Close", on_click=dialog.close).props("flat")

        # Load initial server
        if self.servers:
            self.selected_server = self.servers[0]
            self._load_server_settings(self.servers[0])

    def _on_server_change(self, server_name: str) -> None:
        """Handle server selection change."""
        self.selected_server = server_name
        self._load_server_settings(server_name)

    def _load_server_settings(self, server_name: str) -> None:
        """Load settings for a specific server."""
        # Clear container
        self._settings_container.clear()
        self._filter_text = ""

        # Get tools from config (only enabled ones are discovered via tools/list)
        tools = get_server_tools(server_name)

        # Get disabled tools from config
        disabled = get_disabled_tools(server_name)

        # Merge both lists to show all known tools (including disabled ones)
        self._all_tools = sorted(set(tools) | set(disabled))
        self._current_server = server_name

        with self._settings_container:
            if not self._all_tools:
                ui.label("No tool information available.").classes("text-grey mb-4")
                ui.label("(Add tools to config file or server may be offline)").classes("text-grey text-caption")
                return

            self._count_label = ui.label().classes("text-body2 text-grey mb-2")
            filter_input = ui.input(
                "Filter functions...", placeholder="type to filter",
            ).props("dense clearable outlined").classes("w-full mb-2")
            filter_input.on_value_change(lambda e: self._set_filter(e.value or ""))
            self._rows_container = ui.column().classes("w-full gap-1")

        self._refresh_rows_and_count()

    def _set_filter(self, text: str) -> None:
        """Handle filter-input changes."""
        self._filter_text = text
        self._render_rows()

    def _refresh_rows_and_count(self) -> None:
        """Re-read mask state and refresh the count + rows (post-toggle)."""
        if not hasattr(self, "_rows_container") or self._rows_container is None:
            return
        disabled = get_disabled_tools(self._current_server)
        masked = len([t for t in self._all_tools if t in disabled])
        total = len(self._all_tools)
        self._count_label.text = (
            f"{masked} of {total} functions masked" if masked
            else f"All {total} functions enabled"
        )
        self._render_rows()

    def _render_rows(self) -> None:
        """Render the per-function rows, honoring the current filter."""
        self._rows_container.clear()
        disabled = get_disabled_tools(self._current_server)
        filter_text = (self._filter_text or "").lower()

        with self._rows_container:
            shown = 0
            for tool_name in self._all_tools:
                if filter_text and filter_text not in tool_name.lower():
                    continue
                shown += 1
                is_disabled = tool_name in disabled
                row_classes = "w-full justify-between items-center"
                if is_disabled:
                    row_classes += " opacity-60"
                with ui.row().classes(row_classes):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(tool_name).classes("font-mono text-sm")
                        if is_disabled:
                            ui.badge("MASKED", color="orange").props("outline").classes("text-xs")
                    ui.switch(
                        "Enabled",
                        value=not is_disabled,
                        on_change=lambda e, t=tool_name, s=self._current_server: self._on_toggle(s, t, e.value),
                    )
            if shown == 0:
                ui.label("No functions match the filter.").classes("text-grey text-caption")

    def _on_toggle(self, server_name: str, tool_name: str, is_enabled: bool) -> None:
        """Handle toggle change - save via the management API immediately."""
        async def _apply() -> None:
            await apply_function_mask(server_name, tool_name, is_enabled, on_done=self._refresh_rows_and_count)
            # The refresh runs from a background task outside the dialog's slot
            # context — force propagation of the rebuilt subtree to the client.
            self._settings_container.update()

        # Run the API call on the event loop without freezing the dialog.
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(_apply())
        else:
            asyncio.run(_apply())

    def open(self) -> None:
        """Open the settings dialog."""
        self._build_ui()
        self._dialog.open()


async def show_tool_settings(server_name: str, available_tools: list[str]) -> None:
    """
    Show tool settings dialog for a server.

    Args:
        server_name: Name of the server
        available_tools: List of available tool names
    """
    dialog = ToolSettingsDialog(server_name, available_tools)
    dialog.open()


async def show_global_tool_settings(servers: list[str]) -> None:
    """
    Show global tool settings dialog.

    Args:
        servers: List of server names
    """
    dialog = GlobalToolSettingsDialog(servers)
    dialog.open()
