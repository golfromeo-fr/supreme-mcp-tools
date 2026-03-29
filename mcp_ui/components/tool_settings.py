"""
Tool Settings Component.

UI for managing tool settings including disabled tools.
"""

import json
from pathlib import Path
from nicegui import ui
from typing import List, Dict, Callable, Optional

from ..logging_config import get_logger
from ..api_client import get_client

logger = get_logger(__name__)

# Config file location
TOOLS_CONFIG_FILE = Path.home() / ".config" / "supreme-mcp-tools" / "tools_config.json"


def _load_tools_config() -> dict:
    """Load tools configuration from file."""
    if not TOOLS_CONFIG_FILE.exists():
        return {"disabled_tools": {}, "tools": {}, "version": 1}

    try:
        with open(TOOLS_CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {"disabled_tools": {}, "tools": {}, "version": 1}


def _save_tools_config(config: dict) -> None:
    """Save tools configuration to file."""
    TOOLS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOOLS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_server_tools(server_name: str) -> List[str]:
    """Get known tools for a server from config."""
    config = _load_tools_config()
    return config.get("tools", {}).get(server_name, [])


def get_disabled_tools(server_name: str) -> List[str]:
    """Get disabled tools for a server from config."""
    config = _load_tools_config()
    return config.get("disabled_tools", {}).get(server_name, [])


def set_disabled_tools(server_name: str, disabled_list: List[str]) -> None:
    """Set disabled tools for a server in config."""
    config = _load_tools_config()
    if "disabled_tools" not in config:
        config["disabled_tools"] = {}
    config["disabled_tools"][server_name] = disabled_list
    _save_tools_config(config)


def get_all_servers() -> List[str]:
    """Get list of all servers from config."""
    config = _load_tools_config()
    tools = config.get("tools", {})
    return list(tools.keys())


class ToolSettingsDialog:
    """Dialog for managing tool settings including disabled tools."""

    def __init__(
        self,
        server_name: str,
        available_tools: List[str],
        on_save: Optional[Callable] = None,
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
        self.disabled_tools: List[str] = []
        self._dialog = None
        self._checkboxes: Dict[str, ui.checkbox] = {}

    def load_disabled_tools(self) -> List[str]:
        """Load disabled tools from config file."""
        return get_disabled_tools(self.server_name)

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        disabled = self.disabled_tools

        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            self._dialog = dialog

            with ui.row().classes("w-full justify-between items-center mb-4"):
                ui.label(f"Tool Settings - {self.server_name}").classes("text-h6")
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

    def __init__(self, servers: List[str], on_save: Optional[Callable] = None):
        """
        Initialize global tool settings dialog.

        Args:
            servers: List of server names
            on_save: Optional callback when settings are saved
        """
        self.servers = servers
        self.on_save = on_save
        self.selected_server: Optional[str] = None
        self._dialog = None

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            self._dialog = dialog

            with ui.row().classes("w-full justify-between items-center mb-4"):
                ui.label("Tool Settings").classes("text-h6")
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

        # Get tools from config
        tools = get_server_tools(server_name)

        # Get disabled tools from config
        disabled = get_disabled_tools(server_name)

        with self._settings_container:
            if not tools:
                ui.label("No tool information available.").classes("text-grey mb-4")
                ui.label("(Add tools to config file or server may be offline)").classes("text-grey text-caption")
            else:
                ui.label("Enable or disable tools:").classes("text-body2 text-grey mb-2")
                for tool_name in sorted(tools):
                    is_disabled = tool_name in disabled
                    with ui.row().classes("w-full justify-between items-center"):
                        ui.label(tool_name).classes("font-mono text-sm")
                        switch = ui.switch(
                            "Enabled",
                            value=not is_disabled,
                            on_change=lambda e, t=tool_name, s=server_name: self._on_toggle(s, t, e.value),
                        )

    def _on_toggle(self, server_name: str, tool_name: str, is_enabled: bool) -> None:
        """Handle toggle change - save immediately."""
        disabled = get_disabled_tools(server_name)
        if is_enabled:
            if tool_name in disabled:
                disabled.remove(tool_name)
        else:
            if tool_name not in disabled:
                disabled.append(tool_name)

        set_disabled_tools(server_name, disabled)
        ui.notify(f"Saved. Changes take effect immediately.", type="info", duration=3)

    def open(self) -> None:
        """Open the settings dialog."""
        self._build_ui()
        self._dialog.open()


async def show_tool_settings(server_name: str, available_tools: List[str]) -> None:
    """
    Show tool settings dialog for a server.

    Args:
        server_name: Name of the server
        available_tools: List of available tool names
    """
    dialog = ToolSettingsDialog(server_name, available_tools)
    dialog.open()


async def show_global_tool_settings(servers: List[str]) -> None:
    """
    Show global tool settings dialog.

    Args:
        servers: List of server names
    """
    dialog = GlobalToolSettingsDialog(servers)
    dialog.open()
