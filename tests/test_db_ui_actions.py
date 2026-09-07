"""mgmt action extensions for the UI presets panel (connect_preset /
disconnect_connection) — offline, env-driven throwaway preset."""

import os
import sys
import uuid

import pytest

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "tools", "databasemcp")))


@pytest.fixture()
def ui_preset(tmp_path, monkeypatch):
    """A throwaway DB_PRESET pointing at a temp libsql file."""
    db_file = tmp_path / f"ui_action_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("DB_PRESET_97", f"file:{db_file}")
    monkeypatch.setenv("DB_PRESET_97_DESC", "UI action test")
    import db_tools  # noqa: E402  (imports after env is set)

    return "97"


def test_connect_preset_action_connects_and_is_idempotent(ui_preset):
    import db_tools

    first = db_tools.connect_preset_action({"preset": ui_preset})
    assert first["success"] is True
    assert "Connected preset" in first["message"]

    second = db_tools.connect_preset_action({"preset": ui_preset})
    assert second["success"] is True
    assert "already connected" in second["message"]


def test_connect_preset_action_data_source_reflects_state(ui_preset):
    import db_tools

    db_tools.connect_preset_action({"preset": ui_preset})
    presets = db_tools.get_connection_presets({})["presets"]
    row = next(p for p in presets if p["number"] == ui_preset)
    assert row["connected"] is True
    assert row["desc"] == "UI action test"

    db_tools.disconnect_connection_action({"name": ui_preset})
    presets = db_tools.get_connection_presets({})["presets"]
    row = next(p for p in presets if p["number"] == ui_preset)
    assert row["connected"] is False


def test_disconnect_connection_action(ui_preset):
    import db_tools

    db_tools.connect_preset_action({"preset": ui_preset})
    result = db_tools.disconnect_connection_action({"name": ui_preset})
    assert result["success"] is True
    assert "Disconnected" in result["message"]


def test_action_param_guards():
    import db_tools

    missing = db_tools.connect_preset_action({})
    assert missing["success"] is False

    unknown = db_tools.connect_preset_action({"preset": "no-such-preset"})
    assert unknown["success"] is False
    assert "Unknown preset" in unknown["message"]

    unknown_name = db_tools.disconnect_connection_action({"name": "ghost"})
    assert unknown_name["success"] is False
    assert "Unknown connection" in unknown_name["message"]


def test_setup_extensions_registers_all_customs_despite_duplicates():
    """Regression 2026-09-07: the custom clear_cache collided with the common
    set and the register loop ABORTED — every custom extension after the
    colliding name (connect_preset, disconnect_connection) silently went
    missing from the launcher's registry. One duplicate must not kill the
    rest, and the real custom handlers must be registered first."""
    class FlakyRegistry:
        def __init__(self):
            self.names = set()

        def register(self, tool_name, ext):
            if ext.name in self.names:
                raise ValueError(
                    f"Extension '{ext.name}' already registered for tool '{tool_name}'"
                )
            self.names.add(ext.name)

    import db_tools

    registry = FlakyRegistry()
    db_tools.setup_extensions(registry=registry)

    assert "connect_preset" in registry.names
    assert "disconnect_connection" in registry.names
    assert "clear_cache" in registry.names  # the custom (registry-backed) one
    assert "connection_presets" in registry.names  # common set still lands
