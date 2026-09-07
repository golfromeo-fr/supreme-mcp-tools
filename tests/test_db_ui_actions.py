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
