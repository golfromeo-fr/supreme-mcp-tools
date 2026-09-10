"""M4/H2 — state_docs (shared cluster-state documents) unit tests.

Fake-connection based (CI-safe); the live Turso/PG path was proven for the
identical machinery in users_store's db backend.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.shared import state_docs  # noqa: E402


class _FakeCursor:
    def __init__(self, state, params):
        self._state, self._params = state, params

    def fetchone(self):
        sql = self._state["last_sql"]
        if sql.startswith("SELECT data"):
            name = self._params[0]
            return (self._state["rows"][name],) if name in self._state["rows"] else None
        return None


class _FakeConn:
    def __init__(self):
        self.state = {"rows": {}, "last_sql": "", "statements": []}

    def execute(self, sql, params=()):
        self.state["last_sql"] = sql
        self.state["statements"].append((sql, params))
        if sql.startswith("INSERT INTO mcp_state_docs"):
            (name, data, _ts) = params
            self.state["rows"][name] = data
        return _FakeCursor(self.state, params)


class _FakeSqlStore:
    is_available = True

    def __init__(self):
        self._conn = _FakeConn()


@pytest.fixture()
def db_backend(monkeypatch):
    fake = _FakeSqlStore()
    monkeypatch.setenv("MCP_STATE_BACKEND", "db")
    monkeypatch.setattr("tools.shared.sql_store.get_sql_store", lambda: fake)
    monkeypatch.setattr(state_docs, "_conn_singleton", None)
    monkeypatch.setattr(state_docs, "_init_done", False)
    return fake


def test_round_trip(db_backend):
    assert state_docs.save_doc("tools_config", {"disabled_tools": {"simplemcp": ["get_secret"]}})
    loaded = state_docs.load_doc("tools_config")
    assert loaded == {"disabled_tools": {"simplemcp": ["get_secret"]}}
    # second document is independent
    state_docs.save_doc("other", {"a": 1})
    assert state_docs.load_doc("tools_config") != state_docs.load_doc("other")


def test_missing_row_returns_none(db_backend):
    assert state_docs.load_doc("never_saved") is None


def test_backend_unavailable_returns_none_false(monkeypatch, tmp_path):
    """db selected but no SQL backend → load None / save False (the caller's
    file-fallback contract); nothing touches a local file."""
    import tools.shared.sql_store as sql_store_mod
    from tools.shared.sql_store import NullSqlStore

    monkeypatch.setenv("MCP_STATE_BACKEND", "db")
    monkeypatch.setattr(sql_store_mod, "get_sql_store", lambda: NullSqlStore())
    monkeypatch.setattr(state_docs, "_conn_singleton", None)
    monkeypatch.setattr(state_docs, "_init_done", False)

    assert state_docs.load_doc("tools_config") is None
    assert state_docs.save_doc("tools_config", {"x": 1}) is False


def test_json_default_ignores_backend_env(monkeypatch):
    """Default (json) mode never initializes the db, even with a factory."""
    called = {"n": 0}
    import tools.shared.sql_store as sql_store_mod

    def _boom():
        called["n"] += 1
        raise AssertionError("must not resolve a backend in json mode")

    monkeypatch.setattr(sql_store_mod, "get_sql_store", _boom)
    assert state_docs.load_doc("tools_config") is None
    assert state_docs.save_doc("tools_config", {"x": 1}) is False
    assert called["n"] == 0


def test_tools_config_routes_through_db(db_backend, tmp_path, monkeypatch):
    """db mode: launcher tools_config reads/writes the shared doc; the local
    file is untouched. Explicit config_path still forces the file."""
    from launcher.tools_config import (
        load_tools_config, save_tools_config, _DEFAULT_CONFIG_FILE)

    monkeypatch.setattr("launcher.tools_config._DEFAULT_CONFIG_FILE",
                        tmp_path / "tools_config.json")
    cfg = {"disabled_tools": {"simplemcp": ["get_secret"]},
           "tools": {"simplemcp": ["double", "square", "greet", "get_secret"]},
           "version": 1}
    save_tools_config(cfg)
    # central doc received it, local file NOT written
    assert state_docs.load_doc("tools_config")["version"] == 1
    assert not (tmp_path / "tools_config.json").exists()
    assert load_tools_config() == cfg

    # explicit path → file mode, bypasses the backend
    file_cfg = {"disabled_tools": {}, "tools": {}, "version": 1}
    explicit = tmp_path / "explicit.json"
    save_tools_config(file_cfg, config_path=explicit)
    assert json.loads(explicit.read_text()) == file_cfg
    assert load_tools_config(config_path=explicit) == file_cfg
    _ = _DEFAULT_CONFIG_FILE  # import sanity


def test_function_masks_reads_db_doc(db_backend, monkeypatch):
    """masked_tools in db mode reads the shared doc (cluster-wide masks)."""
    import tools.shared.function_masks as fm

    state_docs.save_doc("tools_config", {
        "disabled_tools": {"simplemcp": ["get_secret"]}})
    assert fm.masked_tools("simplemcp") == ["get_secret"]
    assert fm.masked_tools("memorymcp") == []
    # explicit config_path still forces the file
    f = tmp_path_local = Path(PROJECT_ROOT) / "nonexistent_masks.json"
    assert fm.masked_tools("simplemcp", config_path=f) == []
    _ = tmp_path_local
