"""Review fixes 2026-10-18 (the three security-relevant findings):

1. databasemcp: connection=None rides the registry's ACTIVE entry (or
   lazily creates the legacy env-default Oracle) — under the user-store
   verifier a non-admin caller must have that target grant-backed.
2. memorymcp: _memory_caller fails CLOSED on anomaly paths when the
   user-store verifier is installed (distinct "unknown" owner namespace,
   not the legacy/unscoped one).
3. mcp_ui mono: the literal default admin/admin login is refused on a
   non-loopback bind (mono twin of the P1-1 seeding guard).
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "tools" / "databasemcp",
          PROJECT_ROOT / "tools" / "memorymcp", PROJECT_ROOT / "mcp_ui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class _UserStoreVerifier:
    is_user_store = True
    def __init__(self, entries):
        self._entries = entries
    def tokens(self):
        return self._entries


class _MonoVerifier:
    def __init__(self, entries):
        self._entries = entries
    def tokens(self):
        return self._entries


def _break_http(monkeypatch):
    import fastmcp.server.dependencies as deps
    def _raise():
        raise RuntimeError("no http context")
    monkeypatch.setattr(deps, "get_http_request", _raise)


# ---------------------------------------------------------------------------
# 1. databasemcp — the connection=None path
# ---------------------------------------------------------------------------

@pytest.fixture()
def grant_env(monkeypatch):
    """User-store verifier with a non-admin caller granted preset '02'; env
    presets 01/02 present; a fresh registry."""
    import connections
    import core
    import presets as presets_mod
    monkeypatch.setattr(core.mcp, "auth", _UserStoreVerifier({
        "user-key": {"client_id": "worker", "role": "user",
                     "db_presets": ["02"]},
    }), raising=False)
    monkeypatch.setenv("DB_AUTOCONNECT", "0")  # keep the legacy default off
    def fake_presets():
        return [
            presets_mod.Preset(number="01", dialect="oracle", params={},
                               url_masked="oracle-legacy", name="oracle-legacy"),
            presets_mod.Preset(number="02", dialect="postgres", params={},
                               url_masked="pglocal", name="pglocal"),
        ]
    monkeypatch.setattr(presets_mod, "load_presets", fake_presets)
    fresh = connections.ConnectionRegistry()
    monkeypatch.setattr(connections, "REGISTRY", fresh, raising=False)
    import db_tools
    monkeypatch.setattr(db_tools, "REGISTRY", fresh, raising=False)
    return fresh


def _caller(monkeypatch, token="user-key"):
    import fastmcp.server.dependencies as deps
    req = type("R", (), {"headers": {"authorization": f"Bearer {token}"}})()
    monkeypatch.setattr(deps, "get_http_request", lambda: req, raising=False)


def test_none_connection_without_active_refused(grant_env, monkeypatch):
    import db_tools
    _caller(monkeypatch)
    with pytest.raises(PermissionError, match="no active database connection"):
        db_tools._assert_preset_grant(None)


def test_none_connection_riding_ungranted_active_refused(grant_env, monkeypatch):
    import db_tools
    from connections import ConnectionEntry
    grant_env._entries["default"] = ConnectionEntry(
        name="default", dialect="oracle", params={}, handle=object())
    grant_env._active = "default"
    _caller(monkeypatch)
    with pytest.raises(PermissionError, match="not a granted preset"):
        db_tools._assert_preset_grant(None)


def test_none_connection_riding_granted_active_passes(grant_env, monkeypatch):
    import db_tools
    from connections import ConnectionEntry
    grant_env._entries["pglocal"] = ConnectionEntry(
        name="pglocal", dialect="postgres", params={}, handle=object())
    grant_env._active = "pglocal"
    _caller(monkeypatch)
    db_tools._assert_preset_grant(None)  # granted preset '02' == pglocal


def test_anomaly_fails_closed_under_user_store(grant_env, monkeypatch):
    import db_tools
    _break_http(monkeypatch)
    with pytest.raises(PermissionError, match="identity could not be resolved"):
        db_tools._assert_preset_grant(None)


def test_mono_anomaly_stays_fail_open(monkeypatch):
    import core
    import db_tools
    monkeypatch.setattr(core.mcp, "auth", _MonoVerifier({
        "sys": {"client_id": "databasemcp", "scopes": ["mcp"]}}), raising=False)
    _break_http(monkeypatch)
    db_tools._assert_preset_grant(None)  # mono: historical fail-open, no raise


# ---------------------------------------------------------------------------
# 2. memorymcp — _memory_caller fail-closed
# ---------------------------------------------------------------------------

def test_memory_caller_user_store_anomaly_fails_closed(monkeypatch):
    import memory_tools
    monkeypatch.setattr(memory_tools.mcp, "auth",
                        _UserStoreVerifier({}), raising=False)
    _break_http(monkeypatch)
    assert memory_tools._memory_caller() == ("unknown", False, True)


def test_memory_caller_mono_anomaly_unchanged(monkeypatch):
    import memory_tools
    monkeypatch.setattr(memory_tools.mcp, "auth", None, raising=False)
    assert memory_tools._memory_caller() == (None, False, False)


# ---------------------------------------------------------------------------
# 3. mcp_ui — mono default-password refusal on non-loopback
# ---------------------------------------------------------------------------

@pytest.fixture()
def ui_env(monkeypatch):
    import mcp_ui.management_ui as management_ui
    monkeypatch.setattr(management_ui, "_multi_user_mode", lambda: False)
    monkeypatch.delenv("MCP_UI_USERNAME", raising=False)
    monkeypatch.delenv("MCP_UI_PASSWORD", raising=False)
    return management_ui


def test_mono_default_login_refused_on_open_bind(ui_env, monkeypatch):
    monkeypatch.setenv("MCP_UI_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="default admin/admin login refused"):
        ui_env.try_login_user("admin", "admin")


def test_mono_set_password_login_ok_on_open_bind(ui_env, monkeypatch):
    monkeypatch.setenv("MCP_UI_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_UI_PASSWORD", "operator-pw-1")
    session = ui_env.try_login_user("admin", "operator-pw-1")
    assert session["authenticated"] is True


def test_mono_default_login_ok_on_loopback(ui_env, monkeypatch):
    monkeypatch.setenv("MCP_UI_HOST", "127.0.0.1")
    session = ui_env.try_login_user("admin", "admin")
    assert session["authenticated"] is True


# ---------------------------------------------------------------------------
# 4. F1 (review 2026-10-24): the grant gate covers EVERY entry point
# ---------------------------------------------------------------------------

import asyncio


def _ungranted_active(grant_env):
    """Active entry 'default' — an existing registry entry that is NOT a
    granted preset (the worker is granted only preset '02')."""
    from connections import ConnectionEntry
    grant_env._entries["default"] = ConnectionEntry(
        name="default", dialect="oracle", params={}, handle=object())
    grant_env._active = "default"
    return grant_env


def test_explicit_existing_entry_refused(grant_env, monkeypatch):
    """An explicit name that is a LIVE registry entry but not a granted
    preset must be refused — the old code skipped the check entirely."""
    import db_tools
    _ungranted_active(grant_env)
    _caller(monkeypatch)
    with pytest.raises(PermissionError, match="not a granted preset"):
        db_tools._assert_preset_grant("default")


def test_batch_statements_ungranted_refused(grant_env, monkeypatch):
    import db_tools
    _ungranted_active(grant_env)
    _caller(monkeypatch)
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.execute_sql(statements=["SELECT 1 FROM DUAL"]))


def test_begin_transaction_ungranted_refused(grant_env, monkeypatch):
    import db_tools
    _ungranted_active(grant_env)
    _caller(monkeypatch)
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.begin_transaction())


def test_metadata_tools_ungranted_refused(grant_env, monkeypatch):
    import db_tools
    _ungranted_active(grant_env)
    _caller(monkeypatch)
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.get_schemas("SOMETABLE", connection="default"))
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.list_tables(connection="default"))
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.explain_plan("SELECT 1", connection="default"))


def test_use_database_ungranted_refused(grant_env, monkeypatch):
    import db_tools
    _ungranted_active(grant_env)
    _caller(monkeypatch)
    with pytest.raises(PermissionError):
        asyncio.run(db_tools.use_database("default"))


def test_connect_database_refused_for_non_admin(grant_env, monkeypatch):
    import db_tools
    _caller(monkeypatch)
    with pytest.raises(PermissionError, match="admin-only"):
        asyncio.run(db_tools.connect_database(
            "zz-raw", "postgres",
            {"host": "h", "dbname": "d", "user": "u", "password": "p"}))
    # refused BEFORE any connect attempt — nothing registered
    assert "zz-raw" not in grant_env._entries


def test_connect_database_ok_for_mono(grant_env, monkeypatch):
    """Mono verifier: raw connects keep working (fail-open by design)."""
    import core
    import db_tools
    monkeypatch.setattr(core.mcp, "auth", _MonoVerifier(
        {"sys": {"client_id": "databasemcp"}}), raising=False)
    _break_http(monkeypatch)
    # A libsql file::memory: connect is cheap and side-effect free
    result = asyncio.run(db_tools.connect_database(
        "zz-mono", "libsql", {"url": "file::memory:"}))
    assert "Connected 'zz-mono'" in result
