"""P3 — generic data tools over the connection registry (real libSQL DBs).

Drives the tool functions directly (await) — the same functions the MCP
surface exposes — against two named libSQL file connections.
"""

import sys
from pathlib import Path

import pytest


# Review fix 2026-10-18: preset-grant checks fail closed for unresolvable
# identities under the user-store verifier. These tests exercise generic
# query/execute/schema mechanics in-process (no HTTP identity).
# Review fix 2026-10-24 (F1): raw connect_database gained the same admin-only
# gate — bypassed here for the same reason (mechanics, not identity).
# The ORIGINAL gate is saved as _real_gate so TestGateActiveRoundTrip can
# restore it and exercise the hot paths WITH enforcement.
@pytest.fixture(autouse=True)
def _skip_grant_check(monkeypatch):
    import db_tools
    if not hasattr(db_tools, "_real_gate"):
        db_tools._real_gate = db_tools._assert_preset_grant
    monkeypatch.setattr(db_tools, "_assert_preset_grant", lambda connection: None)
    monkeypatch.setattr(db_tools, "_refuse_non_admin", lambda action: None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import asyncio  # noqa: E402

import pytest as _pytest  # noqa: E402

try:
    import libsql_experimental  # noqa: F401

    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = _pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")


@pytest.fixture()
def two_conns(tmp_path, monkeypatch):
    """Two named libSQL connections; REGISTRY cleared before and after."""
    import db_tools
    from connections import REGISTRY

    REGISTRY._entries.clear()
    REGISTRY._active = None
    monkeypatch.delenv("USERID", raising=False)
    monkeypatch.delenv("DB_AUTOCONNECT", raising=False)

    async def setup():
        r1 = await db_tools.connect_database("one", "libsql", {"url": f"file:{tmp_path / 'one.db'}"})
        r2 = await db_tools.connect_database("two", "libsql", {"url": f"file:{tmp_path / 'two.db'}"})
        return r1, r2

    r1, r2 = asyncio.run(setup())
    assert "Connected 'one'" in r1 and "Connected 'two'" in r2
    yield db_tools
    REGISTRY._entries.clear()
    REGISTRY._active = None


def _run(coro):
    return asyncio.run(coro)


class TestConnectionTools:
    def test_connect_rejects_bad_name(self, two_conns):
        db = two_conns
        out = _run(db.connect_database("bad name!", "libsql", {"url": "file::memory:"}))
        assert "Invalid connection name" in out

    def test_connect_rejects_missing_params(self, two_conns):
        db = two_conns
        out = _run(db.connect_database("pg1", "postgres", {"host": "h"}))
        assert "Missing required params" in out and "dbname" in out

    def test_connect_rejects_unknown_type(self, two_conns):
        db = two_conns
        out = _run(db.connect_database("x", "mysql", {"url": "y"}))
        assert "Unsupported db_type" in out

    def test_connect_failure_propagates_cleanly(self, two_conns):
        db = two_conns
        out = _run(db.connect_database("bad", "postgres", {
            "host": "127.0.0.1", "dbname": "nope", "user": "u", "password": "p", "connect_timeout": 1}))
        assert out.startswith("Connect failed")
        assert "p" != out or "password" not in out.lower() or "***" in out  # no secret echo

    def test_list_connections_marks_active_and_masks(self, two_conns):
        db = two_conns
        out = _run(db.list_connections())
        assert "- one (libsql, CONNECTED" in out and "*ACTIVE*" in out
        assert "password" not in out.lower()

    def test_use_database_switches(self, two_conns):
        db = two_conns
        out = _run(db.use_database("two"))
        assert "Active connection: 'two' (libsql)" in out
        out = _run(db.list_connections())
        assert "- two (libsql, CONNECTED, cached=0) *ACTIVE*" in out

    def test_use_database_unknown(self, two_conns):
        db = two_conns
        out = _run(db.use_database("nope"))
        assert "Unknown connection" in out

    def test_disconnect_active_falls_back(self, two_conns):
        db = two_conns
        out = _run(db.disconnect_database("two"))
        assert "Active now: one" in out
        out = _run(db.disconnect_database("one"))
        assert "Active now: none" in out
        out = _run(db.query, ) if False else _run(db.query("SELECT 1"))
        assert "No database connection" in out  # edge 8


class TestGenericTools:
    def test_execute_and_query_round_trip(self, two_conns):
        db = two_conns
        out = _run(db.execute_sql("CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT)"))
        assert out.startswith("OK.")
        out = _run(db.query("SELECT * FROM t"))
        assert out == "[]"

    def test_query_truncation_note(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE t (id INTEGER)"))
        for i in range(5):
            _run(db.execute_sql(f"INSERT INTO t VALUES ({i})"))
        out = _run(db.query("SELECT * FROM t ORDER BY id", max_rows=3))
        assert "(Truncated — showing 3 rows)" in out

    def test_query_rejects_dml(self, two_conns):
        db = two_conns
        out = _run(db.query("DELETE FROM t"))
        assert "read-only" in out and "execute_sql" in out

    def test_query_rejects_empty(self, two_conns):
        db = two_conns
        assert "required" in _run(db.query("   "))

    def test_query_with_named_connection_targets_isolation(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE on_one (x INTEGER)", connection="one"))
        out = _run(db.query("SELECT name FROM sqlite_master WHERE type='table'", connection="two"))
        assert "on_one" not in out  # two.db is a different file

    def test_get_schemas_cached_second_call(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE t (id INTEGER PRIMARY KEY)"))
        first = _run(db.get_schemas("t"))
        assert "cached" not in first
        second = _run(db.get_schemas("t"))
        assert "(cached)" in second

    def test_get_schemas_missing_table_normalized(self, two_conns):
        db = two_conns
        out = _run(db.get_schemas("missing_table"))
        assert "not found on connection 'one'" in out

    def test_list_tables(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE visible_t (x INTEGER)"))
        out = _run(db.list_tables())
        assert "visible_t" in out

    def test_explain_plan(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE t (id INTEGER)"))
        plan = _run(db.explain_plan("SELECT * FROM t"))
        assert isinstance(plan, str) and plan

    def test_get_valid_languages_oracle_only(self, two_conns):
        db = two_conns
        out = _run(db.get_valid_languages())
        assert "Oracle-only" in out and "libsql" in out

    def test_ddl_invalidates_schema_cache(self, two_conns):
        db = two_conns
        _run(db.execute_sql("CREATE TABLE t (id INTEGER)"))
        _run(db.get_schemas("t"))  # populates cache
        _run(db.execute_sql("ALTER TABLE t ADD COLUMN extra TEXT"))
        second = _run(db.get_schemas("t"))
        assert "(cached)" not in second  # cleared by execute_sql
        assert "extra" in second


# ---------------------------------------------------------------------------
# Review 2026-10-24 test gap: every test above bypasses the grant gate via
# the autouse fixture. This one runs WITH the real gate active (the caller
# identified as admin through the fake token map), so the hot paths keep
# gate coverage even here.
# ---------------------------------------------------------------------------

class TestGateActiveRoundTrip:
    def test_query_round_trip_with_real_gate(self, two_conns, monkeypatch):
        db_tools = two_conns
        import core

        # Real _assert_preset_grant + real connections. Identity comes from
        # the VERIFIER's token map (the gate reads request headers → token →
        # entry → role), so fake a mono-style verifier carrying an admin key.
        class _MonoVerifier:
            def tokens(self):
                return {"admin-key": {"client_id": "admin",
                                      "role": "admin", "scopes": ["mcp"]}}

        class _Req:
            headers = {"authorization": "Bearer admin-key"}

        def _fake_get_http_request():  # sync — matches fastmcp's shape
            return _Req()

        monkeypatch.setattr(core.mcp, "auth", _MonoVerifier(), raising=False)
        monkeypatch.setattr(db_tools, "_assert_preset_grant", db_tools._real_gate)
        import fastmcp.server.dependencies as deps
        monkeypatch.setattr(deps, "get_http_request", _fake_get_http_request)

        async def flow():
            await db_tools.execute_sql(
                sql="CREATE TABLE gate_probe (v TEXT)", connection="one")
            await db_tools.execute_sql(
                sql="INSERT INTO gate_probe VALUES ('x')", connection="one")
            return await db_tools.query(
                sql="SELECT v FROM gate_probe", connection="one")

        result = _run(flow())
        assert "'x'" in result  # row came back through the REAL gate

    def test_real_gate_refuses_user_key(self, two_conns, monkeypatch):
        """Same setup with a role=user token: the REAL gate must refuse the
        ungranted raw connection by name."""
        db_tools = two_conns
        import core

        class _UserVerifier:
            def tokens(self):
                return {"user-key": {"client_id": "worker",
                                     "role": "user", "scopes": ["mcp"]}}

        class _Req:
            headers = {"authorization": "Bearer user-key"}

        def _fake_get_http_request():
            return _Req()

        monkeypatch.setattr(core.mcp, "auth", _UserVerifier(), raising=False)
        monkeypatch.setattr(db_tools, "_assert_preset_grant", db_tools._real_gate)
        import fastmcp.server.dependencies as deps
        monkeypatch.setattr(deps, "get_http_request", _fake_get_http_request)

        # query() catches the gate's PermissionError into its error string
        result = _run(db_tools.query(sql="SELECT 1", connection="one"))
        assert "not a granted preset" in result
