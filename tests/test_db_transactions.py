"""E4 transaction management — registry lifecycle + dialect tx methods.

Two layers:
- ScriptedFakeDialect: records calls, verifies the REGISTRY contract
  (one tx per entry, ownership, disconnect refusal, close_all abort).
- Real libsql (tmp file) and real Postgres (dbmcp_tx_test_* tables):
  verify the DIALECT tx methods end-to-end (isolation, commit, rollback).
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "tools", "databasemcp")))

from dialects._base import DbDialect  # noqa: E402


# ============================================================================
# Registry lifecycle (scripted fake)
# ============================================================================

class ScriptedFakeDialect(DbDialect):
    """Records tx call order; open_tx returns a fresh sentinel object."""

    name = "scripted"
    REQUIRED_PARAMS = ("x",)

    def __init__(self):
        self.calls: list[str] = []
        self._n = 0

    def connect(self, params):
        self._n += 1
        return f"handle-{self._n}"

    def close(self, handle):
        self.calls.append(f"close:{handle}")

    def ping(self, handle):
        pass

    def run_select(self, handle, sql, max_rows):
        return [], False

    def execute(self, handle, sql):
        return 0

    def list_tables(self, handle):
        return []

    def describe_table(self, handle, table):
        return {"columns": [], "constraints": [], "foreign_keys": []}

    def explain(self, handle, sql):
        return ""

    def format_error(self, e):
        return {"error": "DB_ERROR", "code": None, "message": str(e), "offset": None}

    # tx methods (E4)
    def open_tx(self, handle, params):
        self.calls.append(f"open_tx:{handle}")
        return f"tx-{handle}"

    def select_tx(self, tx_handle, sql, max_rows):
        self.calls.append(f"select_tx:{tx_handle}")
        return [{"x": 1}], False

    def execute_tx(self, tx_handle, sql):
        self.calls.append(f"execute_tx:{tx_handle}:{sql[:20]}")
        return 1

    def commit_tx(self, tx_handle):
        self.calls.append(f"commit_tx:{tx_handle}")

    def rollback_tx(self, tx_handle):
        self.calls.append(f"rollback_tx:{tx_handle}")

    def close_tx(self, handle, tx_handle):
        self.calls.append(f"close_tx:{handle}:{tx_handle}")


@pytest.fixture()
def tx_registry(monkeypatch):
    import dialects
    from connections import ConnectionRegistry

    fake = ScriptedFakeDialect()
    monkeypatch.setitem(dialects.DIALECTS, "scripted", fake)
    reg = ConnectionRegistry()
    reg.connect("c1", "scripted", {"x": 1})
    return reg, fake


def test_begin_tx_sets_state_and_calls_open_tx(tx_registry):
    reg, fake = tx_registry
    entry, tx_id = reg.begin_tx("c1")
    assert tx_id == entry.tx_id and len(tx_id) == 32
    assert entry.tx_handle == "tx-handle-1"
    assert fake.calls == ["open_tx:handle-1"]


def test_second_begin_on_same_entry_rejected(tx_registry):
    reg, _ = tx_registry
    reg.begin_tx("c1")
    with pytest.raises(RuntimeError, match="already has an active transaction"):
        reg.begin_tx("c1")


def test_begin_tx_preset_bypass(tx_registry, monkeypatch):
    reg, _ = tx_registry
    monkeypatch.setenv("DB_PRESET_96", "file:/tmp/never_opened.db")
    entry, tx_id = reg.begin_tx("96")
    assert entry.name == "96" and entry.tx_id == tx_id


def test_finish_tx_commit_calls_commit_then_close_and_clears_cache(tx_registry):
    reg, fake = tx_registry
    entry, tx_id = reg.begin_tx("c1")
    entry.schema_cache["t"] = {"stale": True}
    returned = reg.finish_tx(tx_id, commit=True)
    assert returned.name == "c1"
    assert fake.calls[-2:] == [f"commit_tx:tx-handle-1", f"close_tx:handle-1:tx-handle-1"]
    assert entry.tx_id is None and entry.tx_handle is None
    assert entry.schema_cache == {}  # DDL staleness guard on commit


def test_finish_tx_rollback_keeps_cache(tx_registry):
    reg, fake = tx_registry
    entry, tx_id = reg.begin_tx("c1")
    entry.schema_cache["t"] = {"keep": True}
    reg.finish_tx(tx_id, commit=False)
    assert "rollback_tx:tx-handle-1" in fake.calls
    assert entry.schema_cache == {"t": {"keep": True}}


def test_finish_tx_unknown_or_replayed_id(tx_registry):
    reg, _ = tx_registry
    with pytest.raises(LookupError):
        reg.finish_tx("no-such-tx", commit=True)
    _entry, tx_id = reg.begin_tx("c1")
    reg.finish_tx(tx_id, commit=True)
    with pytest.raises(LookupError):  # replayed id
        reg.finish_tx(tx_id, commit=True)


def test_disconnect_refused_with_active_tx(tx_registry):
    reg, _ = tx_registry
    _entry, tx_id = reg.begin_tx("c1")
    message = reg.disconnect("c1")
    assert "active transaction" in message and tx_id[:8] in message
    assert "c1" in reg._entries  # still there


def test_disconnect_works_after_finish(tx_registry):
    reg, _ = tx_registry
    _entry, tx_id = reg.begin_tx("c1")
    reg.finish_tx(tx_id, commit=False)
    assert reg.disconnect("c1") in ("none",)


def test_close_all_force_aborts_active_tx(tx_registry):
    reg, fake = tx_registry
    entry, _tx_id = reg.begin_tx("c1")
    closed, skipped = reg.close_all()
    assert (closed, skipped) == (1, 0)
    assert "rollback_tx:tx-handle-1" in fake.calls  # force rollback happened
    assert entry.tx_id is None


# ============================================================================
# Dialect tx methods — real libsql (tmp file)
# ============================================================================

@pytest.fixture()
def libsql_pair(tmp_path):
    """(entry_handle_registry_conn, dialect) on a throwaway file DB."""
    db_file = tmp_path / f"tx_{uuid.uuid4().hex[:8]}.db"
    from dialects import get_dialect

    dialect = get_dialect("libsql")
    params = {"url": f"file:{db_file}"}
    handle = dialect.connect(params)
    dialect.execute(handle, "CREATE TABLE txp (id INTEGER PRIMARY KEY, v TEXT)")
    dialect.execute(handle, "INSERT INTO txp VALUES (1, 'committed-baseline')")
    yield dialect, handle, params
    try:
        dialect.close(handle)
    except Exception:
        pass


def test_libsql_tx_round_trip_commit(libsql_pair):
    dialect, handle, params = libsql_pair
    tx = dialect.open_tx(handle, params)
    assert dialect.execute_tx(tx, "INSERT INTO txp VALUES (2, 'in-tx')") == 1
    # uncommitted write invisible to the registry connection
    rows, _ = dialect.run_select(handle, "SELECT count(*) AS n FROM txp", 10)
    assert rows[0]["n"] == 1
    # visible through the tx itself
    rows, _ = dialect.select_tx(tx, "SELECT count(*) AS n FROM txp", 10)
    assert rows[0]["n"] == 2
    dialect.commit_tx(tx)
    rows, _ = dialect.run_select(handle, "SELECT count(*) AS n FROM txp", 10)
    assert rows[0]["n"] == 2
    dialect.close_tx(handle, tx)


def test_libsql_tx_round_trip_rollback(libsql_pair):
    dialect, handle, params = libsql_pair
    tx = dialect.open_tx(handle, params)
    dialect.execute_tx(tx, "INSERT INTO txp VALUES (2, 'doomed')")
    dialect.rollback_tx(tx)
    rows, _ = dialect.run_select(handle, "SELECT count(*) AS n FROM txp", 10)
    assert rows[0]["n"] == 1  # rollback discarded the insert
    dialect.close_tx(handle, tx)


def test_libsql_execute_tx_rejects_multi_statement(libsql_pair):
    dialect, handle, params = libsql_pair
    tx = dialect.open_tx(handle, params)
    with pytest.raises(ValueError, match="Multiple statements"):
        dialect.execute_tx(tx, "INSERT INTO txp VALUES (3, 'x'); INSERT INTO txp VALUES (4, 'y')")
    dialect.close_tx(handle, tx)


# ============================================================================
# Dialect tx methods — real Postgres (dbmcp_tx_test_* tables, DROP CASCADE)
# ============================================================================

def _pg_dsn() -> str | None:
    dsn = os.environ.get("POSTGRES_TEST_DSN")
    return dsn or None


@pytest.fixture()
def pg_dialect():
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("POSTGRES_TEST_DSN not set — live PG tx tests skipped")
    from dialects import get_dialect

    dialect = get_dialect("postgres")
    params = {
        "host": os.environ.get("PGTESTHOST", "192.168.0.1"),
        "port": "5432",
        "dbname": "memorymcp",
        "user": "gr",
        "password": os.environ.get("PGTESTPASSWORD", ""),
    }
    # derive real params from the DSN instead of guessing pieces
    import psycopg
    parsed = psycopg.conninfo.conninfo_to_dict(dsn)
    params = {k: str(v) for k, v in parsed.items() if k in ("host", "port", "dbname", "user", "password")}
    return dialect, params


def test_postgres_tx_round_trip_commit(pg_dialect):
    dialect, params = pg_dialect
    table = f"dbmcp_tx_test_{uuid.uuid4().hex[:8]}"
    handle = dialect.connect(params)
    try:
        dialect.execute(handle, f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, v TEXT)")
        tx = dialect.open_tx(handle, params)
        assert dialect.execute_tx(tx, f"INSERT INTO {table} VALUES (1, 'in-tx')") == 1
        # uncommitted invisible to a second connection
        with handle.connection() as other:
            with other.cursor() as cur:
                cur.execute(f"SELECT count(*) AS n FROM {table}")
                assert cur.fetchone()["n"] == 0
        dialect.commit_tx(tx)
        rows, _ = dialect.run_select(handle, f"SELECT count(*) AS n FROM {table}", 10)
        assert rows[0]["n"] == 1
        dialect.close_tx(handle, tx)
    finally:
        try:
            dialect.execute(handle, f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass
        dialect.close(handle)


def test_postgres_tx_round_trip_rollback(pg_dialect):
    dialect, params = pg_dialect
    table = f"dbmcp_tx_test_{uuid.uuid4().hex[:8]}"
    handle = dialect.connect(params)
    try:
        dialect.execute(handle, f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, v TEXT)")
        tx = dialect.open_tx(handle, params)
        dialect.execute_tx(tx, f"INSERT INTO {table} VALUES (1, 'doomed')")
        dialect.rollback_tx(tx)
        rows, _ = dialect.run_select(handle, f"SELECT count(*) AS n FROM {table}", 10)
        assert rows[0]["n"] == 0
        dialect.close_tx(handle, tx)
    finally:
        try:
            dialect.execute(handle, f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass
        dialect.close(handle)


def test_postgres_close_tx_is_idempotent_after_commit(pg_dialect):
    dialect, params = pg_dialect
    handle = dialect.connect(params)
    try:
        tx = dialect.open_tx(handle, params)
        dialect.commit_tx(tx)
        dialect.close_tx(handle, tx)  # rollback-after-commit must not raise
    finally:
        dialect.close(handle)


# ============================================================================
# Tool surface (T2/T3) — in-memory MCP client over the shared mcp instance
# ============================================================================

@pytest.fixture()
def clean_registry():
    """Isolate the module-global REGISTRY per test (unique names + teardown)."""
    from connections import REGISTRY

    yield REGISTRY
    REGISTRY.close_all()


@pytest.fixture()
def sweep_db(clean_registry):
    """A uniquely-named libsql connection to a throwaway file DB."""
    from connections import REGISTRY

    name = f"txsweep{uuid.uuid4().hex[:6]}"
    db_file = Path("/tmp") / f"dbmcp_tx_tool_{uuid.uuid4().hex[:8]}.db"
    REGISTRY.connect(name, "libsql", {"url": f"file:{db_file}"})
    REGISTRY.get(name).params  # params stored
    yield name, f"file:{db_file}"
    for suf in ("", "-wal", "-shm", "-journal"):
        try:
            os.unlink(str(db_file) + suf)
        except FileNotFoundError:
            pass


def _call_tool(tool: str, arguments: dict) -> str:
    """Call a tool on the shared in-process MCP instance."""
    import asyncio

    from fastmcp import Client

    import db_tools

    async def _run():
        async with Client(db_tools.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if getattr(result, "content", None):
                return result.content[0].text
            return ""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class TestAtomicBatch:
    def test_batch_commit_via_registry(self, sweep_db):
        from db_tools import _run_batch

        name, _url = sweep_db
        note = _run_batch(name, [
            "CREATE TABLE bt (id INTEGER PRIMARY KEY, v TEXT)",
            "INSERT INTO bt VALUES (1, 'a')",
            "INSERT INTO bt VALUES (2, 'b')",
        ])
        assert note.startswith("OK. Batch committed: 3 statement(s)"), note
        assert "rowcounts" in note

    def test_batch_commit_visible_afterwards(self, sweep_db):
        from connections import REGISTRY

        name, _url = sweep_db
        _call_tool("execute_sql", {"statements": [
            f"CREATE TABLE bt_{name} (id INTEGER PRIMARY KEY)",
            f"INSERT INTO bt_{name} VALUES (1)",
        ]})
        out = _call_tool("query", {"sql": f"SELECT count(*) AS n FROM bt_{name}", "connection": name})
        assert "'n': 1" in out, out

    def test_batch_rollback_on_failure(self, sweep_db):
        name, _url = sweep_db
        _call_tool("execute_sql", {"statements": [
            "CREATE TABLE brk (id INTEGER PRIMARY KEY, v TEXT)",
            "INSERT INTO brk VALUES (1, 'kept-then-undone')",
            "INSERT INTO brk VALUES (2)",  # wrong column count -> fails
        ]})
        out = _call_tool("query", {"sql": "SELECT count(*) AS n FROM brk", "connection": name})
        # the CREATE TABLE succeeded pre-failure (SQLite DDL is non-transactional
        # on some engines) — the ROWS must be gone; that is the all-or-nothing part
        assert "'n': 0" in out, out

    def test_batch_failure_reports_failing_index(self, sweep_db):
        name, _url = sweep_db
        out = _call_tool("execute_sql", {"statements": [
            f"CREATE TABLE brk2_{name} (id INTEGER PRIMARY KEY)",
            f"INSERT INTO brk2_{name} VALUES ('not-an-int')",
        ]})
        assert "ROLLED BACK at statement 1" in out, out

    def test_batch_param_guards(self):
        out = _call_tool("execute_sql", {"sql": "SELECT 1", "statements": ["SELECT 1"]})
        assert "either sql or statements" in out
        out = _call_tool("execute_sql", {"statements": []})
        assert "at least one statement" in out
        out = _call_tool("execute_sql", {"statements": ["SELECT 1"] * 51})
        assert "at most 50" in out


class TestInteractiveTransactions:
    def test_full_rollback_round_trip(self, sweep_db):
        name, _url = sweep_db
        _call_tool("execute_sql", {"sql": "CREATE TABLE it (id INTEGER PRIMARY KEY, v TEXT)", "connection": name})
        begin = _call_tool("begin_transaction", {"connection": name})
        assert "Transaction " in begin and "opened on" in begin, begin
        tx_id = begin.split("Transaction ")[1].split(" ")[0]

        out = _call_tool("execute_sql", {
            "sql": "INSERT INTO it VALUES (1, 'in-tx')", "connection": name, "tx_id": tx_id,
        })
        assert "not committed" in out, out

        # visible inside the tx, invisible outside
        inside = _call_tool("query", {"sql": "SELECT count(*) AS n FROM it", "tx_id": tx_id})
        assert "'n': 1" in inside, inside
        outside = _call_tool("query", {"sql": "SELECT count(*) AS n FROM it", "connection": name})
        assert "'n': 0" in outside, outside

        rb = _call_tool("rollback_transaction", {"tx_id": tx_id})
        assert "rolled back" in rb, rb
        after = _call_tool("query", {"sql": "SELECT count(*) AS n FROM it", "connection": name})
        assert "'n': 0" in after, after

    def test_full_commit_round_trip(self, sweep_db):
        name, _url = sweep_db
        _call_tool("execute_sql", {"sql": "CREATE TABLE ct (id INTEGER PRIMARY KEY)", "connection": name})
        begin = _call_tool("begin_transaction", {})
        assert "opened on" in begin, begin
        tx_id = begin.split("Transaction ")[1].split(" ")[0]

        _call_tool("execute_sql", {"sql": "INSERT INTO ct VALUES (1)", "connection": name, "tx_id": tx_id})
        cm = _call_tool("commit_transaction", {"tx_id": tx_id})
        assert "committed" in cm, cm
        after = _call_tool("query", {"sql": "SELECT count(*) AS n FROM ct", "connection": name})
        assert "'n': 1" in after, after

    def test_query_read_guard_inside_tx(self, sweep_db):
        name, _url = sweep_db
        begin = _call_tool("begin_transaction", {"connection": name})
        tx_id = begin.split("Transaction ")[1].split(" ")[0]
        out = _call_tool("query", {"sql": "DELETE FROM nothing", "tx_id": tx_id})
        assert "query() is read-only" in out, out
        _call_tool("rollback_transaction", {"tx_id": tx_id})

    def test_second_begin_rejected(self, sweep_db):
        name, _url = sweep_db
        first = _call_tool("begin_transaction", {"connection": name})
        tx_id = first.split("Transaction ")[1].split(" ")[0]
        second = _call_tool("begin_transaction", {"connection": name})
        assert "already has an active transaction" in second, second
        _call_tool("rollback_transaction", {"tx_id": tx_id})

    def test_tx_connection_mismatch_rejected(self, sweep_db):
        name, _url = sweep_db
        other = f"other{uuid.uuid4().hex[:4]}"
        from connections import REGISTRY

        REGISTRY.connect(other, "libsql", {"url": f"file:/tmp/dbmcp_other_{uuid.uuid4().hex[:6]}.db"})
        begin = _call_tool("begin_transaction", {"connection": name})
        tx_id = begin.split("Transaction ")[1].split(" ")[0]
        out = _call_tool("query", {"sql": "SELECT 1", "tx_id": tx_id, "connection": other})
        assert "belongs to connection" in out, out
        _call_tool("rollback_transaction", {"tx_id": tx_id})

    def test_commit_unknown_and_replayed_id(self, sweep_db):
        out = _call_tool("commit_transaction", {"tx_id": "deadbeef" * 4})
        assert "Unknown or already-finished" in out, out
        begin = _call_tool("begin_transaction", {"connection": None})
        tx_id = begin.split("Transaction ")[1].split(" ")[0]
        _call_tool("commit_transaction", {"tx_id": tx_id})
        replay = _call_tool("commit_transaction", {"tx_id": tx_id})
        assert "Unknown or already-finished" in replay, replay

    def test_disconnect_refused_with_active_tx(self, sweep_db):
        name, _url = sweep_db
        begin = _call_tool("begin_transaction", {"connection": name})
        tx_id = begin.split("Transaction ")[1].split(" ")[0]
        out = _call_tool("disconnect_database", {"name": name})
        assert "active transaction" in out, out
        _call_tool("rollback_transaction", {"tx_id": tx_id})
        out = _call_tool("disconnect_database", {"name": name})
        assert "Disconnected" in out, out


class TestReaper:
    def test_reap_idle_transactions(self, clean_registry):
        import time

        from connections import REGISTRY

        a = f"reap{uuid.uuid4().hex[:4]}"
        b = f"keep{uuid.uuid4().hex[:4]}"
        db = f"file:/tmp/dbmcp_reap_{uuid.uuid4().hex[:6]}.db"
        REGISTRY.connect(a, "libsql", {"url": db})
        REGISTRY.connect(b, "libsql", {"url": db})
        _entry_a, tx_a = REGISTRY.begin_tx(a)
        _entry_b, tx_b = REGISTRY.begin_tx(b)

        ea = REGISTRY.get(a)
        ea.tx_last_used = time.monotonic() - 999  # artificially idle

        reaped = REGISTRY.reap_idle_txs(60)
        assert reaped == 1
        assert ea.tx_id is None                      # reaped
        assert REGISTRY.get(b).tx_id == tx_b         # still active
        REGISTRY.finish_tx(tx_b, commit=False)

    def test_reaper_disabled_with_zero_timeout(self, monkeypatch):
        monkeypatch.setenv("DB_TX_IDLE_TIMEOUT", "0")
        import db_tools

        assert db_tools.start_tx_reaper() is None  # disabled, no task

    def test_begin_starts_reaper_once(self, sweep_db, monkeypatch):
        monkeypatch.setenv("DB_TX_IDLE_TIMEOUT", "300")
        import db_tools

        name, _url = sweep_db
        db_tools._reaper_started = False  # reset module flag for this test
        db_tools._reaper_task = None
        try:
            begin = _call_tool("begin_transaction", {"connection": name})
            assert "opened on" in begin, begin
            # the reaper was created exactly once (bound to the call's loop)
            assert db_tools._reaper_started is True
            assert db_tools._reaper_task is not None
            tx_id = begin.split("Transaction ")[1].split(" ")[0]
            _call_tool("rollback_transaction", {"tx_id": tx_id})
        finally:
            if db_tools._reaper_task is not None:
                try:
                    db_tools._reaper_task.cancel()
                except RuntimeError:
                    pass  # task's loop already closed (in-memory-client artifact)
            db_tools._reaper_started = False
            db_tools._reaper_task = None
