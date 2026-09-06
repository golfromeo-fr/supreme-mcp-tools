"""P2 — PostgresDialect contract against a real Postgres (skips when down).

VERIFIED (2026-09-06 probe): information_schema/current_schema() do NOT
see TEMP tables — these tests use real dbmcp_test_* tables with DROP
CASCADE teardown.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))



def _make_db(pg_dsn):
    """Build a dialect handle from a DSN via the pool kwargs path."""
    u = urlparse(pg_dsn)
    from dialects.postgres import PostgresDialect

    dialect = PostgresDialect()
    handle = dialect.connect({
        "host": u.hostname,
        "port": u.port or 5432,
        "dbname": (u.path or "/").lstrip("/"),
        "user": unquote(u.username or "postgres"),
        "password": unquote(u.password or ""),
    })
    return dialect, handle


@pytest.fixture()
def pgdb(pg_dsn):
    if pg_dsn is None:
        pytest.skip("Postgres not configured/reachable (pg_dsn fixture)")
    dialect, handle = _make_db(pg_dsn)
    with handle.connection() as conn:  # pre-drop leftovers from earlier failed runs
        with conn.cursor() as cur:
            cur.execute(
                "SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND relname LIKE 'dbmcp_test_%'"
            )
            for row in cur.fetchall():
                # dict_row: access by key — tuple-unpacking a dict yields its KEYS
                cur.execute(f'DROP TABLE IF EXISTS "{row["relname"]}" CASCADE')
    yield dialect, handle
    with handle.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND relname LIKE 'dbmcp_test_%'"
            )
            for row in cur.fetchall():
                # dict_row: access by key — tuple-unpacking a dict yields its KEYS
                cur.execute(f'DROP TABLE IF EXISTS "{row["relname"]}" CASCADE')


class TestPostgresContract:
    def test_ping(self, pgdb):
        dialect, handle = pgdb
        dialect.ping(handle)

    def test_execute_rowcount(self, pgdb):
        dialect, handle = pgdb
        dialect.execute(handle, "CREATE TABLE dbmcp_test_t (id INTEGER PRIMARY KEY, label TEXT)")
        assert dialect.execute(handle, "INSERT INTO dbmcp_test_t VALUES (1, 'a')") == 1

    def test_run_select_truncation(self, pgdb):
        dialect, handle = pgdb
        dialect.execute(handle, "CREATE TABLE dbmcp_test_t (id INTEGER PRIMARY KEY)")
        for i in range(5):
            dialect.execute(handle, f"INSERT INTO dbmcp_test_t VALUES ({i})")
        rows, truncated = dialect.run_select(handle, "SELECT * FROM dbmcp_test_t", max_rows=3)
        assert len(rows) == 3 and truncated is True
        rows, truncated = dialect.run_select(handle, "SELECT * FROM dbmcp_test_t", max_rows=10)
        assert len(rows) == 5 and truncated is False

    def test_list_tables_sees_real_not_temp(self, pgdb):
        dialect, handle = pgdb
        dialect.execute(handle, "CREATE TABLE dbmcp_test_real (x INTEGER)")
        names = [t["name"] for t in dialect.list_tables(handle)]
        assert "dbmcp_test_real" in names

    def test_describe_table_with_fk(self, pgdb):
        dialect, handle = pgdb
        dialect.execute(handle, "CREATE TABLE dbmcp_test_parent (id INTEGER PRIMARY KEY, name TEXT)")
        dialect.execute(handle, "CREATE TABLE dbmcp_test_child (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES dbmcp_test_parent(id))")
        desc = dialect.describe_table(handle, "dbmcp_test_child")
        cols = {c["name"]: c for c in desc["columns"]}
        assert cols["pid"]["nullable"] is True
        assert any(c["type"] == "PRIMARY" for c in desc["constraints"])
        fks = desc["foreign_keys"]
        assert len(fks) == 1
        assert fks[0]["ref_table"] == "dbmcp_test_parent"
        assert fks[0]["ref_column"] == "id"

    def test_describe_missing_table_returns_empty(self, pgdb):
        dialect, handle = pgdb
        desc = dialect.describe_table(handle, "dbmcp_test_missing_xyz")
        assert desc["columns"] == []

    def test_explain(self, pgdb):
        dialect, handle = pgdb
        dialect.execute(handle, "CREATE TABLE dbmcp_test_t (id INTEGER)")
        plan = dialect.explain(handle, "SELECT * FROM dbmcp_test_t")
        assert "Seq Scan" in plan or "scan" in plan.lower()

    def test_format_error_sqlstate(self, pgdb):
        dialect, handle = pgdb
        try:
            dialect.execute(handle, "THIS IS NOT SQL")
            raised = None
        except Exception as e:
            raised = e
        assert raised is not None
        err = dialect.format_error(raised)
        assert err["code"] == "42601"  # VERIFIED: psycopg SyntaxError sqlstate
