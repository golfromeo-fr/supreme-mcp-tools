"""P2 — LibsqlDialect contract, verified against a real local file DB.

Shapes pinned by the 2026-09-06 probe (see spec VERIFIED FACTS):
fetchmany exists, PRAGMA table_info=(cid,name,type,notnull,dflt,pk),
foreign_key_list=(id,seq,table,from,to,...), EXPLAIN QUERY PLAN text is
the last tuple element, multi-statement execute works.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


try:
    import libsql_experimental  # noqa: F401

    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")


@pytest.fixture()
def db(tmp_path):
    from dialects.libsql import LibsqlDialect

    dialect = LibsqlDialect()
    handle = dialect.connect({"url": f"file:{tmp_path / 'contract.db'}"})
    yield dialect, handle
    dialect.close(handle)


class TestLibsqlContract:
    def test_ping(self, db):
        dialect, handle = db
        dialect.ping(handle)  # no raise

    def test_execute_rowcount(self, db):
        dialect, handle = db
        assert dialect.execute(handle, "CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT)") >= 0
        assert dialect.execute(handle, "INSERT INTO t VALUES (1, 'a')") == 1

    def test_run_select_truncation(self, db):
        dialect, handle = db
        dialect.execute(handle, "CREATE TABLE t (id INTEGER PRIMARY KEY)")
        for i in range(5):
            dialect.execute(handle, f"INSERT INTO t VALUES ({i})")
        rows, truncated = dialect.run_select(handle, "SELECT * FROM t", max_rows=3)
        assert len(rows) == 3 and truncated is True
        assert rows[0] == {"id": 0}
        rows, truncated = dialect.run_select(handle, "SELECT * FROM t", max_rows=10)
        assert len(rows) == 5 and truncated is False

    def test_list_tables(self, db):
        dialect, handle = db
        dialect.execute(handle, "CREATE TABLE alpha (x INTEGER)")
        dialect.execute(handle, "CREATE TABLE beta (y INTEGER)")
        names = [t["name"] for t in dialect.list_tables(handle)]
        assert "alpha" in names and "beta" in names
        assert not any(n.startswith("sqlite_") for n in names)

    def test_describe_table_with_fk(self, db):
        dialect, handle = db
        dialect.execute(handle, "CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT)")
        dialect.execute(handle, "CREATE TABLE child (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES parent(id))")
        desc = dialect.describe_table(handle, "child")
        cols = {c["name"]: c for c in desc["columns"]}
        assert cols["pid"]["type"] == "INTEGER" and cols["pid"]["nullable"] is True
        assert any(c["type"] == "PRIMARY" for c in desc["constraints"])
        fks = desc["foreign_keys"]
        assert len(fks) == 1
        assert fks[0]["ref_table"] == "parent"
        assert fks[0]["ref_column"] == "id"

    def test_describe_missing_table_returns_empty(self, db):
        dialect, handle = db
        desc = dialect.describe_table(handle, "nope")
        assert desc["columns"] == []  # edge case 7 normalization happens in the tool layer

    def test_explain(self, db):
        dialect, handle = db
        dialect.execute(handle, "CREATE TABLE t (id INTEGER)")
        plan = dialect.explain(handle, "SELECT * FROM t")
        assert isinstance(plan, str) and plan  # plan text non-empty

    def test_multi_statement_rejected(self, db):
        # Probe correction: file-backed libsql silently ran only the first
        # statement — the dialect must reject, never partially execute.
        dialect, handle = db
        with pytest.raises(ValueError, match="one statement at a time"):
            dialect.execute(handle, "CREATE TABLE a (x INTEGER); CREATE TABLE b (y INTEGER)")
        names = [t["name"] for t in dialect.list_tables(handle)]
        assert "a" not in names and "b" not in names

    def test_format_error_shape(self, db):
        dialect, handle = db
        err = dialect.format_error(ValueError("boom"))
        assert err == {"error": "DB_ERROR", "code": None, "message": "boom", "offset": None}
