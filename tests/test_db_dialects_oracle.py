"""P2 — OracleDialect against a mocked oracledb module (no Oracle needed).

Pins: create_pool usage (VERIFIED fact B — NOT SessionPool direct), env
pool sizing, ping SQL, :table_name bind style in describe_table,
call_timeout on acquire, ORA error-code parsing.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


try:
    import oracledb  # noqa: F401

    HAS_ORACLEDB = True
except ImportError:
    HAS_ORACLEDB = False

pytestmark = pytest.mark.skipif(not HAS_ORACLEDB, reason="oracledb not installed")

PARAMS = {"user": "scott", "password": "tiger", "host": "db", "port": "1521", "service_name": "XE"}


@pytest.fixture()
def oracle_mod(monkeypatch):
    import tools.databasemcp.dialects.oracle as mod

    fake = mock.MagicMock()
    # Real-ish DatabaseError class so isinstance() checks in format_error work
    fake.DatabaseError = type("DatabaseError", (Exception,), {})
    fake.POOL_GETMODE_WAIT = "WAIT"
    fake.makedsn = mock.MagicMock(return_value="DSN-STRING")
    pool = mock.MagicMock()
    fake.create_pool.return_value = pool

    # acquire() context manager yielding a mock connection
    conn = mock.MagicMock()
    cursor = mock.MagicMock()
    conn.cursor.return_value = cursor
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = conn
    ctx.__exit__.return_value = False
    pool.acquire.return_value = ctx

    monkeypatch.setattr(mod, "oracledb", fake)
    yield mod, fake, pool, conn, cursor


class TestOracleDialect:
    def test_connect_uses_create_pool_with_env_sizing(self, oracle_mod, monkeypatch):
        mod, fake, pool, conn, cursor = oracle_mod
        monkeypatch.setenv("ORACLE_MIN_CONNECTIONS", "2")
        monkeypatch.setenv("ORACLE_MAX_CONNECTIONS", "7")
        dialect = mod.OracleDialect()
        handle = dialect.connect(PARAMS)
        assert handle is pool
        fake.create_pool.assert_called_once()
        kwargs = fake.create_pool.call_args.kwargs
        assert kwargs["user"] == "scott" and kwargs["password"] == "tiger"
        assert kwargs["dsn"] == "DSN-STRING"
        assert kwargs["min"] == 2 and kwargs["max"] == 7
        assert kwargs["getmode"] == "WAIT"

    def test_ping_runs_dual_health_check(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        dialect = mod.OracleDialect()
        dialect.ping(pool)
        cursor.execute.assert_called_with("SELECT 1 FROM DUAL")

    def test_call_timeout_set_on_acquire(self, oracle_mod, monkeypatch):
        mod, fake, pool, conn, cursor = oracle_mod
        monkeypatch.setenv("ORACLE_QUERY_TIMEOUT", "12")
        dialect = mod.OracleDialect()
        dialect.ping(pool)
        assert conn.call_timeout == 12000

    def test_run_select_fetchmany_and_dicts(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        cursor.description = [("ID", None, None, None, None, None, None), ("NAME", None, None, None, None, None, None)]
        cursor.fetchmany.return_value = [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
        dialect = mod.OracleDialect()
        rows, truncated = dialect.run_select(pool, "SELECT * FROM t", max_rows=3)
        cursor.fetchmany.assert_called_with(4)  # max_rows + 1
        assert truncated is True and len(rows) == 3
        assert rows[0] == {"ID": 1, "NAME": "a"}

    def test_describe_uses_named_binds(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        cursor.fetchall.side_effect = [[], [], []]
        dialect = mod.OracleDialect()
        desc = dialect.describe_table(pool, "T1")
        assert desc == {"columns": [], "constraints": [], "foreign_keys": []}
        bind_calls = [c for c in cursor.execute.call_args_list if c.args and c.args[1] == {"table_name": "T1"}]
        assert len(bind_calls) == 3  # columns + constraints + FKs, all :table_name bound

    def test_describe_maps_constraint_types(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        cursor.fetchall.side_effect = [
            [("ID", "NUMBER", None, None, None, "Y", None, "")],  # columns (8-tuple, real query shape)
            [("pk_T", "ID", "P"), ("fk_T", "X", "R")],         # constraints
            [("fk_T", "X", "PARENT", "ID")],                   # FKs
        ]
        dialect = mod.OracleDialect()
        desc = dialect.describe_table(pool, "T")
        assert desc["constraints"] == [{"name": "pk_T", "type": "PRIMARY"}, {"name": "fk_T", "type": "FOREIGN"}]
        assert desc["foreign_keys"] == [{"name": "fk_T", "column": "X", "ref_table": "PARENT", "ref_column": "ID"}]

    def test_format_error_parses_ora_code(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        err_obj = mock.MagicMock()
        err_obj.__str__ = mock.MagicMock(return_value="ORA-01234: something broke")
        err_obj.offset = 17
        e = fake.DatabaseError(err_obj)
        err = mod.OracleDialect().format_error(e)
        assert err == {"error": "ORA_ERROR", "code": "01234", "message": "ORA-01234: something broke", "offset": 17}

    def test_format_error_non_oracle(self, oracle_mod):
        mod, fake, pool, conn, cursor = oracle_mod
        err = mod.OracleDialect().format_error(ValueError("plain"))
        assert err["error"] == "DB_ERROR" and err["message"] == "plain"
