"""Oracle dialect — oracledb thin mode with a real session pool.

VERIFIED (2026-09-06 probe): oracledb 3.4.2 SessionPool.__init__ takes
(dsn, params, kwargs) — pools must be created via oracledb.create_pool,
which accepts user/password/dsn/min/max/getmode/timeout directly.
"""
import contextlib
import os
from typing import Any

import oracledb

from ._base import DbDialect


class OracleDialect(DbDialect):
    name = "oracle"
    REQUIRED_PARAMS = ("user", "password", "host", "port", "service_name")

    def connect(self, params: dict) -> Any:
        dsn = oracledb.makedsn(
            params["host"], int(params["port"]), service_name=params["service_name"]
        )
        return oracledb.create_pool(
            user=params["user"],
            password=params["password"],
            dsn=dsn,
            min=int(os.environ.get("ORACLE_MIN_CONNECTIONS", "1")),
            max=int(os.environ.get("ORACLE_MAX_CONNECTIONS", "10")),
            getmode=oracledb.POOL_GETMODE_WAIT,
            timeout=int(os.environ.get("ORACLE_QUERY_TIMEOUT", "30")) * 3,
        )

    def close(self, handle) -> None:
        handle.close()

    @contextlib.contextmanager
    def _acquire(self, handle):
        with handle.acquire() as conn:
            conn.call_timeout = int(os.environ.get("ORACLE_QUERY_TIMEOUT", "30")) * 1000
            yield conn

    def ping(self, handle) -> None:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            cursor.fetchall()

    def run_select(self, handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            cols = [c[0] for c in cursor.description]
            data = [dict(zip(cols, row)) for row in rows]
        truncated = len(data) > max_rows
        return data[:max_rows], truncated

    def execute(self, handle, sql: str) -> int:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rowcount = cursor.rowcount
            conn.commit()
        return rowcount

    def list_tables(self, handle) -> list[dict]:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT table_name, NVL(comments, '') FROM user_tab_comments ORDER BY table_name"
            )
            rows = cursor.fetchall()
        return [{"name": r[0], "comment": r[1] or ""} for r in rows]

    def describe_table(self, handle, table: str) -> dict:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT utc.column_name, utc.data_type, utc.data_length, utc.data_precision, utc.data_scale, utc.nullable, utc.data_default, ucc.comments
                FROM user_tab_columns utc
                LEFT JOIN user_col_comments ucc
                ON utc.table_name = ucc.table_name AND utc.column_name = ucc.column_name
                WHERE utc.table_name = :table_name
            """, {"table_name": table})
            columns = cursor.fetchall()

            cursor.execute("""
                SELECT cons.constraint_name, cols.column_name, cons.constraint_type
                FROM user_constraints cons, user_cons_columns cols
                WHERE cols.table_name = :table_name
                  AND cons.constraint_type IN ('P', 'R', 'C', 'U')
                  AND cons.constraint_name = cols.constraint_name
            """, {"table_name": table})
            constraints = cursor.fetchall()

            cursor.execute("""
                SELECT a.constraint_name, a.column_name, c_pk.table_name AS referenced_table, b.column_name AS referenced_column
                FROM user_cons_columns a
                JOIN user_constraints c ON a.constraint_name = c.constraint_name
                JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
                JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name AND a.position = b.position
                WHERE c.constraint_type = 'R' AND a.table_name = :table_name
            """, {"table_name": table})
            foreign_keys = cursor.fetchall()

        type_map = {"P": "PRIMARY", "U": "UNIQUE", "R": "FOREIGN", "C": "CHECK"}
        return {
            "columns": [
                {"name": c[0], "type": c[1], "nullable": c[5] == "Y", "comment": c[7] or ""}
                for c in columns
            ],
            "constraints": [
                {"name": c[0], "type": type_map.get(c[2], c[2])} for c in constraints
            ],
            "foreign_keys": [
                {"name": f[0], "column": f[1], "ref_table": f[2], "ref_column": f[3]}
                for f in foreign_keys
            ],
        }

    def explain(self, handle, sql: str) -> str:
        with self._acquire(handle) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM PLAN_TABLE")
            except Exception:
                pass
            cursor.execute(f"EXPLAIN PLAN FOR {sql}")
            try:
                cursor.execute("SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY())")
                plan_rows = cursor.fetchall()
                plan_text = "\n".join(row[0] for row in plan_rows)
            except Exception:
                cursor.execute("SELECT * FROM PLAN_TABLE")
                plan_text = str(cursor.fetchall())
        return plan_text

    def format_error(self, e: Exception) -> dict:
        try:
            if isinstance(e, oracledb.DatabaseError):
                error_obj = e.args[0]
                error_msg = str(error_obj)
                error_code = None
                if error_msg.startswith('ORA-'):
                    error_code = error_msg[4:9]
                return {
                    "error": "ORA_ERROR",
                    "code": error_code,
                    "message": error_msg,
                    "offset": getattr(error_obj, 'offset', None),
                }
        except Exception as fmt_err:
            from core import logger
            logger.error(f"Error formatting Oracle error: {fmt_err}")
        return {"error": "DB_ERROR", "code": None, "message": str(e)}

    # ------------------------------------------------------------------
    # Transactions (E4): the session pool is the only connection source, so
    # a transaction HOLDS one acquired session (max ORACLE_MAX_CONNECTIONS;
    # getmode WAIT + pool timeout make concurrent acquires wait). Held
    # sessions are the pool-exhaustion trade-off documented in the plan.
    # P0-d stays documented-only: no live Oracle instance in the workbench.
    # ------------------------------------------------------------------

    def open_tx(self, handle, params: dict) -> Any:
        conn = handle.acquire()
        conn.call_timeout = int(os.environ.get("ORACLE_QUERY_TIMEOUT", "30")) * 1000
        return conn

    def select_tx(self, tx_handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        cursor = tx_handle.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        cols = [c[0] for c in cursor.description]
        data = [dict(zip(cols, row)) for row in rows]
        truncated = len(data) > max_rows
        return data[:max_rows], truncated

    def execute_tx(self, tx_handle, sql: str) -> int:
        cursor = tx_handle.cursor()
        cursor.execute(sql)
        return cursor.rowcount  # NO commit — explicit commit_tx only

    def commit_tx(self, tx_handle) -> None:
        tx_handle.commit()

    def rollback_tx(self, tx_handle) -> None:
        tx_handle.rollback()

    def close_tx(self, handle, tx_handle) -> None:
        try:
            tx_handle.rollback()  # no-op when the caller already committed
        except Exception:
            pass
        release = getattr(handle, "release", None)
        if release:
            release(tx_handle)
        else:
            close = getattr(tx_handle, "close", None)
            if close:
                close()
