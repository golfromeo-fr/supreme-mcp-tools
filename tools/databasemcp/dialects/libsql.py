"""libSQL dialect — local file / in-memory / Turso (libsql://) backends.

Cursor shapes VERIFIED live (2026-09-06 probe, libsql_experimental 0.0.55):
fetchmany exists; description is 7-tuples; PRAGMA table_info rows are
(cid, name, type, notnull, dflt_value, pk); PRAGMA foreign_key_list rows
are (id, seq, table, from, to, on_update, on_delete, match);
EXPLAIN QUERY PLAN text is the LAST tuple element; multi-statement
strings execute as-is; PRAGMA on a missing table returns [].

Concurrency: one shared autocommit connection, NO Python lock — the
libsql_experimental C binding serializes statements via an internal mutex
(same finding as tools/shared/impls/turso_sql.py:62-71).
"""
from typing import Any

from ._base import DbDialect


def _reject_multi_statement(sql: str) -> None:
    """Quote-aware multi-statement rejection (libsql silently runs only the
    FIRST statement of a multi-statement string on file DBs)."""
    body = sql.rstrip().rstrip(";")
    in_str = False
    for ch in body:
        if ch == "'":
            in_str = not in_str
        elif ch == ";" and not in_str:
            raise ValueError(
                "Multiple statements detected; execute one statement at a time."
            )


def _connect_params(params: dict) -> tuple[str, str | None]:
    """(url, auth_token) from preset/connection params."""
    return params["url"], params.get("auth_token")


class LibsqlDialect(DbDialect):
    name = "libsql"
    REQUIRED_PARAMS = ("url",)

    def connect(self, params: dict) -> Any:
        import libsql_experimental as libsql

        if params.get("auth_token"):
            conn = libsql.connect(params["url"], auth_token=params["auth_token"])
        else:
            conn = libsql.connect(params["url"])
        conn.autocommit = True
        return conn

    def close(self, handle) -> None:
        close = getattr(handle, "close", None)
        if close:
            close()

    def ping(self, handle) -> None:
        handle.execute("SELECT 1").fetchall()

    def run_select(self, handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        cursor = handle.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        cols = [c[0] for c in cursor.description]
        data = [dict(zip(cols, row)) for row in rows]
        truncated = len(data) > max_rows
        return data[:max_rows], truncated

    def execute(self, handle, sql: str) -> int:
        # Multi-statement guard: libsql silently runs only the FIRST
        # statement of a multi-statement string on file DBs (probe
        # correction, 2026-09-06). Reject instead of partially executing.
        _reject_multi_statement(sql)
        cursor = handle.execute(sql)
        return cursor.rowcount

    def list_tables(self, handle) -> list[dict]:
        rows = handle.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [{"name": r[0], "comment": ""} for r in rows]

    def describe_table(self, handle, table: str) -> dict:
        # PRAGMA interpolation guard: reject quote/NUL in table names
        if '"' in table or "\0" in table:
            raise ValueError(f"Invalid table name: {table!r}")
        columns = handle.execute(f'PRAGMA table_info("{table}")').fetchall()
        fks = handle.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()

        return {
            "columns": [
                {
                    "name": c[1],
                    "type": c[2] or "",
                    "nullable": not c[3],
                    "comment": "",
                }
                for c in columns
            ],
            "constraints": [
                {"name": f"pk_{c[1]}", "type": "PRIMARY"} for c in columns if c[5]
            ],
            "foreign_keys": [
                {
                    "name": f"fk_{f[0]}",
                    "column": f[3],
                    "ref_table": f[2],
                    "ref_column": f[4],
                }
                for f in fks
            ],
        }

    def explain(self, handle, sql: str) -> str:
        rows = handle.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        return "\n".join(str(r[-1]) for r in rows)

    def format_error(self, e: Exception) -> dict:
        return {
            "error": "DB_ERROR",
            "code": None,
            "message": str(e),
            "offset": None,
        }

    # ------------------------------------------------------------------
    # Transactions (E4): a dedicated SECOND connection to the same DB with
    # autocommit off. P0-a/P0-b verified live (2026-09-07): autocommit=False
    # works, commit/rollback methods exist, uncommitted writes are invisible
    # to other connections until commit.
    # ------------------------------------------------------------------

    def open_tx(self, handle, params: dict) -> Any:
        import libsql_experimental as libsql

        url, auth_token = _connect_params(params)
        if auth_token:
            tx = libsql.connect(url, auth_token=auth_token)
        else:
            tx = libsql.connect(url)
        tx.autocommit = False
        return tx

    def select_tx(self, tx_handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        cursor = tx_handle.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        cols = [c[0] for c in cursor.description]
        data = [dict(zip(cols, row)) for row in rows]
        truncated = len(data) > max_rows
        return data[:max_rows], truncated

    def execute_tx(self, tx_handle, sql: str) -> int:
        _reject_multi_statement(sql)
        return tx_handle.execute(sql).rowcount

    def commit_tx(self, tx_handle) -> None:
        tx_handle.commit()

    def rollback_tx(self, tx_handle) -> None:
        tx_handle.rollback()

    def close_tx(self, handle, tx_handle) -> None:
        try:
            tx_handle.rollback()  # no-op when the caller already committed
        except Exception:
            pass
        tx_handle.close()
