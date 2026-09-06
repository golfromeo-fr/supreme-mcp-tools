"""Postgres dialect — psycopg 3 with a psycopg_pool connection pool.

Catalog SQL VERIFIED live against POSTGRES_TEST_DSN (2026-09-06 probe):
note that information_schema/pg_catalog with current_schema() do NOT see
TEMP tables — describe/list target the default schema's real tables.
"""
import contextlib
from typing import Any

from ._base import DbDialect


class _SingleConnShim:
    """Fallback when psycopg_pool is unavailable: connect per acquire."""

    def __init__(self, kwargs: dict):
        self._kwargs = kwargs

    @contextlib.contextmanager
    def connection(self):
        import psycopg

        with psycopg.connect(**self._kwargs) as conn:
            yield conn


class PostgresDialect(DbDialect):
    name = "postgres"
    REQUIRED_PARAMS = ("host", "dbname", "user", "password")

    def connect(self, params: dict) -> Any:
        kwargs = {
            "host": params["host"],
            "port": int(params.get("port", 5432)),
            "dbname": params["dbname"],
            "user": params["user"],
            "password": params["password"],
            "connect_timeout": int(params.get("connect_timeout", 10)),
            "autocommit": True,
        }
        try:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row

            pool_kwargs = dict(kwargs)
            pool_kwargs["row_factory"] = dict_row
            pool_kwargs["options"] = (
                f"-c statement_timeout={int(__import__('os').environ.get('DB_QUERY_TIMEOUT_MS', '30000'))}"
            )
            pool = ConnectionPool(
                min_size=1, max_size=5, open=False, kwargs=pool_kwargs
            )
            pool.open(wait=True)
            return pool
        except Exception:
            # Pool unavailable or failed to open — per-connection fallback
            from core import logger

            logger.info("psycopg_pool unavailable/failed; using per-connection fallback")
            return _SingleConnShim(kwargs)

    def close(self, handle) -> None:
        close = getattr(handle, "close", None)
        if close:
            close()

    def ping(self, handle) -> None:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()

    def run_select(self, handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchmany(max_rows + 1)
                data = [dict(r) for r in rows]
        truncated = len(data) > max_rows
        return data[:max_rows], truncated

    def execute(self, handle, sql: str) -> int:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                return cursor.rowcount

    def list_tables(self, handle) -> list[dict]:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT c.relname AS name, COALESCE(obj_description(c.oid), '') AS comment "
                    "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = current_schema() AND c.relkind = 'r' ORDER BY 1"
                )
                rows = cursor.fetchall()
        return [{"name": r["name"], "comment": r["comment"] or ""} for r in rows]

    def describe_table(self, handle, table: str) -> dict:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (table,),
                )
                columns = cursor.fetchall()

                cursor.execute(
                    "SELECT conname, contype FROM pg_constraint con "
                    "JOIN pg_class rel ON rel.oid = con.conrelid "
                    "JOIN pg_namespace ns ON ns.oid = rel.relnamespace "
                    "WHERE ns.nspname = current_schema() AND rel.relname = %s",
                    (table,),
                )
                constraints = cursor.fetchall()

                cursor.execute(
                    "SELECT tc.constraint_name, kcu.column_name, ccu.table_name, "
                    "ccu.column_name AS ref_column "
                    "FROM information_schema.table_constraints tc "
                    "JOIN information_schema.key_column_usage kcu "
                    "ON kcu.constraint_name = tc.constraint_name "
                    "JOIN information_schema.constraint_column_usage ccu "
                    "ON ccu.constraint_name = tc.constraint_name "
                    "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s",
                    (table,),
                )
                foreign_keys = cursor.fetchall()

        type_map = {"p": "PRIMARY", "u": "UNIQUE", "f": "FOREIGN", "c": "CHECK"}
        return {
            "columns": [
                {
                    "name": c["column_name"],
                    "type": c["data_type"],
                    "nullable": c["is_nullable"] == "YES",
                    "comment": "",
                }
                for c in columns
            ],
            "constraints": [
                {"name": c["conname"], "type": type_map.get(c["contype"], c["contype"])}
                for c in constraints
            ],
            "foreign_keys": [
                {
                    "name": f["constraint_name"],
                    "column": f["column_name"],
                    "ref_table": f["table_name"],
                    "ref_column": f["ref_column"],
                }
                for f in foreign_keys
            ],
        }

    def explain(self, handle, sql: str) -> str:
        with handle.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"EXPLAIN (FORMAT TEXT) {sql}")
                rows = cursor.fetchall()
        return "\n".join(list(r.values())[0] for r in rows)

    def format_error(self, e: Exception) -> dict:
        message = str(e).splitlines()[0] if str(e) else "unknown error"
        return {
            "error": "DB_ERROR",
            "code": getattr(e, "sqlstate", None),
            "message": message,
            "offset": None,
        }
