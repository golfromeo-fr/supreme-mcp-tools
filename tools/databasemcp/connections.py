"""
databasemcp connection machinery (P1: legacy single-Oracle-connection path,
ported verbatim from the former oraclemcp monolith).

P2 replaces this with the multi-database ConnectionRegistry (Oracle pool /
Postgres / libSQL) per plans/databasemcp-overhaul-2026-09-06.md.
"""
import os
import time
import threading
from typing import Any

import oracledb

from core import logger, metrics


# Thread-safety guard for the shared connection and caches.
#
# databasemcp runs behind FastMCP which may serve concurrent requests via
# asyncio.to_thread.  A single global `connection` and two global dicts
# (`schema_cache`, `table_columns_cache`) are shared across those threads.
# Without locking, concurrent cursor operations on one Oracle connection
# produce garbled results or InterfaceError.
#
# The lock serialises:
#   - connection check / reconnect / reset-to-None  (get_db_connection)
#   - cache writes (fetch_schema_from_cache, clear_cache)
#
# Known limitations (safe to ship, fix when Oracle is available for CI):
#   - Serialises ALL DB access through one lock — limits throughput under
#     heavy concurrency.  Replace with `oracledb.SessionPool` (the env
#     vars ORACLE_MIN/MAX_CONNECTIONS are already declared in
#     get_pool_config() but unused) to allow parallel queries.
#   - Lock scope covers cursor creation + health-check query ("SELECT 1
#     FROM DUAL").  Ideally only the global-state mutation should be
#     locked; the cursor work should happen outside.  Tightening the
#     scope requires refactoring the function contract and is deferred
#     until we can test against a live Oracle instance.
#   - `metrics` dict is intentionally NOT locked — it is only used for
#     counters and a stale read is acceptable.
_db_lock = threading.Lock()
connection = None
schema_cache = {}
table_columns_cache = {}


def get_db_connection():
    """Get database connection with automatic reconnection."""
    global connection
    with _db_lock:
        try:
            if connection is None:
                raise oracledb.DatabaseError("Connection is not established")
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
        except oracledb.DatabaseError:
            try:
                user_id = os.getenv('USERID')
                if not user_id:
                    raise OSError("USERID environment variable not set")

                login, password = user_id.split('/')

                db_host = os.getenv('DB_HOST')
                db_port = int(os.getenv('DB_PORT') or 1521)
                db_service_name = os.getenv('DB_SERVICE_NAME')
                if not db_host or not db_port or not db_service_name:
                    raise OSError("Database connection environment variables not set")

                dsn_tns = oracledb.makedsn(db_host, db_port, service_name=db_service_name)
                connection = oracledb.connect(user=login, password=password, dsn=dsn_tns)
                metrics["connection_count"] += 1
                logger.info("Database connection re-established successfully.")
            except Exception as e:
                logger.error(f"Error re-establishing database connection: {e}")
                metrics["connection_errors"] += 1
                connection = None
                raise
        return connection


def fetch_schema_from_cache(table_name):
    """Fetch schema for a table, using cache when possible."""
    global schema_cache
    logger.info(f"Querying schema for table: {table_name}")
    if table_name not in schema_cache:
        return "Table not found"

    if schema_cache[table_name] is None:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT utc.column_name, utc.data_type, utc.data_length, utc.data_precision, utc.data_scale, utc.nullable, utc.data_default, ucc.comments
                FROM user_tab_columns utc
                LEFT JOIN user_col_comments ucc
                ON utc.table_name = ucc.table_name AND utc.column_name = ucc.column_name
                WHERE utc.table_name = :table_name
            """, {"table_name": table_name})
            columns = cursor.fetchall()

            cursor.execute("""
                SELECT cols.column_name, cons.constraint_type, cons.search_condition
                FROM user_constraints cons, user_cons_columns cols
                WHERE cols.table_name = :table_name
                  AND cons.constraint_type IN ('P', 'R', 'C', 'U')
                  AND cons.constraint_name = cols.constraint_name
            """, {"table_name": table_name})
            constraints = cursor.fetchall()

            cursor.execute("""
                SELECT a.constraint_name, a.column_name, c_pk.table_name AS referenced_table, b.column_name AS referenced_column
                FROM user_cons_columns a
                JOIN user_constraints c ON a.constraint_name = c.constraint_name
                JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
                JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name AND a.position = b.position
                WHERE c.constraint_type = 'R' AND a.table_name = :table_name
            """, {"table_name": table_name})
            foreign_keys = cursor.fetchall()

            with _db_lock:
                schema_cache[table_name] = {
                    "columns": columns,
                    "constraints": constraints,
                    "foreign_keys": foreign_keys
                }
            metrics["schema_lookups"] += 1
            logger.info(f"Schema details for table {table_name} cached successfully.")
        except Exception as e:
            logger.error(f"Error fetching schema details for table {table_name}: {e}")
            with _db_lock:
                # NOTE (ported verbatim): assigns a LOCAL name in the original —
                # `connection` was never declared global here. Kept for P1
                # behavior parity; dies with the P2 registry rewrite.
                connection = None
            return "Error fetching schema details"

    return schema_cache[table_name]


def format_oracle_error(e):
    """Format Oracle error details into a structured response."""
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
                "offset": getattr(error_obj, 'offset', None)
            }
    except Exception as format_error:
        logger.error(f"Error formatting Oracle error: {format_error}")

    return {
        "error": "DB_ERROR",
        "code": None,
        "message": str(e)
    }


def execute_query(sql_query):
    """Execute a SQL query and return results or error details."""
    start_time = time.time()
    logger.info(f"[SQL] Executing query: {sql_query[:200]}{'...' if len(sql_query) > 200 else ''}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        metrics["query_count"] += 1
        return {"success": True, "data": results}
    except oracledb.DatabaseError as e:
        metrics["query_errors"] += 1
        error_details = format_oracle_error(e)
        logger.error(f"Oracle error executing query: {error_details}")
        return {"success": False, "error": error_details}
    except Exception as e:
        metrics["query_errors"] += 1
        logger.error(f"Error executing query: {e}")
        return {"success": False, "error": {"error": "EXECUTION_ERROR", "message": str(e)}}
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        metrics["total_query_time_ms"] += elapsed_ms


def get_pool_config() -> dict:
    """Get pool config from env vars (hot-reload)."""
    return {
        "min_connections": int(os.environ.get("ORACLE_MIN_CONNECTIONS", "1")),
        "max_connections": int(os.environ.get("ORACLE_MAX_CONNECTIONS", "10")),
        "increment": 1,
        "query_timeout_seconds": int(os.environ.get("ORACLE_QUERY_TIMEOUT", "30")),
    }
