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


# ============================================================================
# Connection Registry (P2) — named heterogeneous connections
# ============================================================================

import time as _time
from dataclasses import dataclass, field
from typing import Any

from dialects import DIALECTS, get_dialect

_SECRET_KEYS = {"password", "userid", "auth_token", "token", "secret", "key"}


def _mask_params(params: dict) -> dict:
    """Copy of params with secret-named values replaced by '***'."""
    return {
        k: ("***" if str(k).lower() in _SECRET_KEYS else v)
        for k, v in (params or {}).items()
    }


@dataclass
class ConnectionEntry:
    name: str
    dialect: str
    params: dict          # may hold secrets — never log/return unmasked
    handle: Any = None
    state: str = "CONNECTED"          # CONNECTED | ERROR | CLOSED
    created_at: float = field(default_factory=_time.time)
    last_used: float = field(default_factory=_time.time)
    last_error: str | None = None
    schema_cache: dict = field(default_factory=dict)   # table -> describe_table() result
    lock: threading.Lock = field(default_factory=threading.Lock)


class ConnectionRegistry:
    """Named pool of heterogeneous DB connections.

    Map mutations (connect/disconnect/switch) serialize on _map_lock; each
    entry's handle use serializes on its own per-entry lock — parallel
    queries on DIFFERENT connections never block each other.
    """

    def __init__(self):
        self._entries: dict[str, ConnectionEntry] = {}
        self._active: str | None = None
        self._map_lock = threading.Lock()

    def connect(self, name: str, db_type: str, params: dict) -> ConnectionEntry:
        with self._map_lock:
            if name in self._entries:
                existing = self._entries[name].dialect
                raise ValueError(
                    f"Connection '{name}' already exists (dialect {existing}). "
                    "Use disconnect_database first or another name."
                )
            dialect = get_dialect(db_type)
            handle = dialect.connect(params)  # connect errors propagate (no params in message)
            entry = ConnectionEntry(
                name=name, dialect=db_type, params=dict(params), handle=handle
            )
            self._entries[name] = entry
            if self._active is None:
                self._active = name
            return entry

    def disconnect(self, name: str) -> str:
        with self._map_lock:
            entry = self._entries.get(name)
            if entry is None:
                raise LookupError(
                    f"Unknown connection '{name}'. Available: {sorted(self._entries) or 'none'}"
                )
            if not entry.lock.acquire(timeout=1):
                return f"Connection '{name}' is busy (a query is running); retry after it completes."
            try:
                try:
                    DIALECTS[entry.dialect].close(entry.handle)
                except Exception as close_err:
                    logger.warning(f"Close error for '{name}': {close_err}")
                del self._entries[name]
                if self._active == name:
                    self._active = next(iter(self._entries), None)
            finally:
                entry.lock.release()
        return self._active or "none"

    def get(self, name: str | None = None) -> ConnectionEntry:
        if name is not None:
            entry = self._entries.get(name)
            if entry is None:
                raise LookupError(
                    f"Unknown connection '{name}'. Available: {sorted(self._entries) or 'none'}"
                )
            return entry
        # No name: legacy env default (lazy), double-checked under the map
        # lock so two concurrent first calls create exactly one entry.
        with self._map_lock:
            if not self._entries:
                autoconnect = os.environ.get("DB_AUTOCONNECT", "1") != "0"
                user_id = os.environ.get("USERID")
                db_host = os.environ.get("DB_HOST")
                if autoconnect and user_id and db_host:
                    login, password = user_id.split("/", 1)
                    params = {
                        "user": login,
                        "password": password,
                        "host": db_host,
                        "port": os.environ.get("DB_PORT", "1521"),
                        "service_name": os.environ.get("DB_SERVICE_NAME", ""),
                    }
                    handle = DIALECTS["oracle"].connect(params)
                    entry = ConnectionEntry(
                        name="default", dialect="oracle", params=params, handle=handle
                    )
                    self._entries["default"] = entry
                    self._active = "default"
                    metrics["connection_count"] += 1
                    logger.info("Legacy env default Oracle connection established (lazy).")
        if self._active and self._active in self._entries:
            return self._entries[self._active]
        raise LookupError(
            "No database connection. Use connect_database(name, db_type, params) "
            "— db_type: oracle | postgres | libsql"
        )

    def set_active(self, name: str) -> ConnectionEntry:
        with self._map_lock:
            entry = self._entries.get(name)
            if entry is None:
                raise LookupError(
                    f"Unknown connection '{name}'. Available: {sorted(self._entries) or 'none'}"
                )
            self._active = name
            return entry

    def list(self) -> list[dict]:
        with self._map_lock:
            return [
                {
                    "name": e.name,
                    "dialect": e.dialect,
                    "state": e.state,
                    "active": e.name == self._active,
                    "cached_tables": len(e.schema_cache),
                    "created_at": e.created_at,
                    "last_used": e.last_used,
                    "last_error": e.last_error,
                }
                for e in self._entries.values()
            ]

    def close_all(self) -> tuple[int, int]:
        """Close every idle connection; busy entries are skipped. Returns (closed, skipped)."""
        closed = skipped = 0
        with self._map_lock:
            names = list(self._entries)
        for name in names:
            with self._map_lock:
                entry = self._entries.get(name)
                if entry is None:
                    continue
                if not entry.lock.acquire(timeout=1):
                    skipped += 1
                    continue
                try:
                    try:
                        DIALECTS[entry.dialect].close(entry.handle)
                    except Exception as close_err:
                        logger.warning(f"Close error for '{name}': {close_err}")
                    del self._entries[name]
                    closed += 1
                finally:
                    entry.lock.release()
        if self._active not in self._entries:
            self._active = next(iter(self._entries), None)
        return closed, skipped


REGISTRY = ConnectionRegistry()
