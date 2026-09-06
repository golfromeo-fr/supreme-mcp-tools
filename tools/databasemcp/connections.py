"""
databasemcp connection registry — named heterogeneous DB connections.

Oracle / Postgres / libSQL via the dialect layer; legacy env-default
("default" Oracle, DB_AUTOCONNECT-switchable); per-entry locks so parallel
queries on DIFFERENT connections never block each other. Replaces the
former single-connection machinery (P2/P4,
plans/databasemcp-overhaul-2026-09-06.md).
"""
import os
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any

from core import logger, metrics
from dialects import DIALECTS, get_dialect
import presets as _presets


# The legacy single-connection machinery — the global connection/
# schema_cache/table_columns_cache, get_db_connection, fetch_schema_from_cache,
# execute_query, format_oracle_error, get_pool_config — was deleted in P4:
# the registry below is the only connection path. format_oracle_error lives
# on in dialects/oracle.py as OracleDialect.format_error.


# ============================================================================
# Connection Registry (P2) — named heterogeneous connections
# ============================================================================

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
            if entry is not None:
                return entry
            # Preset bypass: an unconnected preset number or NAME alias
            # connects lazily right here — query(sql, connection="01") or
            # connection="pglocal" works without an explicit connect step.
            with self._map_lock:
                entry = self._entries.get(name)  # double-check under the lock
                if entry is not None:
                    return entry
                try:
                    preset = _presets.get_preset(name)
                except LookupError:
                    preset = None
                if preset is not None:
                    if preset.connection_name in self._entries:
                        return self._entries[preset.connection_name]
                    handle = DIALECTS[preset.dialect].connect(preset.params)
                    entry = ConnectionEntry(
                        name=preset.connection_name,
                        dialect=preset.dialect,
                        params=dict(preset.params),
                        handle=handle,
                    )
                    self._entries[entry.name] = entry
                    if self._active is None:
                        self._active = entry.name
                    logger.info(
                        f"Preset {preset.number} connected lazily as '{entry.name}' ({preset.dialect})."
                    )
                    return entry
            raise LookupError(
                f"Unknown connection '{name}'. Available: {sorted(self._entries) or 'none'}"
            )
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
