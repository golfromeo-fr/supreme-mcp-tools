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
import uuid
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
    # Active transaction (E4) — at most ONE per entry. tx_lock serializes
    # begin/commit/rollback/reap against in-flight tx statements (it is NOT
    # the entry lock, which is never held across calls).
    tx_id: str | None = None
    tx_handle: Any = None
    tx_opened_at: float | None = None    # monotonic()
    tx_last_used: float | None = None    # monotonic()
    tx_lock: threading.Lock = field(default_factory=threading.Lock)


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
            if entry.tx_id:
                age = int(_time.monotonic() - (entry.tx_last_used or entry.tx_opened_at or 0))
                return (
                    f"Connection '{name}' has an active transaction (tx {entry.tx_id[:8]}…, "
                    f"idle {age}s). Commit or roll it back first."
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
                    "tx": e.tx_id[:8] if e.tx_id else None,
                }
                for e in self._entries.values()
            ]

    def close_all(self) -> tuple[int, int]:
        """Close every connection; busy entries are skipped. Active
        transactions are force-rolled-back first (admin escape hatch).
        Returns (closed, skipped)."""
        closed = skipped = 0
        with self._map_lock:
            names = list(self._entries)
        for name in names:
            with self._map_lock:
                entry = self._entries.get(name)
                if entry is None:
                    continue
                if entry.tx_id:
                    # force-rollback (reset_connections is the admin path)
                    if not entry.tx_lock.acquire(timeout=1):
                        skipped += 1
                        continue
                    try:
                        self._abort_tx(entry)
                    finally:
                        entry.tx_lock.release()
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

    # ------------------------------------------------------------------
    # Transactions (E4) — at most ONE active tx per entry.
    # ------------------------------------------------------------------

    def begin_tx(self, name: str | None = None) -> tuple[ConnectionEntry, str]:
        """Open a transaction on the named (or active) connection; the
        preset bypass applies. Returns (entry, tx_id)."""
        entry = self.get(name)
        with entry.tx_lock:
            if entry.tx_id:
                age = int(_time.monotonic() - (entry.tx_last_used or entry.tx_opened_at or 0))
                raise RuntimeError(
                    f"Connection '{entry.name}' already has an active transaction "
                    f"(tx {entry.tx_id[:8]}…, idle {age}s). One transaction per connection."
                )
            tx_handle = DIALECTS[entry.dialect].open_tx(entry.handle, entry.params)
            entry.tx_id = uuid.uuid4().hex
            entry.tx_handle = tx_handle
            entry.tx_opened_at = _time.monotonic()
            entry.tx_last_used = entry.tx_opened_at
            metrics["transactions_begun"] += 1
            logger.info(
                f"[databasemcp] tx {entry.tx_id[:8]}… opened on '{entry.name}' ({entry.dialect})"
            )
            return entry, entry.tx_id

    def locate_tx(self, tx_id: str) -> ConnectionEntry:
        """Find the entry owning tx_id (map-lock protected)."""
        with self._map_lock:
            for entry in self._entries.values():
                if entry.tx_id == tx_id:
                    return entry
        raise LookupError(
            f"Unknown or already-finished transaction '{(tx_id or '')[:8]}…'. "
            "Open one with begin_transaction."
        )

    def finish_tx(self, tx_id: str, commit: bool) -> ConnectionEntry:
        """Commit (True) or roll back (False) the transaction; releases the
        dedicated handle. Returns the entry."""
        entry = self.locate_tx(tx_id)
        verb = "commit" if commit else "rollback"
        with entry.tx_lock:
            if entry.tx_id != tx_id:
                raise LookupError(f"Transaction '{tx_id[:8]}…' already finished.")
            try:
                if commit:
                    DIALECTS[entry.dialect].commit_tx(entry.tx_handle)
                else:
                    DIALECTS[entry.dialect].rollback_tx(entry.tx_handle)
                DIALECTS[entry.dialect].close_tx(entry.handle, entry.tx_handle)
            finally:
                entry.tx_id = None
                entry.tx_handle = None
                entry.tx_opened_at = None
                entry.tx_last_used = None
            if commit:
                entry.schema_cache.clear()  # DDL staleness guard, as in execute_sql
            metrics["transactions_committed" if commit else "transactions_rolled_back"] += 1
            logger.info(f"[databasemcp] tx {tx_id[:8]}… {verb} on '{entry.name}'")
            return entry

    def _abort_tx(self, entry: ConnectionEntry, reason: str = "aborted") -> None:
        """Roll back + release an entry's tx without ownership checks
        (reaper/reset path — caller holds tx_lock)."""
        tx_id = entry.tx_id
        try:
            if entry.tx_handle is not None:
                try:
                    DIALECTS[entry.dialect].rollback_tx(entry.tx_handle)
                except Exception as rb_err:
                    logger.warning(f"tx {tx_id[:8]}… rollback error: {rb_err}")
                try:
                    DIALECTS[entry.dialect].close_tx(entry.handle, entry.tx_handle)
                except Exception as cl_err:
                    logger.warning(f"tx {tx_id[:8]}… close error: {cl_err}")
        finally:
            entry.tx_id = None
            entry.tx_handle = None
            entry.tx_opened_at = None
            entry.tx_last_used = None
            metrics["transactions_reaped"] += 1
            logger.info(f"[databasemcp] tx {tx_id[:8]}… {reason} on '{entry.name}'")

    def reap_idle_txs(self, idle_timeout: float) -> int:
        """Roll back transactions idle longer than idle_timeout (monotonic
        seconds). Non-blocking per entry — a tx with an in-flight statement
        is skipped and reaped on a later sweep. Returns the count reaped."""
        reaped = 0
        with self._map_lock:
            candidates = [e for e in self._entries.values() if e.tx_id]
        for entry in candidates:
            if not entry.tx_lock.acquire(blocking=False):
                continue  # statement in flight — next sweep
            try:
                if entry.tx_id is None:
                    continue  # finished while we waited
                now = _time.monotonic()
                idle = now - (entry.tx_last_used or entry.tx_opened_at or now)
                if idle < idle_timeout:
                    continue
                self._abort_tx(entry, reason=f"reaped (idle {int(idle)}s)")
                reaped += 1
            finally:
                entry.tx_lock.release()
        return reaped


REGISTRY = ConnectionRegistry()
