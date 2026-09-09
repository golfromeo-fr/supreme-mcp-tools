"""
databasemcp MCP tools — all @mcp.tool() registrations + FEF setup_extensions.

Ported verbatim (behavior-preserving) from the former oraclemcp monolith;
P2/P3 add the connection-registry tools and per-dialect dispatch.
"""
import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Any

import oracledb
import openai

from core import (
    mcp, logger, metrics, TOOL_NAME,
    FEF_V3_AVAILABLE, ToolExtensionManager, register_common_extensions,
    setup_tool_extensions, Extension, ExtensionType,
)
import rules
import connections
from connections import REGISTRY, _SECRET_KEYS
from dialects import DIALECTS, get_dialect
import presets


# ============================================================================
# Connection Registry Tools (P2)
# ============================================================================

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


@mcp.tool()
async def connect_database(name: str, db_type: str, params: dict) -> str:
    """Connect to a database and register it under a name.

    db_type: oracle | postgres | libsql. params per type:
    - oracle: user, password, host, port, service_name
    - postgres: host, dbname, user, password (optional: port, connect_timeout)
    - libsql: url ("file:/path.db", "file::memory:", or "libsql://..." with
      optional auth_token). NOTE: a file: URL to a nonexistent path creates
      an empty database (SQLite semantics).
    The first connection becomes the active one for query/execute_sql/etc.
    Params are never echoed back.
    """
    start_time = time.perf_counter()
    if not _NAME_RE.match(name or ""):
        _timing_update(start_time, "connect_database", False)
        return "Invalid connection name: use 1-32 chars of letters, digits, '_' or '-'"
    try:
        dialect = get_dialect(db_type)
    except ValueError as e:
        _timing_update(start_time, "connect_database", False)
        return f"Error: {e}"
    missing = [p for p in dialect.REQUIRED_PARAMS if not params.get(p)]
    if missing:
        _timing_update(start_time, "connect_database", False)
        return f"Missing required params for {db_type}: {', '.join(missing)}"

    def _connect():
        REGISTRY.connect(name, db_type, params)

    try:
        await asyncio.to_thread(_connect)
    except Exception as e:
        _timing_update(start_time, "connect_database", False)
        logger.error(f"Connect failed for '{name}' ({db_type}): {type(e).__name__}")
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        for k, v in (params or {}).items():  # never echo secret values
            if str(k).lower() in _SECRET_KEYS and v:
                msg = msg.replace(str(v), "***")
        return f"Connect failed: {msg}"
    _timing_update(start_time, "connect_database", True)
    active = REGISTRY._active
    return f"Connected '{name}' ({db_type}). Active: '{active}'"


@mcp.tool()
async def list_presets() -> str:
    """Lists the connection presets defined in .env (DB_PRESET_<NN>), with
    dialect, description, autoconnect flag — passwords never shown. A preset
    can be used directly: pass its number or NAME as `connection` in
    query/execute_sql/etc. (connects lazily), or via connect_preset."""
    start_time = time.perf_counter()
    found = presets.load_presets()
    if not found:
        _timing_update(start_time, "list_presets", True)
        return (
            "none — add DB_PRESET_<NN>=oracle://user:pass@host:port/service "
            "(or postgres://..., or file:/path.db) to .env and restart"
        )
    lines = []
    for p_ in found:
        flags = []
        if p_.autoconnect:
            flags.append("autoconnect")
        label = f" — {p_.desc}" if p_.desc else ""
        alias = f" (name: {p_.name})" if p_.name else ""
        lines.append(
            f"- {p_.number}{alias} [{p_.dialect}]{label} {p_.url_masked}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )
    _timing_update(start_time, "list_presets", True)
    return "\n".join(lines)


@mcp.tool()
async def connect_preset(preset: str) -> str:
    """Connects using a preset from .env — by number ("01") or NAME alias
    ("pglocal") — and registers the connection (first one becomes active).
    Equivalent to connect_database with the preset's params."""
    start_time = time.perf_counter()
    try:
        p_ = presets.get_preset(preset)
    except LookupError as e:
        _timing_update(start_time, "connect_preset", False)
        return f"Error: {e}"
    # E3.5: preset grants are per user (admin bypasses)
    from connections import preset_grant_error, _current_caller_grants
    err = preset_grant_error(p_.number, p_.name, p_.connection_name,
                             *_current_caller_grants())
    if err:
        _timing_update(start_time, "connect_preset", False)
        return f"Error: {err}"

    def _connect():
        REGISTRY.connect(p_.connection_name, p_.dialect, p_.params)

    try:
        await asyncio.to_thread(_connect)
    except Exception as e:
        _timing_update(start_time, "connect_preset", False)
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        for k, v in (p_.params or {}).items():
            if str(k).lower() in _SECRET_KEYS and v:
                msg = msg.replace(str(v), "***")
        return f"Connect failed: {msg}"
    _timing_update(start_time, "connect_preset", True)
    return f"Connected preset {p_.number} as '{p_.connection_name}' ({p_.dialect}). Active: '{REGISTRY._active}'"


@mcp.tool()
async def disconnect_database(name: str) -> str:
    """Disconnect and remove a named database connection (refused while an
    interactive transaction is open on it)."""
    start_time = time.perf_counter()
    try:
        result = await asyncio.to_thread(REGISTRY.disconnect, name)
    except LookupError as e:
        _timing_update(start_time, "disconnect_database", False)
        return f"Error: {e}"
    if result.startswith("Connection '") and (
        "busy" in result or "active transaction" in result
    ):
        _timing_update(start_time, "disconnect_database", False)
        return result
    _timing_update(start_time, "disconnect_database", True)
    return f"Disconnected '{name}'. Active now: {result}"


@mcp.tool()
async def list_connections() -> str:
    """List all registered database connections (never shows credentials)."""
    start_time = time.perf_counter()
    conns = REGISTRY.list()
    if not conns:
        _timing_update(start_time, "list_connections", True)
        return "none — use connect_database(name, db_type, params)"
    lines = []
    for c in conns:
        active = " *ACTIVE*" if c["active"] else ""
        last_err = f" last_error={c['last_error']}" if c["last_error"] else ""
        tx = f" tx={c['tx']}…" if c.get("tx") else ""
        lines.append(
            f"- {c['name']} ({c['dialect']}, {c['state']}, cached={c['cached_tables']}){tx}{active}{last_err}"
        )
    _timing_update(start_time, "list_connections", True)
    return "\n".join(lines)


@mcp.tool()
async def use_database(name: str) -> str:
    """Switch the active database connection used by query/execute_sql/etc."""
    start_time = time.perf_counter()
    try:
        entry = await asyncio.to_thread(REGISTRY.set_active, name)
    except LookupError as e:
        _timing_update(start_time, "use_database", False)
        return f"Error: {e}"
    _timing_update(start_time, "use_database", True)
    return f"Active connection: '{entry.name}' ({entry.dialect})"


# ============================================================================
# Interactive Transactions (E4) — begin / commit / rollback + idle reaper
# ============================================================================

DB_TX_SWEEP_INTERVAL_S = 60.0

_reaper_task: asyncio.Task | None = None
_reaper_started = False


def _ensure_reaper() -> None:
    """Start the reaper lazily on the first begin_transaction — it needs a
    running loop, and no abandoned transaction can predate this process's
    first begin (transactions die with the server process)."""
    global _reaper_task, _reaper_started
    if _reaper_started:
        return
    _reaper_started = True
    _reaper_task = start_tx_reaper()


@mcp.tool()
async def begin_transaction(connection: str | None = None) -> str:
    """Begins an interactive transaction on the active connection (or the
    named one / an unconnected preset). Returns a tx_id to pass to query or
    execute_sql; finish with commit_transaction or rollback_transaction.
    Exactly one transaction per connection; idle transactions are rolled
    back automatically after DB_TX_IDLE_TIMEOUT seconds (default 300)."""
    start_time = time.perf_counter()
    _ensure_reaper()
    # E3: attribute the transaction to the caller (F7 — identity visible
    # inside tool functions). Unknown owner (in-memory/mono-legacy) is fine.
    owner = None
    try:
        from fastmcp.server.dependencies import get_http_request
        from tools.shared.identity import resolve_identity

        auth_verifier = getattr(mcp, "auth", None)
        if auth_verifier is not None and hasattr(auth_verifier, "tokens"):
            ident = resolve_identity(
                get_http_request(), lambda: auth_verifier.tokens()
            )
            owner = ident[0] if ident else None
    except Exception:
        owner = None
    try:
        entry, tx_id = await asyncio.to_thread(REGISTRY.begin_tx, connection, owner)
    except Exception as e:
        _timing_update(start_time, "begin_transaction", False)
        return _render_error(e)
    _timing_update(start_time, "begin_transaction", True)
    idle = os.environ.get("DB_TX_IDLE_TIMEOUT", "300")
    return (
        f"Transaction {tx_id} opened on '{entry.name}' ({entry.dialect}). "
        f"Pass tx_id to query/execute_sql for statements inside it, then "
        f"commit_transaction or rollback_transaction with the same tx_id. "
        f"Idle transactions roll back automatically after {idle}s."
    )


@mcp.tool()
async def commit_transaction(tx_id: str) -> str:
    """Commits the open transaction (all its statements become permanent)
    and releases its dedicated connection."""
    start_time = time.perf_counter()
    try:
        entry = await asyncio.to_thread(REGISTRY.finish_tx, tx_id, True)
    except Exception as e:
        _timing_update(start_time, "commit_transaction", False)
        return _render_error(e)
    _timing_update(start_time, "commit_transaction", True)
    return f"Transaction committed on '{entry.name}' — all its statements are now permanent."


@mcp.tool()
async def rollback_transaction(tx_id: str) -> str:
    """Rolls back the open transaction (all its statements are discarded)
    and releases its dedicated connection."""
    start_time = time.perf_counter()
    try:
        entry = await asyncio.to_thread(REGISTRY.finish_tx, tx_id, False)
    except Exception as e:
        _timing_update(start_time, "rollback_transaction", False)
        return _render_error(e)
    _timing_update(start_time, "rollback_transaction", True)
    return f"Transaction rolled back on '{entry.name}' — none of its statements were applied."


async def _tx_reaper_loop(sweep_interval: float, idle_timeout: float) -> None:
    """Sweep for abandoned transactions; resilient loop (a failed sweep is
    logged and the next tick continues)."""
    while True:
        await asyncio.sleep(sweep_interval)
        try:
            reaped = await asyncio.to_thread(REGISTRY.reap_idle_txs, idle_timeout)
            if reaped:
                logger.info(f"[databasemcp] tx reaper rolled back {reaped} idle transaction(s)")
        except Exception:
            logger.exception("[databasemcp] tx reaper sweep failed — continuing")


def start_tx_reaper() -> asyncio.Task | None:
    """Start the idle-transaction reaper next to the launcher's event loop.
    DB_TX_IDLE_TIMEOUT (secs, default 300); 0 disables it (with a warning)."""
    idle_timeout = float(os.environ.get("DB_TX_IDLE_TIMEOUT", "300"))
    if idle_timeout <= 0:
        logger.warning(
            "DB_TX_IDLE_TIMEOUT=0 — idle-transaction reaper DISABLED. "
            "Abandoned transactions hold locks and a pinned connection until "
            "the DB server's own idle-in-transaction timeout reclaims them."
        )
        return None
    logger.info(
        f"[databasemcp] tx reaper active: sweep every {int(DB_TX_SWEEP_INTERVAL_S)}s, "
        f"roll back after {int(idle_timeout)}s idle"
    )
    return asyncio.create_task(_tx_reaper_loop(DB_TX_SWEEP_INTERVAL_S, idle_timeout))


# ============================================================================
# FEF V3 State + Extension Handlers
# ============================================================================

fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def _record_request(tool_name: str, success: bool, elapsed_ms: float) -> None:
    """FEF per-request metrics (factored from the monolith's repeated blocks)."""
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name=tool_name,
            success=success, duration_ms=elapsed_ms
        )


def _timing_update(start_time: float, tool_name: str, success: bool) -> float:
    """Shared metrics bookkeeping; returns elapsed ms."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if success:
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
    else:
        metrics["query_errors"] += 1
    _record_request(tool_name, success, elapsed_ms)
    return elapsed_ms


def get_query_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get query statistics."""
    avg_query_time = (
        metrics["total_query_time_ms"] / metrics["query_count"]
        if metrics["query_count"] > 0 else 0.0
    )
    return {
        "total_queries": metrics["query_count"],
        "query_errors": metrics["query_errors"],
        "avg_query_time_ms": round(avg_query_time, 2),
        "schema_lookups": metrics["schema_lookups"]
    }


def get_connections_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: connection registry statistics."""
    conns = REGISTRY.list()
    return {
        "active_connection": next((c["name"] for c in conns if c["active"]), None),
        "total_connections": len(conns),
        "active_transactions": sum(1 for c in conns if c.get("tx")),
        "connections": [
            {"name": c["name"], "dialect": c["dialect"], "state": c["state"],
             "cached_tables": c["cached_tables"], "last_error": c["last_error"]}
            for c in conns
        ],
    }


def get_connection_presets(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: connection presets from .env (masked)."""
    return {
        "presets": [
            {
                "number": p_.number,
                "dialect": p_.dialect,
                "name": p_.name,
                "desc": p_.desc,
                "url": p_.url_masked,
                "autoconnect": p_.autoconnect,
                "connected": p_.connection_name in REGISTRY._entries,
            }
            for p_ in presets.load_presets()
        ]
    }


def get_schema_cache_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: per-connection schema cache statistics."""
    conns = REGISTRY.list()
    return {
        "cached_tables": sum(c["cached_tables"] for c in conns),
        "schema_lookups": metrics["schema_lookups"],
        "per_connection": {c["name"]: c["cached_tables"] for c in conns},
    }


def reset_connections(params: dict[str, Any]) -> dict[str, Any]:
    """Action: close all connections (lazy reconnect on next use)."""
    closed, skipped = REGISTRY.close_all()
    logger.info(f"[databasemcp] reset_connections: closed {closed}, skipped {skipped} busy")
    return {
        "success": True,
        "message": f"Closed {closed} connection(s); skipped {skipped} busy. They reconnect lazily on next use.",
    }


def clear_cache(params: dict[str, Any]) -> dict[str, Any]:
    """Action: clear every connection's schema cache."""
    cleared = 0
    for entry in REGISTRY._entries.values():
        with entry.lock:
            entry.schema_cache.clear()
            cleared += 1
    logger.info(f"[databasemcp] Schema cache cleared on {cleared} connection(s)")
    return {
        "success": True,
        "message": f"Schema cache cleared on {cleared} connection(s)",
    }


def connect_preset_action(params: dict[str, Any]) -> dict[str, Any]:
    """Action: connect a .env preset by number/NAME alias (mgmt UI path).
    Idempotent: an already-connected preset reports success without reconnecting."""
    key = str(params.get("preset", "")).strip()
    if not key:
        return {"success": False,
                "message": "Provide 'preset': the preset number (e.g. '02') or NAME alias."}
    try:
        preset = presets.get_preset(key)
    except LookupError as e:
        return {"success": False, "message": str(e)}
    name = preset.connection_name
    if name in REGISTRY._entries:
        return {"success": True,
                "message": f"'{name}' ({preset.dialect}) is already connected."}
    try:
        REGISTRY.connect(name, preset.dialect, preset.params)
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        for k, v in (preset.params or {}).items():
            if str(k).lower() in _SECRET_KEYS and v:
                msg = msg.replace(str(v), "***")
        logger.warning(f"[databasemcp] UI preset connect {preset.number} failed: {type(e).__name__}")
        return {"success": False, "message": f"Connect failed: {msg}"}
    logger.info(f"[databasemcp] preset {preset.number} connected via mgmt action as '{name}'")
    return {"success": True,
            "message": f"Connected preset {preset.number} as '{name}' ({preset.dialect}). "
                       f"Active: '{REGISTRY._active}'"}


def disconnect_connection_action(params: dict[str, Any]) -> dict[str, Any]:
    """Action: disconnect a named connection (mgmt UI path)."""
    name = str(params.get("name", "")).strip()
    if not name:
        return {"success": False, "message": "Provide 'name': the connection to disconnect."}
    try:
        result = REGISTRY.disconnect(name)
    except LookupError as e:
        return {"success": False, "message": str(e)}
    except Exception as e:
        return {"success": False, "message": f"{type(e).__name__}: {e}"}
    if "busy" in result:
        return {"success": False, "message": result}
    logger.info(f"[databasemcp] '{name}' disconnected via mgmt action")
    return {"success": True, "message": f"Disconnected '{name}'. Active now: {result}"}


# ============================================================================
# MCP Tools (ported verbatim; error rendering identical)
# ============================================================================

def _format_db_error(error_msg: dict) -> str:
    if error_msg.get("code"):
        formatted = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
        if error_msg.get("offset"):
            formatted += f"\nAt position: {error_msg['offset']}"
    else:
        formatted = f"Error: {error_msg['message']}"
    return formatted


@mcp.tool()
async def get_schemas(table_name: str | None = None) -> str:
    """Get schema information for a specified table."""
    start_time = time.perf_counter()
    if not table_name:
        _timing_update(start_time, "get_schemas", False)
        return "Error: table_name is required"

    table_name = table_name.upper()
    schema = connections.fetch_schema_from_cache(table_name)

    if not schema or not isinstance(schema, dict):
        _timing_update(start_time, "get_schemas", False)
        return f"Invalid schema for table '{table_name}'."

    _timing_update(start_time, "get_schemas", True)
    return str({
        "columns": schema["columns"],
        "constraints": schema["constraints"]
    })


# ============================================================================
# Generic Data Tools (P3) — dialect dispatch over the connection registry
# ============================================================================

def _with_entry(connection: str | None, fn):
    """Run fn(entry) under the entry's lock; returns (ok, result_or_error).

    Shared metrics timing + [SQL] logging live with the callers.
    """
    entry = REGISTRY.get(connection)
    if not entry.lock.acquire(timeout=int(os.environ.get("DB_LOCK_WAIT_S", "10"))):
        raise TimeoutError(f"connection '{entry.name}' is busy")
    entry.last_used = time.time()
    try:
        return fn(entry)
    except Exception as e:
        entry.last_error = str(e)[:200]
        raise
    finally:
        entry.lock.release()


def _run_entry_select(entry, sql: str, max_rows: int):
    logger.info(f"[SQL] Executing query: {sql[:200]}{'...' if len(sql) > 200 else ''}")
    start = time.time()
    try:
        rows, truncated = DIALECTS[entry.dialect].run_select(entry.handle, sql, max_rows)
        metrics["query_count"] += 1
        return rows, truncated
    except Exception as e:
        metrics["query_errors"] += 1
        raise
    finally:
        elapsed_ms = (time.time() - start) * 1000
        metrics["total_query_time_ms"] += elapsed_ms


def _assert_preset_grant(connection: str | None) -> None:
    """E3.5: check preset grants ON THE EVENT LOOP (before to_thread).
    Non-admin callers may only use presets granted via db_presets.
    Runs on the event loop because get_http_request() needs the HTTP
    request context. Fail-open for mono mode / non-HTTP scope."""
    if not connection:
        return
    verifier = getattr(mcp, "auth", None)
    get_tokens = getattr(verifier, "tokens", None)
    if get_tokens is None:
        return
    try:
        from fastmcp.server.dependencies import get_http_request
        request = get_http_request()
    except Exception:
        return
    auth = (request.headers.get("authorization", "") or "")
    token = auth[7:] if auth.lower().startswith("bearer ") else \
        request.headers.get("x-api-key")
    if not token:
        return
    entry = get_tokens().get(token)
    if entry is None:
        return
    role = entry.get("role", "admin")  # legacy mono map = admin
    if role == "admin":
        return
    granted = set(entry.get("db_presets") or [])
    try:
        preset = presets.get_preset(connection)
    except LookupError:
        return
    if preset.number not in granted and preset.name not in granted \
            and preset.connection_name not in granted:
        raise PermissionError(
            f"Preset '{connection}' is not granted to your account "
            f"({entry.get('client_id', 'unknown')})."
        )


def _read_guard(sql: str) -> str | None:
    """Lexical read-only guard for query() (accident prevention, not security:
    PG CTE-DML 'WITH x AS (DELETE ...)' passes — client holds execute_sql anyway)."""
    s = sql.strip()
    while s.startswith("("):
        s = s[1:].lstrip()
    first = s.split(None, 1)[0].upper() if s else ""
    if first not in ("SELECT", "WITH"):
        return f"query() is read-only (got '{first or 'empty'}'). Use execute_sql for DML/DDL."
    return None


_ORACLE_DDL_WORDS = {"CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE",
                     "ANALYZE", "COMMENT"}


def _oracle_ddl_warning(sql: str) -> str:
    """Oracle DDL commits implicitly — warn when such a statement runs inside
    a transaction (best-effort keyword check, not a parser)."""
    first = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
    if first in _ORACLE_DDL_WORDS:
        return ("⚠️ Oracle DDL commits implicitly — this statement just ended the "
                "transaction server-side; a later commit will report it as finished. ")
    return ""


def _tx_entry(tx_id: str, connection: str | None):
    """Resolve the entry owning tx_id; a mismatched connection param is an
    error (edge case 5 — the tx pins its connection)."""
    entry = REGISTRY.locate_tx(tx_id)
    if connection and connection != entry.name:
        raise ValueError(
            f"tx belongs to connection '{entry.name}', not '{connection}'. "
            "Omit the connection parameter or use the tx's connection."
        )
    return entry


def _run_tx_select(entry, tx_id: str, sql: str, max_rows: int):
    """SELECT inside the open transaction (tx_lock, not entry lock)."""
    if not entry.tx_lock.acquire(timeout=int(os.environ.get("DB_LOCK_WAIT_S", "10"))):
        raise TimeoutError(f"transaction on '{entry.name}' is busy")
    try:
        if entry.tx_id != tx_id:
            raise LookupError(f"Transaction '{tx_id[:8]}…' already finished.")
        start = time.time()
        try:
            rows, truncated = DIALECTS[entry.dialect].select_tx(entry.tx_handle, sql, max_rows)
            metrics["query_count"] += 1
            return rows, truncated
        except Exception:
            metrics["query_errors"] += 1
            raise
        finally:
            metrics["total_query_time_ms"] += (time.time() - start) * 1000
            entry.tx_last_used = time.monotonic()
    finally:
        entry.tx_lock.release()


def _run_tx_execute(entry, tx_id: str, sql: str) -> int:
    """Statement inside the open transaction — NO commit (explicit only)."""
    if not entry.tx_lock.acquire(timeout=int(os.environ.get("DB_LOCK_WAIT_S", "10"))):
        raise TimeoutError(f"transaction on '{entry.name}' is busy")
    try:
        if entry.tx_id != tx_id:
            raise LookupError(f"Transaction '{tx_id[:8]}…' already finished.")
        start = time.time()
        try:
            rowcount = DIALECTS[entry.dialect].execute_tx(entry.tx_handle, sql)
            metrics["query_count"] += 1
            return rowcount
        except Exception:
            metrics["query_errors"] += 1
            raise
        finally:
            metrics["total_query_time_ms"] += (time.time() - start) * 1000
            entry.tx_last_used = time.monotonic()
    finally:
        entry.tx_lock.release()


def _run_batch(connection: str | None, statements: list[str]) -> str:
    """Tier 1 atomic batch: all statements on ONE dedicated connection inside
    one transaction; commit-or-rollback as a unit. Caller holds nothing —
    the entry lock covers the whole call."""
    entry = REGISTRY.get(connection)
    if not entry.lock.acquire(timeout=int(os.environ.get("DB_LOCK_WAIT_S", "10"))):
        raise TimeoutError(f"connection '{entry.name}' is busy")
    entry.last_used = time.time()
    try:
        dialect = DIALECTS[entry.dialect]
        tx = dialect.open_tx(entry.handle, entry.params)
        rowcounts: list[int] = []
        try:
            for i, stmt in enumerate(statements):
                if not stmt or not stmt.strip():
                    raise ValueError(f"statement {i} is empty")
                logger.info(f"[SQL] Batch {i + 1}/{len(statements)}: {stmt[:120]}")
                rowcounts.append(dialect.execute_tx(tx, stmt))
            dialect.commit_tx(tx)
        except Exception as e:
            try:
                dialect.rollback_tx(tx)
            except Exception as rb_err:
                logger.warning(f"batch rollback error: {rb_err}")
            finally:
                dialect.close_tx(entry.handle, tx)
            metrics["query_errors"] += 1
            failed_at = len(rowcounts)
            return (
                f"Error: batch ROLLED BACK at statement {failed_at} "
                f"(statements 0..{failed_at - 1} rowcounts {rowcounts} were undone). "
                f"Cause: {str(e).splitlines()[0] if str(e) else type(e).__name__}"
            )
        dialect.close_tx(entry.handle, tx)
        entry.schema_cache.clear()  # batches may contain DDL
        metrics["query_count"] += len(statements)
        logger.info(f"[SQL] Batch committed: {len(statements)} statement(s)")
        return f"OK. Batch committed: {len(statements)} statement(s), rowcounts {rowcounts}"
    finally:
        entry.lock.release()


@mcp.tool()
async def get_valid_languages() -> str:
    """Get valid language codes from the LANGUES table (Oracle work DB only)."""
    start_time = time.perf_counter()
    try:
        def _run(entry):
            if entry.dialect != "oracle":
                raise RuntimeError(
                    f"Oracle-only tool (queries the LANGUES table); active connection is {entry.dialect}."
                )
            sql = """
                SELECT LANCODE, LANLIBC, LANLIBL, LANUSED
                FROM LANGUES
                WHERE ROWNUM <= 10
                ORDER BY LANCODE
            """
            return _run_entry_select(entry, sql, max_rows=10)

        rows, _trunc = await asyncio.to_thread(_with_entry, None, _run)
        _timing_update(start_time, "get_valid_languages", True)
        return str(rows)
    except Exception as e:
        _timing_update(start_time, "get_valid_languages", False)
        err = DIALECTS.get("oracle").format_error(e) if "oracle" in DIALECTS else None
        if err and err.get("code"):
            return _format_db_error(err)
        return f"Error: {str(e).splitlines()[0] if str(e) else type(e).__name__}"


@mcp.tool()
async def query(sql: str, max_rows: int = 100, connection: str | None = None,
                tx_id: str | None = None) -> str:
    """Executes a read-only SQL query (SELECT/WITH) on the active connection
    (or the named one) and returns up to max_rows rows as JSON-ish text.
    Pass tx_id to read inside an open transaction (see begin_transaction)."""
    start_time = time.perf_counter()
    if not sql or not sql.strip():
        _timing_update(start_time, "query", False)
        return "Error: sql query is required"
    max_rows = max(1, min(int(max_rows), 5000))
    guard = _read_guard(sql)
    if guard:
        _timing_update(start_time, "query", False)
        return guard
    try:
        if tx_id:
            def _run_tx(entry):
                return _run_tx_select(entry, tx_id, sql, max_rows)
            tx_entry = await asyncio.to_thread(_tx_entry, tx_id, connection)
            rows, truncated = await asyncio.to_thread(_run_tx, tx_entry)
        else:
            _assert_preset_grant(connection)

            def _run(entry):
                return _run_entry_select(entry, sql, max_rows)

            rows, truncated = await asyncio.to_thread(_with_entry, connection, _run)
    except Exception as e:
        _timing_update(start_time, "query", False)
        return _render_error(e)
    _timing_update(start_time, "query", True)
    out = str(rows)
    if truncated:
        out += f"\n\n(Truncated — showing {max_rows} rows)"
    return out


@mcp.tool()
async def execute_sql(sql: str = "", connection: str | None = None,
                      statements: list[str] | None = None,
                      tx_id: str | None = None) -> str:
    """Executes an SQL statement (INSERT/UPDATE/DELETE/DDL) on the active
    connection (or the named one); commits and reports affected rows.

    Atomic batch: pass statements=[...] (mutually exclusive with sql, max 50)
    to run several statements in ONE all-or-nothing transaction — any failure
    rolls the whole batch back and reports the failing index.
    Pass tx_id to execute inside an open transaction WITHOUT committing
    (see begin_transaction)."""
    start_time = time.perf_counter()
    if statements is not None and sql.strip():
        _timing_update(start_time, "execute_sql", False)
        return "Error: pass either sql or statements, not both"
    if statements is not None:
        if not statements:
            _timing_update(start_time, "execute_sql", False)
            return "Error: statements must contain at least one statement"
        if len(statements) > 50:
            _timing_update(start_time, "execute_sql", False)
            return "Error: statements supports at most 50 per batch"
        try:
            note = await asyncio.to_thread(_run_batch, connection, statements)
        except Exception as e:
            _timing_update(start_time, "execute_sql", False)
            return _render_error(e)
        _timing_update(start_time, "execute_sql", True)
        return note
    if not sql or not sql.strip():
        _timing_update(start_time, "execute_sql", False)
        return "Error: sql statement is required"
    _assert_preset_grant(connection)
    logger.info(f"[SQL] Executing statement: {sql[:200]}{'...' if len(sql) > 200 else ''}")
    try:
        if tx_id:
            entry0 = REGISTRY.locate_tx(tx_id)
            rowcount = await asyncio.to_thread(_run_tx_execute, entry0, tx_id, sql)
            _timing_update(start_time, "execute_sql", True)
            warn = _oracle_ddl_warning(sql) if entry0.dialect == "oracle" else ""
            return f"{warn}OK (in transaction, not committed). Rows affected: {rowcount}"

        def _run(entry):
            start = time.time()
            try:
                rowcount = DIALECTS[entry.dialect].execute(entry.handle, sql)
                entry.schema_cache.clear()  # DDL staleness guard
                metrics["query_count"] += 1
                return rowcount
            except Exception:
                metrics["query_errors"] += 1
                raise
            finally:
                elapsed_ms = (time.time() - start) * 1000
                metrics["total_query_time_ms"] += elapsed_ms

        _assert_preset_grant(connection)
        rowcount = await asyncio.to_thread(_with_entry, connection, _run)
    except Exception as e:
        _timing_update(start_time, "execute_sql", False)
        return _render_error(e)
    _timing_update(start_time, "execute_sql", True)
    return f"OK. Rows affected: {rowcount}"


@mcp.tool()
async def get_schemas(table_name: str, connection: str | None = None) -> str:
    """Returns columns, constraints and foreign keys for a table on the
    active connection (or the named one)."""
    start_time = time.perf_counter()
    if not table_name or not table_name.strip():
        _timing_update(start_time, "get_schemas", False)
        return "Error: table_name is required"
    try:
        def _run(entry):
            name = table_name.strip()
            if entry.dialect == "oracle":
                name = name.upper()
            if name in entry.schema_cache:
                return entry.schema_cache[name], True
            desc = DIALECTS[entry.dialect].describe_table(entry.handle, name)
            metrics["schema_lookups"] += 1
            entry.schema_cache[name] = desc
            return desc, False

        desc, cached = await asyncio.to_thread(_with_entry, connection, _run)
    except Exception as e:
        _timing_update(start_time, "get_schemas", False)
        return _render_error(e)
    if not desc["columns"]:
        entry_name = connection if connection else (REGISTRY._active or "?")
        _timing_update(start_time, "get_schemas", False)
        return f"Table '{table_name.strip()}' not found on connection '{entry_name}'."
    _timing_update(start_time, "get_schemas", True)
    label = " (cached)" if cached else ""
    return str({"table": table_name.strip() + label, **desc})


@mcp.tool()
async def list_tables(connection: str | None = None) -> str:
    """Lists all tables on the active connection (or the named one) with their comments."""
    start_time = time.perf_counter()
    try:
        def _run(entry):
            start = time.time()
            try:
                tables = DIALECTS[entry.dialect].list_tables(entry.handle)
                metrics["query_count"] += 1
                return tables
            except Exception:
                metrics["query_errors"] += 1
                raise
            finally:
                elapsed_ms = (time.time() - start) * 1000
                metrics["total_query_time_ms"] += elapsed_ms

        tables = await asyncio.to_thread(_with_entry, connection, _run)
    except Exception as e:
        _timing_update(start_time, "list_tables", False)
        return _render_error(e)
    _timing_update(start_time, "list_tables", True)
    return "\n".join(f"{t['name']} — {t['comment']}" for t in tables) if tables else "No tables found."


@mcp.tool()
async def explain_plan(sql: str, connection: str | None = None) -> str:
    """Returns the execution plan for a SQL statement (dialect-specific)."""
    start_time = time.perf_counter()
    if not sql or not sql.strip():
        _timing_update(start_time, "explain_plan", False)
        return "Error: sql query is required"
    try:
        def _run(entry):
            start = time.time()
            try:
                plan = DIALECTS[entry.dialect].explain(entry.handle, sql)
                metrics["query_count"] += 1
                return plan
            except Exception:
                metrics["query_errors"] += 1
                raise
            finally:
                elapsed_ms = (time.time() - start) * 1000
                metrics["total_query_time_ms"] += elapsed_ms

        plan = await asyncio.to_thread(_with_entry, connection, _run)
    except Exception as e:
        _timing_update(start_time, "explain_plan", False)
        return _render_error(e)
    _timing_update(start_time, "explain_plan", True)
    return plan


def _render_error(e: Exception) -> str:
    """Render a dialect error like the legacy tool strings; strip internals."""
    dialect = None
    try:
        active_entry = REGISTRY.get(None)
        dialect = DIALECTS[active_entry.dialect]
    except Exception:
        dialect = None
    if dialect is not None:
        details = dialect.format_error(e)
        if details.get("code"):
            out = f"Oracle Error {details['code']}: {details['message']}" if details["code"].startswith("ORA") \
                else f"Database Error {details['code']}: {details['message']}"
            if details.get("offset"):
                out += f"\nAt position: {details['offset']}"
            return out
        return f"Error: {details['message']}"
    return f"Error: {str(e).splitlines()[0] if str(e) else type(e).__name__}"


@mcp.tool()
async def get_sql_optimization_rules() -> str:
    """Returns the SQL optimization rules from the user-local optimization.json."""
    start_time = time.perf_counter()
    try:
        text = rules.load("optimization.json")
        _timing_update(start_time, "get_sql_optimization_rules", True)
        return text
    except FileNotFoundError as e:
        _timing_update(start_time, "get_sql_optimization_rules", False)
        return f"Error: {e}"


@mcp.tool()
async def optimize_sql_with_ai(sql: str) -> str:
    """Accepts a SQL query, references the user-local optimization rules, and
    calls an AI (via AI_BASE_URL + AI_API_KEY) to suggest optimizations."""
    start_time = time.perf_counter()
    if not sql:
        _timing_update(start_time, "optimize_sql_with_ai", False)
        return "Error: sql query is required"

    api_key = os.getenv('AI_API_KEY')
    base_url = os.getenv('AI_BASE_URL')
    if not api_key or api_key == "put_your_api_key_here":
        _timing_update(start_time, "optimize_sql_with_ai", False)
        logger.error("AI_API_KEY environment variable not properly configured")
        return "Error: The AI optimization feature requires a valid API key. Please update the AI_API_KEY environment variable."
    if not base_url:
        _timing_update(start_time, "optimize_sql_with_ai", False)
        return "Error: The AI optimization feature requires AI_BASE_URL (your OpenAI-compatible gateway URL) in the environment."

    try:
        def extract_table_names_from_sql(sql_query):
            pattern = r"(?:from|join|into|update|with)\s+([a-zA-Z0-9_]+)"
            return list(set(re.findall(pattern, sql_query, re.IGNORECASE)))

        # Live schema context via the registry (cached describe; any dialect)
        table_descriptions = None
        try:
            entry = REGISTRY.get(None)
            schemas_result = {}
            for table_name in extract_table_names_from_sql(sql):
                name = table_name.upper() if entry.dialect == "oracle" else table_name
                if name not in entry.schema_cache:
                    entry.schema_cache[name] = DIALECTS[entry.dialect].describe_table(entry.handle, name)
                desc = entry.schema_cache[name]
                if desc.get("columns"):
                    schemas_result[table_name] = desc
            if schemas_result:
                table_descriptions = str(schemas_result)
        except Exception as e:
            logger.error(f"Error fetching table schemas for AI optimization: {e}")

        rules_text = rules.load("optimization.json")

        prompt = "You are an expert SQL query optimizer."
        if table_descriptions:
            prompt += "\n\nHere are the relevant table structures and comments for context:\n"
            prompt += f"{table_descriptions}\n"
        prompt += (
            "\nGiven the following SQL query and a set of optimization rules, "
            "suggest improvements or rewrite the query to be as efficient as possible.\n\n"
            "Optimization Rules:\n"
            f"{rules_text}\n\n"
            "SQL Query:\n"
            f"{sql}\n\n"
            "Optimized SQL and/or suggestions (include comments explaining optimizations):"
        )

        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        def _call():
            return client.chat.completions.create(
                model=os.getenv("AI_MODEL", "gpt-4.1"),
                messages=[{"role": "user", "content": prompt}],
            )

        response = await asyncio.to_thread(_call)
        _timing_update(start_time, "optimize_sql_with_ai", True)
        return response.choices[0].message.content
    except Exception as e:
        _timing_update(start_time, "optimize_sql_with_ai", False)
        logger.error(f"Error optimizing SQL with AI: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_proc_rules() -> str:
    """Returns the Pro*C coding rules from the user-local proc_rules.md."""
    start_time = time.perf_counter()
    try:
        text = rules.load("proc_rules.md")
        _timing_update(start_time, "get_proc_rules", True)
        return text
    except FileNotFoundError as e:
        _timing_update(start_time, "get_proc_rules", False)
        return f"Error: {e}"



# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================

def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", "8100"))

    custom_extensions = [
        Extension(
            name="query_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "total_queries": {"type": "integer"},
                        "query_errors": {"type": "integer"},
                        "avg_query_time_ms": {"type": "number"},
                        "schema_lookups": {"type": "integer"}
                    }
                }
            },
            handler=get_query_stats,
            metadata={"description": "Database query execution statistics", "category": "metrics"}
        ),
        Extension(
            name="connection_presets",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {"presets": {"type": "array"}}
                }
            },
            handler=get_connection_presets,
            metadata={"description": "Connection presets defined in .env (masked)", "category": "config"}
        ),
        Extension(
            name="connections",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "active_connections": {"type": "integer"},
                        "connection_errors": {"type": "integer"},
                        "config": {"type": "object"}
                    }
                }
            },
            handler=get_connections_stats,
            metadata={"description": "Connection registry statistics", "category": "metrics"}
        ),
        Extension(
            name="schema_cache",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "cached_tables": {"type": "integer"},
                        "schema_lookups": {"type": "integer"}
                    }
                }
            },
            handler=get_schema_cache_stats,
            metadata={"description": "Per-connection schema cache statistics", "category": "metrics"}
        ),
        Extension(
            name="reset_connections",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=reset_connections,
            metadata={"description": "Close all connections (lazy reconnect on next use)", "category": "maintenance"}
        ),
        Extension(
            name="clear_cache",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=clear_cache,
            metadata={"description": "Clear every connection's schema cache", "category": "maintenance"}
        ),
        Extension(
            name="connect_preset",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {"preset": {"type": "string"}},
                    "required": ["preset"],
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=connect_preset_action,
            metadata={"description": "Connect a .env preset by number or NAME alias", "category": "connections"}
        ),
        Extension(
            name="disconnect_connection",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=disconnect_connection_action,
            metadata={"description": "Disconnect a named connection", "category": "connections"}
        ),
    ]

    if registry is not None:
        fef_registry = registry
        fef_manager = ToolExtensionManager(TOOL_NAME)
        # Custom extensions FIRST so the real handlers win name collisions with
        # the common set (the generic common clear_cache must not shadow this
        # tool's registry-backed one). Each registration is independent: one
        # duplicate must never abort the rest — that is exactly how the
        # connect_preset/disconnect_connection actions went missing on
        # 2026-09-07 (custom clear_cache collided after the common batch had
        # claimed the name, aborting the loop before them).
        for ext in custom_extensions:
            try:
                fef_registry.register(TOOL_NAME, ext)
            except Exception as e:
                logger.warning(f"[{TOOL_NAME}] custom extension {ext.name} not registered: {e}")
        register_common_extensions(TOOL_NAME, fef_registry, fef_manager)
        fef_http_server = None
        logger.info(f"[{TOOL_NAME}] FEF V3 registered with launcher's registry")
    else:
        fef_manager, fef_registry, fef_http_server = setup_tool_extensions(
            tool_name=TOOL_NAME,
            mgmt_port=mgmt_port,
            custom_extensions=custom_extensions
        )
        logger.info(f"[{TOOL_NAME}] FEF V3 standalone mode on port {mgmt_port}")

    fef_setup_done = True
