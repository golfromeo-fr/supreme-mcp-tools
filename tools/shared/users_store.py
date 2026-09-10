"""E3/M2 — the user store: single source of identity for the deployment.

users.json (path: MCP_USERS_STORE or ~/.config/supreme-mcp-tools/users.json;
local-only, same trust domain as the tool config.json keys). Import is I/O-
free and side-effect-free: everything reads the file lazily through an
mtime-cached loader, so a store edit reaches every process within one
request without restarts.

Schema (v1):
{
  "version": 1,
  "users": {
    "<username>": {
      "username": str,
      "password_hash": "pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>",
      "role": "admin" | "user",
      "mcp_key": str,                      # bearer credential (plaintext,
                                           # same threat model as config keys)
      "servers": [str],                    # reachable MCP servers
      "masked_functions": {server: [fn]},  # per-user deny-list
      "enabled": bool,
      "created_at": iso8601,
      "key_rotated_at": iso8601
    }
  }
}

Multi-user activates ONLY when MCP_AUTH_MODE=multi (factory handles the
mode); the store functions themselves are mode-agnostic.

Backend selection (E3 multi-host prep): MCP_USERS_BACKEND=json (default,
this file) or db - the shared SQL backend (POSTGRES_* / TURSO_DATABASE_URL),
single-document table mcp_users_store; one-time auto-import from users.json.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from tools.shared.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

USERS_PATH = Path(
    os.environ.get("MCP_USERS_STORE")
    or Path.home() / ".config" / "supreme-mcp-tools" / "users.json"
)

PBKDF2_ITERATIONS = 390_000
USERNAME_RE = re.compile(r"^[a-z0-9_-]{2,32}$")

# E3 dig (2026-09-08): identity-blind destructive/shared-state tools.
# role=user accounts get these PRE-MASKED by default (per server where the
# user has access); admins are unaffected, and the Users tab can unmask any
# of them per user. Veto-able policy — adjust the lists, not the code.
DEFAULT_USER_MASKS: dict[str, list[str]] = {
    "memorymcp": [
        "deleteMemory", "decayOrExpire", "mergeDuplicates",
        "reindexMemory", "migrateMemoryBackend", "attachProvenance",
    ],
    "ragmcp": ["clear_index", "start_indexing", "stop_indexing", "reindex"],
    "databasemcp": ["execute_sql", "connect_database", "disconnect_database"],
}

_store_cache: dict[str, Any] = {"mtime": None, "store": None}

# -- E3 db backend (MCP_USERS_BACKEND=db) ---------------------------
# Whole-document storage in the shared SQL backend: one row, one JSON doc.
# Chosen over per-user rows because the module API is document-shaped
# (load_users/save_users) and the user count is tiny; last-write-wins per
# document is the same semantics the JSON file always had. Reads are always
# fresh (no cache) so rotation/disable propagate on the next call, exactly
# like the mtime path. Both SqlStore impls expose a DB-API ``_conn`` whose
# execute() returns a cursor - the only dialect difference is the
# placeholder (libsql "?", psycopg "%s"), resolved per connection.
_db_conn_singleton: object | None = None
_db_init_done = False


def sys_path_tools() -> list:
    return sys.path

_DB_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS mcp_users_store ("
    " id INTEGER PRIMARY KEY CHECK (id = 1),"
    " data TEXT NOT NULL,"
    " updated_at TEXT NOT NULL)",
)


def _users_backend() -> str:
    return os.environ.get("MCP_USERS_BACKEND", "json").strip().lower()


def _db_conn():
    """DB-API connection for the users store, or None to stay on JSON.

    Initialises lazily and once per process: resolves the shared SqlStore
    singleton, grabs its raw connection, ensures the schema, and imports the
    JSON file's data if the table is empty (one-time adoption). Any failure
    logs loudly and permanently downgrades this process to the JSON file
    (fail-open to the pre-E3 local behavior)."""
    global _db_conn_singleton, _db_init_done
    if _db_init_done:
        return _db_conn_singleton
    _db_init_done = True
    try:
        # dialect-agnostic executor from state_docs (turso _conn OR postgres
        # pool; '?' placeholders normalized per dialect inside)
        from tools.shared.state_docs import shared_exec

        ex = shared_exec()
        if ex is None:
            logger.warning(
                "MCP_USERS_BACKEND=db but no SQL backend is configured "
                "(POSTGRES_* / TURSO_DATABASE_URL) - staying on users.json"
            )
            return None
        for stmt in _DB_SCHEMA:
            ex.execute(stmt)
        cur = ex.execute(
            "SELECT COUNT(*) FROM mcp_users_store WHERE id = ?", (1,))
        row = cur.fetchone()
        if not row or not row[0]:
            if USERS_PATH.exists():
                try:
                    data = USERS_PATH.read_text(encoding="utf-8")
                    json.loads(data)  # only import parseable files
                    ex.execute(
                        f"INSERT INTO mcp_users_store (id, data, updated_at) "
                        f"VALUES (1, ?, ?)",
                        (data, _now_iso()),
                    )
                    logger.warning(
                        "[E3] users store: imported existing users.json into "
                        "the SQL backend (one-time migration; the file is now "
                        "a backup)"
                    )
                except Exception as e:
                    logger.warning(f"[E3] users store: skipped JSON import ({e})")
        _db_conn_singleton = ex
        logger.warning("[E3] users store: SQL backend active (shared exec)")
    except Exception as e:
        logger.warning(
            f"MCP_USERS_BACKEND=db init failed ({type(e).__name__}: {e}) - "
            "staying on users.json"
        )
        _db_conn_singleton = None
    return _db_conn_singleton


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _db_load(ex) -> dict:
    cur = ex.execute(
        "SELECT data FROM mcp_users_store WHERE id = ?", (1,))
    row = cur.fetchone()
    if not row:
        return {"version": 1, "users": {}}
    store = json.loads(row[0])
    if not isinstance(store, dict) or not isinstance(store.get("users", {}), dict):
        raise ValueError("users store (db): 'users' must be an object")
    return store


def _db_save(ex, store: dict) -> None:
    ex.execute(
        f"INSERT INTO mcp_users_store (id, data, updated_at) "
        f"VALUES (1, ?, ?) "
        f"ON CONFLICT(id) DO UPDATE SET data = excluded.data, "
        f"updated_at = excluded.updated_at",
        (json.dumps(store, ensure_ascii=False), _now_iso()),
    )


# ---------------------------------------------------------------------------
# password hashing (stdlib only)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}$"
        f"{base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify; malformed stored strings fail closed."""
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iters)
        )
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def authenticate(username: str, password: str) -> dict | None:
    """username+password → user record (WITHOUT secrets) | None.

    Timing-hardened: an unknown username still runs one hash comparison
    against a dummy hash so response time does not enumerate users.
    """
    _ensure_seeded()
    users = load_users().get("users", {})
    record = users.get((username or "").lower())
    stored = record.get("password_hash") if record else _dummy_hash()
    if not record or not verify_password(password or "", stored):
        return None
    public = _public_view(record)
    return public or None


_DUMMY_HASH = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("timing-equalizer")
    return _DUMMY_HASH


def _touch(record: dict) -> None:
    """E3/M4 seam: per-record updated_at (ISO) — the future DB migration's
    last-writer-wins comparison key. Stamped by EVERY record mutation."""
    record["updated_at"] = _now_iso()

def _public_view(record: dict) -> dict:

    return {k: v for k, v in record.items()
            if k not in ("mcp_key", "password_hash")}


# ---------------------------------------------------------------------------
# file I/O — tolerant, mtime-cached, atomic writes
# ---------------------------------------------------------------------------

def load_users() -> dict:
    """Full store dict; missing/corrupt source → empty/last-good + loud log."""
    if _users_backend() == "db":
        conn = _db_conn()
        if conn is not None:
            try:
                return _db_load(conn)
            except Exception as e:
                logger.warning(
                    f"user store (db) unreadable ({e}) - continuing with "
                    "last-good in-memory copy" if _store_cache["store"] is not None
                    else f"user store (db) unreadable ({e}) - no users loaded"
                )
                return _store_cache["store"] if _store_cache["store"] is not None \
                    else {"version": 1, "users": {}}
        # db requested but unavailable at init → JSON fallback (logged there)
    try:
        mtime = USERS_PATH.stat().st_mtime_ns if USERS_PATH.exists() else None
    except OSError:
        mtime = None
    if _store_cache["store"] is not None and _store_cache["mtime"] == mtime:
        return _store_cache["store"]
    try:
        store = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        if not isinstance(store, dict) or not isinstance(
            store.get("users", {}), dict
        ):
            raise ValueError("users.json: 'users' must be an object")
    except FileNotFoundError:
        store = {"version": 1, "users": {}}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"user store unreadable ({USERS_PATH}: {e}) — continuing with "
            "last-good in-memory copy" if _store_cache["store"] is not None
            else f"user store unreadable ({USERS_PATH}: {e}) — no users loaded"
        )
        store = _store_cache["store"] if _store_cache["store"] is not None \
            else {"version": 1, "users": {}}
    _store_cache["mtime"] = mtime
    _store_cache["store"] = store
    return store


def save_users(store: dict) -> None:
    if _users_backend() == "db":
        conn = _db_conn()
        if conn is not None:
            _db_save(conn, store)
            _store_cache["store"] = store
            _store_cache["mtime"] = None
            return
        logger.warning("save_users: db backend unavailable - writing users.json")
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(USERS_PATH, store)
    _store_cache["mtime"] = USERS_PATH.stat().st_mtime_ns
    _store_cache["store"] = store


# ---------------------------------------------------------------------------
# CRUD (each mutates + saves; returns the affected public view or raises)
# ---------------------------------------------------------------------------

def list_users() -> list[dict]:
    """Public views only — mcp_key and password_hash NEVER leave the store."""
    users = load_users().get("users", {})
    return [_public_view(u) for u in users.values()]


def get_user_record(username: str) -> dict | None:
    return load_users().get("users", {}).get((username or "").lower())


def _default_masks_for(role: str, servers: list[str]) -> dict:
    if role == "admin":
        return {}
    return {s: list(DEFAULT_USER_MASKS.get(s, []))
            for s in servers if s in DEFAULT_USER_MASKS}


def create_user(username: str, password: str, role: str = "user",
                servers: list[str] | None = None,
                masked_functions: dict | None = None,
                db_presets: list[str] | None = None,
                rag_collections: list[str] | None = None) -> dict:
    """Create a user; returns {"username", "mcp_key"} — mcp_key shown ONCE.
    db_presets/rag_collections = E3.5 data-plane grants."""
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError(
            "username must match [a-z0-9_-]{2,32} (lowercase)"
        )
    if role not in ("admin", "user"):
        raise ValueError("role must be 'admin' or 'user'")
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    store = load_users()
    users = store.setdefault("users", {})
    if username in users:
        raise ValueError(f"user '{username}' already exists")
    if username == "system" or username in _tool_names():
        raise ValueError(f"'{username}' is a reserved name")
    mcp_key = _unique_key(store)
    users[username] = {
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "mcp_key": mcp_key,
        "servers": sorted(set(servers or [])),
        "masked_functions": (dict(masked_functions) if masked_functions is not None
                             else _default_masks_for(role, sorted(set(servers or [])))),
        "db_presets": list(db_presets or []),
        "rag_collections": list(rag_collections or []),
        "enabled": True,
        "created_at": _now_iso(),
        "key_rotated_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    save_users(store)
    logger.info(f"user '{username}' created (role={role})")
    return {"username": username, "mcp_key": mcp_key}


def delete_user(username: str) -> None:
    username = (username or "").lower()
    store = load_users()
    users = store.setdefault("users", {})
    if username not in users:
        raise ValueError(f"user '{username}' does not exist")
    _refuse_last_admin(store, username)
    del users[username]
    save_users(store)
    logger.info(f"user '{username}' deleted")


def set_enabled(username: str, enabled: bool) -> None:
    record = _require(get_user_record(username), username)
    store = load_users()
    record = store["users"][record["username"]]
    if record["role"] == "admin" and not enabled:
        _refuse_last_admin(store, username)
    record["enabled"] = bool(enabled)
    _touch(record)
    save_users(store)


def set_password(username: str, password: str) -> None:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["password_hash"] = hash_password(password)
    _touch(record)
    save_users(store)


def rotate_key(username: str) -> dict:
    """New globally-unique mcp_key; old key invalid immediately. Shown ONCE."""
    username = (username or "").lower()
    store = load_users()
    record = store["users"].get(username)
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["mcp_key"] = _unique_key(store)
    record["key_rotated_at"] = _now_iso()
    _touch(record)
    save_users(store)
    return {"username": username, "mcp_key": record["mcp_key"]}


def set_servers(username: str, servers: list) -> None:
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["servers"] = sorted(set(servers or []))
    _touch(record)
    save_users(store)


def set_masked_functions(username: str, masks: dict) -> None:
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    clean = {str(k): sorted({str(x) for x in v}) for k, v in (masks or {}).items()}
    record["masked_functions"] = clean
    _touch(record)
    save_users(store)


def set_db_presets(username: str, presets: list) -> None:
    """E3.5: which databasemcp presets/connections this user may use."""
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["db_presets"] = sorted({str(x) for x in (presets or [])})
    _touch(record)
    save_users(store)


def set_rag_collections(username: str, collections: list) -> None:
    """E3.5: which ragmcp collections this user may use."""
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["rag_collections"] = sorted({str(x) for x in (collections or [])})
    _touch(record)
    save_users(store)


def get_data_grants(username: str) -> dict:
    record = get_user_record(username)
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    return {
        "db_presets": record.get("db_presets", []),
        "rag_collections": record.get("rag_collections", []),
    }


def revoke_system_key(tool_name: str) -> None:
    """OPTIONAL: retire one tool's system key (its config.json key stops
    authenticating; only user keys remain). Off by default — admin action."""
    store = load_users()
    revoked = store.setdefault("revoked_system_keys", [])
    tool = (tool_name or "").strip().lower()
    if tool not in revoked:
        revoked.append(tool)
        save_users(store)
        logger.warning(f"[E3] system key for '{tool}' REVOKED — user keys only")


# ---------------------------------------------------------------------------
# tool-server side: token map for one tool's verifier/gate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# .env declarative seeding (round 5): MCP_USER_<NAME>_PASSWORD [+ _KEY, _ROLE,
# _SERVERS] — same philosophy as DB_PRESET_<NN>. Idempotent, once per process.
# ---------------------------------------------------------------------------

_seed_ran = False


def seed_from_env() -> int:
    """Seed users from MCP_USER_<NAME>_* env lines + the admin pair.

    The admin (MCP_UI_USERNAME / MCP_UI_PASSWORD, default admin/admin) is
    seeded as role=admin with reach to every known tool — this IS the
    original admin/password pair, unchanged. Declarative users:
    MCP_USER_<NAME>_PASSWORD plus optional _KEY (authoritative when given),
    _ROLE (user|admin), _SERVERS (comma list). Returns created/synced count.
    """
    created_or_synced = 0
    store = load_users()
    users = store.setdefault("users", {})
    changed = False

    admin_name = (os.environ.get("MCP_UI_USERNAME") or "admin").strip().lower()
    admin_password = os.environ.get("MCP_UI_PASSWORD") or "admin"
    if admin_name not in users:
        users[admin_name] = {
            "username": admin_name,
            "password_hash": hash_password(admin_password),
            "role": "admin",
            "mcp_key": _unique_key(store),
            "servers": sorted(_tool_names()),
            "masked_functions": {},
            "enabled": True,
            "created_at": _now_iso(),
            "key_rotated_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        changed = True
        created_or_synced += 1
        logger.info(f"[E3] seeded admin '{admin_name}' from env pair (all tools)")

    for env_key, value in sorted(os.environ.items()):
        m = re.match(r"^MCP_USER_([A-Z0-9_]+)_PASSWORD$", env_key)
        if not m or not value:
            continue
        name = m.group(1).lower()
        if not USERNAME_RE.match(name) or name == "system" or name in _tool_names():
            logger.warning(f"[E3] seed skipped for reserved/invalid name '{name}'")
            continue
        prefix = f"MCP_USER_{m.group(1)}_"
        role = os.environ.get(prefix + "ROLE", "user")
        servers = [s.strip() for s in os.environ.get(prefix + "SERVERS", "").split(",") if s.strip()]
        record = users.get(name)
        if record is None:
            key = os.environ.get(prefix + "KEY") or _unique_key(store)
            users[name] = {
                "username": name,
                "password_hash": hash_password(value),
                "role": role if role in ("admin", "user") else "user",
                "mcp_key": key,
                "servers": servers,
                "masked_functions": _default_masks_for(
                    role if role in ("admin", "user") else "user", servers),
                "enabled": True,
                "created_at": _now_iso(),
                "key_rotated_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            changed = True
            created_or_synced += 1
            logger.info(f"[E3] seeded user '{name}' from env ({role}, {len(servers)} server(s))")
        elif os.environ.get(prefix + "KEY") and record["mcp_key"] != os.environ[prefix + "KEY"]:
            record["mcp_key"] = os.environ[prefix + "KEY"]  # declarative key wins
            record["key_rotated_at"] = _now_iso()
            _touch(record)
            changed = True
            created_or_synced += 1
            logger.info(f"[E3] synced env key for user '{name}'")

    if changed:
        save_users(store)
    return created_or_synced


def _ensure_seeded() -> None:
    global _seed_ran
    if _seed_ran:
        return
    _seed_ran = True
    try:
        seed_from_env()
    except Exception as e:
        logger.warning(f"[E3] user seeding failed ({type(e).__name__}: {e}) — continuing")


def tokens_map_for_tool(tool_name: str, system_key: str) -> dict[str, dict]:
    """{system_key: admin entry} ∪ {user key: entry} for reachable users.

    System keys are attributed to the ADMIN USERNAME (round 2: the current
    mono user) unless revoked. Users are omitted unless enabled AND the tool
    is in their servers list. mtime-cached file read inside load_users.
    """
    _ensure_seeded()
    store = load_users()
    tool = (tool_name or "").lower()
    if tool in store.get("revoked_system_keys", []):
        tokens: dict[str, dict] = {}
    else:
        admin_name = _admin_username(store)
        tokens = {
            system_key: {
                "client_id": admin_name or "system",
                "role": "admin",
                "masked": [],
                "scopes": ["mcp"],
            }
        }
    for record in store.get("users", {}).values():
        if not record.get("enabled", False):
            continue
        if tool not in (record.get("servers") or []):
            continue
        tokens[record["mcp_key"]] = {
            "client_id": record["username"],
            "role": record.get("role", "user"),
            "masked": list((record.get("masked_functions") or {}).get(tool, [])),
            "scopes": ["mcp"],
            "db_presets": list(record.get("db_presets", [])),
            "rag_collections": list(record.get("rag_collections", [])),
        }
    return tokens


def central_tokens() -> dict[str, dict]:
    """E3 multi-admin: {admin mcp_key: {"client_id": username}} for ENABLED
    admins — per-admin credentials for the central management API (8200).
    Revocation rides the existing key lifecycle (rotate_key / set_enabled /
    delete_user); non-admin keys never appear here."""
    _ensure_seeded()
    out: dict[str, dict] = {}
    for record in load_users().get("users", {}).values():
        if record.get("role") != "admin" or not record.get("enabled", False):
            continue
        out[record["mcp_key"]] = {"client_id": record["username"], "role": "admin"}
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _admin_username(store: dict) -> str | None:
    for record in store.get("users", {}).values():
        if record.get("role") == "admin" and record.get("enabled", False):
            return record["username"]
    return None


def _refuse_last_admin(store: dict, username: str) -> None:
    record = store["users"][username]
    if record.get("role") != "admin":
        return
    admins = [
        u for u in store["users"].values()
        if u.get("role") == "admin" and u.get("enabled", True)
        and u["username"] != username
    ]
    if not admins:
        raise ValueError(
            "refusing: this would leave the deployment without an enabled "
            "admin. Create/promote another admin first (break-glass: edit "
            f"{USERS_PATH} directly)."
        )


def _require(record: dict | None, username: str) -> dict:
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    return record


def _unique_key(store: dict) -> str:
    """Globally unique mcp_key (round 2 #7): must not collide with any user
    key. System keys are config/config-env material and practically
    disjoint from token_urlsafe(32) output; the collision loop guards the
    in-store case."""
    users = store.get("users", {})
    while True:
        key = secrets.token_urlsafe(32)
        if not any(u.get("mcp_key") == key for u in users.values()):
            return key


def _tool_names() -> list[str]:
    """Known tool names (for username reservations); tolerant."""
    try:
        tools_dir = Path(__file__).resolve().parents[1]  # .../supreme-mcp-tools/tools
        return [
            d.name.lower() for d in tools_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}_fastmcp.py").exists()
        ]
    except Exception:
        return []


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
