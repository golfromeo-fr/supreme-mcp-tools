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
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
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

_store_cache: dict[str, Any] = {"mtime": None, "store": None}


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


def _public_view(record: dict) -> dict:
    return {k: v for k, v in record.items()
            if k not in ("mcp_key", "password_hash")}


# ---------------------------------------------------------------------------
# file I/O — tolerant, mtime-cached, atomic writes
# ---------------------------------------------------------------------------

def load_users() -> dict:
    """Full store dict; missing/corrupt file → empty store + loud log."""
    try:
        mtime = USERS_PATH.stat().st_mtime if USERS_PATH.exists() else None
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
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(USERS_PATH, store)
    _store_cache["mtime"] = USERS_PATH.stat().st_mtime
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


def create_user(username: str, password: str, role: str = "user",
                servers: list[str] | None = None,
                masked_functions: dict | None = None) -> dict:
    """Create a user; returns {"username", "mcp_key"} — mcp_key shown ONCE."""
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
        "masked_functions": dict(masked_functions or {}),
        "enabled": True,
        "created_at": _now_iso(),
        "key_rotated_at": _now_iso(),
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
    save_users(store)


def set_password(username: str, password: str) -> None:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["password_hash"] = hash_password(password)
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
    save_users(store)
    return {"username": username, "mcp_key": record["mcp_key"]}


def set_servers(username: str, servers: list) -> None:
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    record["servers"] = sorted(set(servers or []))
    save_users(store)


def set_masked_functions(username: str, masks: dict) -> None:
    store = load_users()
    record = store["users"].get((username or "").lower())
    if record is None:
        raise ValueError(f"user '{username}' does not exist")
    clean = {str(k): sorted({str(x) for x in v}) for k, v in (masks or {}).items()}
    record["masked_functions"] = clean
    save_users(store)


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

def tokens_map_for_tool(tool_name: str, system_key: str) -> dict[str, dict]:
    """{system_key: admin entry} ∪ {user key: entry} for reachable users.

    System keys are attributed to the ADMIN USERNAME (round 2: the current
    mono user) unless revoked. Users are omitted unless enabled AND the tool
    is in their servers list. mtime-cached file read inside load_users.
    """
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
        }
    return tokens


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
