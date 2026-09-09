"""E3/M2 — users_store unit tests (isolated store path per test).

Covers the plan's store contract: hashing format, CRUD, tolerant load,
mtime hot-reload, tokens_map_for_tool shapes (system key attributed to
admin, reachability via servers, per-user masked sets), reserved names,
global key uniqueness, last-admin guard.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def store_path(tmp_path, monkeypatch):
    path = tmp_path / "users.json"
    monkeypatch.setattr("tools.shared.users_store.USERS_PATH", path)
    # reset the module-level mtime cache per test (isolation)
    import tools.shared.users_store as us

    monkeypatch.setattr(us, "_store_cache", {"mtime": None, "store": None})
    return path, us


class TestPasswords:
    def test_hash_verify_round_trip(self, store_path):
        _path, us = store_path
        stored = us.hash_password("s3cret-pass")
        assert stored.startswith("pbkdf2_sha256$390000$")
        assert us.verify_password("s3cret-pass", stored)
        assert not us.verify_password("wrong", stored)

    def test_malformed_stored_hash_fails_closed(self, store_path):
        _path, us = store_path
        assert us.verify_password("x", "not-a-hash") is False
        assert us.verify_password("x", "") is False

    def test_authenticate_timing_safe_on_unknown_user(self, store_path):
        _path, us = store_path
        assert us.authenticate("nobody", "whatever") is None


class TestCrud:
    def test_create_returns_key_once_and_stores_hash(self, store_path):
        path, us = store_path
        created = us.create_user("alice", "password-1", role="user",
                                 servers=["simplemcp"])
        assert created["username"] == "alice"
        # mcp_key IS stored in the file (plaintext bearer, config-key threat
        # model); the password is only a HASH
        raw = path.read_text()
        assert created["mcp_key"] in raw
        assert "password-1" not in raw
        assert "pbkdf2_sha256$" in raw

    def test_list_users_never_leaks_secrets(self, store_path):
        _path, us = store_path
        created = us.create_user("bob", "password-2", role="user")
        listed = us.list_users()
        blob = json.dumps(listed)
        assert created["mcp_key"] not in blob
        assert "password_hash" not in blob
        assert listed[0]["username"] == "bob"

    def test_create_rejects_bad_username_short_and_toolname(self, store_path):
        _path, us = store_path
        with pytest.raises(ValueError, match="a-z0-9_-"):
            us.create_user("A!", "password-1")
        with pytest.raises(ValueError, match="reserved"):
            us.create_user("simplemcp", "password-1")
        with pytest.raises(ValueError, match="reserved"):
            us.create_user("system", "password-1")

    def test_create_rejects_short_password_and_bad_role(self, store_path):
        _path, us = store_path
        with pytest.raises(ValueError, match="8 characters"):
            us.create_user("carol", "short")
        with pytest.raises(ValueError, match="role"):
            us.create_user("carol", "password-1", role="superuser")

    def test_duplicate_and_reserved_rejections(self, store_path):
        _path, us = store_path
        us.create_user("alice", "password-1")
        with pytest.raises(ValueError, match="already exists"):
            us.create_user("alice", "password-2")

    def test_delete_and_last_admin_guard(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin")
        with pytest.raises(ValueError, match="without an enabled admin"):
            us.delete_user("root")
        us.create_user("root2", "password-1", role="admin")
        us.delete_user("root")  # second admin exists -> allowed
        assert us.get_user_record("root") is None

    def test_disable_last_admin_guard(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin")
        with pytest.raises(ValueError, match="without an enabled admin"):
            us.set_enabled("root", False)

    def test_rotate_key_replaces_and_is_unique(self, store_path):
        _path, us = store_path
        created = us.create_user("alice", "password-1")
        rotated = us.rotate_key("alice")
        assert rotated["mcp_key"] != created["mcp_key"]
        record = us.get_user_record("alice")
        assert record["mcp_key"] == rotated["mcp_key"]
        with pytest.raises(ValueError):
            us.rotate_key("ghost")

    def test_authenticate_round_trip(self, store_path):
        _path, us = store_path
        us.create_user("alice", "password-1")
        user = us.authenticate("alice", "password-1")
        assert user is not None and user["username"] == "alice"
        assert "mcp_key" not in user and "password_hash" not in user
        assert us.authenticate("alice", "wrong") is None

    def test_role_user_gets_default_masks(self, store_path):
        """E3 dig: role=user accounts are seeded with the destructive-tool
        mask profile (memorymcp deletes/expiry, ragmcp index mutations,
        databasemcp writes) — overridable per user via the Users tab."""
        _path, us = store_path
        us.create_user("alice", "password-1", role="user",
                       servers=["memorymcp", "ragmcp", "databasemcp"])
        masked = us.get_user_record("alice")["masked_functions"]
        assert "deleteMemory" in masked["memorymcp"]
        assert "decayOrExpire" in masked["memorymcp"]
        assert "clear_index" in masked["ragmcp"]
        assert "execute_sql" in masked["databasemcp"]

    def test_role_admin_gets_no_default_masks(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin",
                       servers=["memorymcp"])
        assert us.get_user_record("root")["masked_functions"] == {}

    def test_updated_at_stamped_on_every_mutation(self, store_path):
        """E3/M4 seam: per-record updated_at is the future DB migration's
        last-writer-wins key — every mutation must stamp it."""
        _path, us = store_path
        created = us.create_user("alice", "password-1")
        rec = us.get_user_record("alice")
        assert "updated_at" in rec

        us.set_servers("alice", ["simplemcp"])
        assert us.get_user_record("alice")["updated_at"] >= rec["updated_at"]

        us.set_masked_functions("alice", {"simplemcp": ["x"]})
        assert us.get_user_record("alice")["updated_at"] >= rec["updated_at"]

        us.set_password("alice", "new-password-9")
        assert us.get_user_record("alice")["updated_at"] >= rec["updated_at"]

        us.set_enabled("alice", False)
        assert us.get_user_record("alice")["updated_at"] >= rec["updated_at"]

        us.set_enabled("alice", True)
        us.rotate_key("alice")
        assert us.get_user_record("alice")["updated_at"] >= rec["updated_at"]


class TestTokensMapForTool:
    def test_system_key_attributed_to_admin(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin")
        tokens = us.tokens_map_for_tool("simplemcp", "SYSTEM-KEY")
        entry = tokens["SYSTEM-KEY"]
        assert entry["client_id"] == "root" and entry["role"] == "admin"

    def test_users_only_when_server_reachable_and_enabled(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin")
        us.create_user("alice", "password-1", servers=["simplemcp"])
        us.create_user("bob", "password-1", servers=["webmcp"])
        us.create_user("carol", "password-1", servers=["simplemcp"])
        us.set_enabled("carol", False)
        tokens = us.tokens_map_for_tool("simplemcp", "SYSTEM-KEY")
        alice_key = us.get_user_record("alice")["mcp_key"]
        carol_key = us.get_user_record("carol")["mcp_key"]
        bob_key = us.get_user_record("bob")["mcp_key"]
        assert alice_key in tokens
        assert bob_key not in tokens       # different server
        assert carol_key not in tokens     # disabled
        assert tokens[alice_key]["client_id"] == "alice"

    def test_masked_functions_scoped_per_tool(self, store_path):
        _path, us = store_path
        us.create_user(
            "alice", "password-1", servers=["databasemcp"],
            masked_functions={"databasemcp": ["execute_sql"]},
        )
        alice_key = us.get_user_record("alice")["mcp_key"]
        tokens = us.tokens_map_for_tool("databasemcp", "SYSTEM-KEY")
        assert tokens[alice_key]["masked"] == ["execute_sql"]
        tokens_other = us.tokens_map_for_tool("webmcp", "SYSTEM-KEY")
        assert alice_key not in tokens_other  # server not granted

    def test_revoked_system_key_disappears(self, store_path):
        _path, us = store_path
        us.create_user("root", "password-1", role="admin")
        assert "SYSTEM-KEY" in us.tokens_map_for_tool("simplemcp", "SYSTEM-KEY")
        us.revoke_system_key("simplemcp")
        assert "SYSTEM-KEY" not in us.tokens_map_for_tool("simplemcp", "SYSTEM-KEY")


class TestTolerantLoad:
    def test_missing_file_loads_empty(self, store_path):
        _path, us = store_path
        assert us.load_users() == {"version": 1, "users": {}}

    def test_corrupt_file_tolerated(self, store_path):
        _path, us = store_path
        _path.write_text("{not json")
        store = us.load_users()
        assert store.get("users") == {}

    def test_hot_reload_on_mtime_change(self, store_path):
        _path, us = store_path
        us.create_user("alice", "password-1")
        # external edit (the documented operational path) with an mtime
        # bumped well past the cached one — mtime granularity on some
        # filesystems makes "just written" ambiguous
        import os
        import time

        data = json.loads(_path.read_text())
        data["users"]["bob"] = {
            "username": "bob", "password_hash": "x", "role": "user",
            "mcp_key": "bob-key", "servers": [], "masked_functions": {},
            "enabled": True, "created_at": "now", "key_rotated_at": "now",
        }
        _path.write_text(json.dumps(data))
        future = time.time() + 10
        os.utime(_path, (future, future))
        assert us.get_user_record("bob") is not None


def test_central_tokens_admins_only_enabled_only(tmp_path, monkeypatch):
    """central_tokens surfaces ENABLED ADMINS' mcp_keys only — the E3
    multi-admin credential map for the central API."""
    from tools.shared import users_store

    store = {"version": 1, "users": {
        "root": {"username": "root", "role": "admin", "enabled": True,
                 "mcp_key": "key-admin-1"},
        "second": {"username": "second", "role": "admin", "enabled": True,
                   "mcp_key": "key-admin-2"},
        "plain": {"username": "plain", "role": "user", "enabled": True,
                  "mcp_key": "key-user-1"},
        "off": {"username": "off", "role": "admin", "enabled": False,
                "mcp_key": "key-admin-disabled"},
    }}
    monkeypatch.setattr(users_store, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(users_store, "_store_cache",
                        {"mtime": None, "store": store})

    tokens = users_store.central_tokens()
    assert set(tokens) == {"key-admin-1", "key-admin-2"}
    assert tokens["key-admin-1"] == {"client_id": "root", "role": "admin"}


# ── E3 db backend (MCP_USERS_BACKEND=db) ─────────────────────────────

class _FakeCursor:
    def __init__(self, state, params):
        self._state, self._params = state, params

    def fetchone(self):
        sql = self._state["last_sql"]
        if "COUNT(*)" in sql:
            return [1 if self._state.get("has_row") else 0]
        if sql.startswith("SELECT data"):
            if self._state.get("has_row"):
                return [self._state["data"]]
            return None
        return None


class _FakeConn:
    """Minimal DB-API connection standing in for libsql/psycopg."""

    def __init__(self):
        self.state = {"has_row": False, "data": None, "last_sql": "",
                      "statements": []}

    def execute(self, sql, params=()):
        self.state["last_sql"] = sql
        self.state["statements"].append((sql, params))
        if sql.startswith("INSERT INTO mcp_users_store"):
            self.state["has_row"] = True
            self.state["data"] = params[0]
        return _FakeCursor(self.state, params)


class _FakeSqlStore:
    is_available = True

    def __init__(self):
        self._conn = _FakeConn()


import tools.shared.users_store as users_store  # noqa: E402


def _activate_db_backend(monkeypatch):
    """Point MCP_USERS_BACKEND=db at a fake SqlStore; resets singletons."""
    import tools.shared.sql_store as sql_store_mod
    fake = _FakeSqlStore()
    monkeypatch.setenv("MCP_USERS_BACKEND", "db")
    monkeypatch.setattr(sql_store_mod, "get_sql_store", lambda: fake)
    monkeypatch.setattr(users_store, "_db_conn_singleton", None)
    monkeypatch.setattr(users_store, "_db_init_done", False)
    return fake


def test_db_backend_round_trip(tmp_path, monkeypatch):
    """save_users→load_users through the db backend preserves the document;
    nothing is written to users.json in db mode."""
    fake = _activate_db_backend(monkeypatch)
    monkeypatch.setattr(users_store, "USERS_PATH", tmp_path / "users.json")

    users_store.save_users({"version": 1, "users": {
        "admin": {"username": "admin", "role": "admin", "mcp_key": "k1"}}})
    loaded = users_store.load_users()
    assert loaded["users"]["admin"]["mcp_key"] == "k1"
    assert not (tmp_path / "users.json").exists()  # db is the only sink


def test_db_backend_one_time_import_from_json(tmp_path, monkeypatch):
    """First db use imports an existing users.json (adoption path)."""
    fake = _activate_db_backend(monkeypatch)
    users_json = tmp_path / "users.json"
    users_json.write_text(json.dumps({"version": 1, "users": {
        "admin": {"username": "admin", "role": "admin", "mcp_key": "seed"}}}))
    monkeypatch.setattr(users_store, "USERS_PATH", users_json)

    loaded = users_store.load_users()
    assert loaded["users"]["admin"]["mcp_key"] == "seed"
    # the imported doc landed in the table
    assert fake._conn.state["has_row"]


def test_db_backend_unavailable_falls_back_to_json(tmp_path, monkeypatch):
    """db requested but no SQL backend → loud fallback to users.json."""
    import tools.shared.sql_store as sql_store_mod
    from tools.shared.sql_store import NullSqlStore

    monkeypatch.setenv("MCP_USERS_BACKEND", "db")
    monkeypatch.setattr(sql_store_mod, "get_sql_store", lambda: NullSqlStore())
    monkeypatch.setattr(users_store, "_db_conn_singleton", None)
    monkeypatch.setattr(users_store, "_db_init_done", False)
    monkeypatch.setattr(users_store, "USERS_PATH", tmp_path / "users.json")

    users_store.create_user("fallback", "fallback-pass-9")
    assert users_store.get_user_record("fallback") is not None
    assert (tmp_path / "users.json").exists()
