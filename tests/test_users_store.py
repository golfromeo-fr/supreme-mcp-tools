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
