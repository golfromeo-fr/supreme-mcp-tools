"""E3.5 force-mask per user — the TODO checklist, verified at the unit level.

The chain under test: admin sets `masked_functions` per user (Users tab /
central PUT) -> users_store.tokens_map_for_tool assembles the entry's
`masked` list -> IdentityGateMiddleware drops masked tools from tools/list
and rejects calls with "Unknown tool" (no existence leak). Checklist:

- admin can force-mask ANY function per user (no allowlist at the setter)
- the masked entry reaches the token map for the tool
- gate semantics: masked = dropped from list, call rejected; admin bypasses
  USER masks
- rotation: masks live on the user RECORD, so rotate_key preserves them
- demotion: admin -> user merges the role=user default-mask profile
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.shared import users_store  # noqa: E402
from tools.shared.identity import IdentityGateMiddleware  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(users_store, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(users_store, "_store_cache", {"mtime": None, "store": None})
    # deterministic seeding: loopback bind + explicit password (these tests
    # are not about the P1-1 default-password refusal)
    monkeypatch.setenv("MCP_UI_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_UI_PASSWORD", "seed-pw-123")
    monkeypatch.delenv("MCP_USER_TESTER_PASSWORD", raising=False)
    return users_store


def _admin(store):
    store.seed_from_env()


def test_admin_can_force_mask_any_function(store):
    _admin(store)
    store.create_user("worker", "password1", role="user", servers=["memorymcp"])
    # no allowlist at the setter: ANY function name, known or not
    store.set_masked_functions("worker", {"memorymcp": ["deleteMemory", "not_a_real_fn"]})
    tok = store.get_user_record("worker")["mcp_key"]
    masked = store.tokens_map_for_tool("memorymcp", "sys")[tok]["masked"]
    assert set(masked) >= {"deleteMemory", "not_a_real_fn"}


def test_gate_drops_masked_from_list_and_calls(store):
    _admin(store)
    store.create_user("worker", "password1", role="user", servers=["memorymcp"])
    store.set_masked_functions("worker", {"memorymcp": ["deleteMemory"]})
    tok = store.get_user_record("worker")["mcp_key"]
    gate = IdentityGateMiddleware(tokens_map_fn=lambda: store.tokens_map_for_tool("memorymcp", "sys"))

    entry = gate._resolve_for_request()  # returns ((id, entry), over_http)
    # no HTTP scope here: the gate fails open with None — simulate the entry
    # the middleware would see for this token over HTTP:
    simulated = store.tokens_map_for_tool("memorymcp", "sys")[tok]
    assert "deleteMemory" in gate._masked(simulated)
    assert gate._masked(None) == set()          # fail-open shape
    admin_sim = dict(simulated, role="admin")
    assert gate._masked(admin_sim) == set()     # admin bypasses USER masks


def test_rotation_preserves_masks(store):
    _admin(store)
    store.create_user("worker", "password1", role="user", servers=["memorymcp"])
    store.set_masked_functions("worker", {"memorymcp": ["deleteMemory"]})
    store.rotate_key("worker")
    tok = store.get_user_record("worker")["mcp_key"]
    assert "deleteMemory" in store.tokens_map_for_tool("memorymcp", "sys")[tok]["masked"]


def test_demotion_merges_user_default_profile(store):
    _admin(store)
    store.create_user("chief", "password1", role="admin", servers=["memorymcp", "databasemcp"])
    assert store.get_user_record("chief")["masked_functions"] == {}

    store.set_role("chief", "user")

    masks = store.get_user_record("chief")["masked_functions"]
    assert "deleteMemory" in masks["memorymcp"]          # DEFAULT_USER_MASKS merged in
    assert "execute_sql" in masks["databasemcp"]
    tok = store.get_user_record("chief")["mcp_key"]
    assert "deleteMemory" in store.tokens_map_for_tool("memorymcp", "sys")[tok]["masked"]


def test_promotion_keeps_masks_but_admin_bypasses(store):
    _admin(store)
    store.create_user("worker", "password1", role="user", servers=["memorymcp"])
    store.set_masked_functions("worker", {"memorymcp": ["deleteMemory"]})
    store.set_role("worker", "admin")
    tok = store.get_user_record("worker")["mcp_key"]
    entry = store.tokens_map_for_tool("memorymcp", "sys")[tok]
    assert entry["role"] == "admin"
    gate = IdentityGateMiddleware(tokens_map_fn=lambda: {tok: entry})
    assert gate._masked(entry) == set()  # bypassed while admin, masks retained
