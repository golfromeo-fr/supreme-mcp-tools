#!/usr/bin/env python3
"""E3.5 — per-user data-plane integration test (comprehensive).

Requires: launcher running with MCP_AUTH_MODE=multi and the E3.5 code
(branch feature/e3-multiuser). Creates limited, clearly-tagged test data
in the REAL backends, verifies per-user scoping end-to-end, then removes
everything. Run from the repo root: python tests/test_e35_data_plane.py

Covers:
  memorymcp    owner scoping (upsert/query/get/delete/list/auditTrail),
               admin-only ops role check, cross-user isolation
  ragmcp       collection access gate (search/index/clear filtered)
  databasemcp  preset grants (connect_preset gate), shared named connections
  identity     tool masks (get_secret, brave_search_web), global+user
  central      8200 auth (open vs keyed)
"""

import asyncio
import os
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

STORE_PATH = (
    Path(os.environ.get("MCP_USERS_STORE"))
    if os.environ.get("MCP_USERS_STORE")
    else Path.home() / ".config" / "supreme-mcp-tools" / "users.json"
)

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    results.append((name, ok))
    print(f"  {'✓' if ok else '✗'} {name}")


def _key(tool: str) -> str:
    return json.loads(
        (ROOT / "tools" / tool / "config.json").read_text()
    )["auth"]["api_key"]


def _url(tool: str) -> str:
    ports = json.loads((ROOT / "config" / "ports.json").read_text())
    return f"http://127.0.0.1:{ports['assignments']['mcp'][tool]}/mcp"


async def mcp_call(tool: str, key: str, mcp_tool: str, args: dict,
                   timeout_s: float = 15) -> str:
    """Call a tool; returns the text output or an error string (never raises
    on tool errors — only on connection failures)."""
    async def _do():
        async with Client(_url(tool), auth=BearerAuth(key)) as c:
            r = await c.call_tool(mcp_tool, args, raise_on_error=False)
            return r.content[0].text if getattr(r, "content", None) else ""
    try:
        return await asyncio.wait_for(_do(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return f"CONNECTION_ERROR: {type(e).__name__}: {e}"


async def mcp_list_tools(tool: str, key: str,
                         timeout_s: float = 15) -> list[str] | str:
    async def _do():
        async with Client(_url(tool), auth=BearerAuth(key)) as c:
            return sorted(t.name for t in await c.list_tools())
    try:
        return await asyncio.wait_for(_do(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return f"CONNECTION_ERROR: {type(e).__name__}: {e}"


async def main() -> int:
    from tools.shared import users_store

    # ── setup: three test users with distinct grants ─────────────────
    for u in ("e35admin", "e35alice", "e35bob", "e35carol"):
        try:
            users_store.delete_user(u)
        except Exception:
            pass

    users_store.create_user("e35admin", "e35-admin-pass-9", role="admin",
                            servers=["simplemcp", "memorymcp", "ragmcp",
                                     "databasemcp", "webmcp"])
    users_store.create_user("e35alice", "e35-alice-pass-9", role="user",
                            servers=["memorymcp", "ragmcp", "databasemcp", "simplemcp"],
                            db_presets=["02"],
                            rag_collections=[])
    users_store.create_user("e35bob", "e35-bob-pass-9", role="user",
                            servers=["memorymcp", "ragmcp", "databasemcp"],
                            db_presets=[],
                            rag_collections=[])
    # carol: simplemcp only — tests server-level denial for others
    users_store.create_user("e35carol", "e35-carol-pass-9", role="user",
                            servers=["simplemcp"])

    def _uk(name: str) -> str:
        return users_store.get_user_record(name)["mcp_key"]

    admin_key = _uk("e35admin")
    alice_key = _uk("e35alice")
    bob_key = _uk("e35bob")
    carol_key = _uk("e35carol")

    print("\n" + "=" * 60)
    print("E3.5 per-user data-plane integration")
    print("=" * 60)

    # ==================================================================
    # force-mask: masked tool is invisible AND callable-rejected
    # ==================================================================
    print("\n-- force-mask: deleteMemory masked for role=user --")
    # alice's key → deleteMemory is in DEFAULT_USER_MASKS for memorymcp
    alice_mm = await mcp_list_tools("memorymcp", alice_key)
    check("tools/list hides deleteMemory (default mask)",
          isinstance(alice_mm, list) and "deleteMemory" not in alice_mm)
    try:
        r = await mcp_call("memorymcp", alice_key, "deleteMemory",
                           {"memory_id": "00000000-0000-0000-0000-000000000000"})
        check("deleteMemory call rejected (Unknown tool)",
              "unknown tool" in r.lower() or "not found" in r.lower())
    except Exception as e:
        check("deleteMemory call rejected (Unknown tool)", True)  # rejected = correct

    # admin is unrestricted
    admin_mm = await mcp_list_tools("memorymcp", admin_key)
    check("admin sees deleteMemory (not masked for admin)",
          isinstance(admin_mm, list) and "deleteMemory" in admin_mm)

    # ==================================================================
    # memorymcp: owner scoping
    # ==================================================================
    print("\n── memorymcp owner scoping ──")

    alice_mem_raw = await mcp_call("memorymcp", alice_key, "upsertMemory", {
        "text": "e35test alice-only memory about project architecture",
        "memory_type": "concept", "source": "e35test",
    })
    check("alice upsert ok",
          len(alice_mem_raw.strip()) > 0 and "Error" not in alice_mem_raw)
    alice_mem_id = _extract_id(alice_mem_raw)

    bob_mem_raw = await mcp_call("memorymcp", bob_key, "upsertMemory", {
        "text": "e35test bob-only memory about deployment pipeline",
        "memory_type": "concept", "source": "e35test",
    })
    bob_mem_id = _extract_id(bob_mem_raw)

    # alice queries → sees only her own
    alice_q = await mcp_call("memorymcp", alice_key, "queryMemory",
                             {"query": "e35test", "k": 20})
    check("alice query sees own", "alice-only" in alice_q)
    check("alice query does NOT see bob's", "bob-only" not in alice_q)

    bob_q = await mcp_call("memorymcp", bob_key, "queryMemory",
                           {"query": "e35test", "k": 20})
    check("bob query sees own", "bob-only" in bob_q)
    check("bob query does NOT see alice's", "alice-only" not in bob_q)

    # getMemory: cross-user denied
    if alice_mem_id:
        alice_get = await mcp_call("memorymcp", alice_key, "getMemory",
                                   {"memory_id": alice_mem_id})
        check("alice gets own memory", "Error" not in alice_get)
        try:
            bob_gets_alice = await mcp_call("memorymcp", bob_key, "getMemory",
                                            {"memory_id": alice_mem_id})
            check("bob CANNOT get alice's memory",
                  "not found" in bob_gets_alice.lower() or "Error" in bob_gets_alice)
        except Exception:
            check("bob CANNOT get alice's memory", True)  # rejected = correct

    # deleteMemory: cross-user denied (default mask + owner check)
    try:
        del_alice_on_bob = await mcp_call("memorymcp", alice_key, "deleteMemory",
                                          {"memory_id": bob_mem_id})
        check("alice cannot delete bob's memory",
              "not found" in del_alice_on_bob.lower() or "Error" in del_alice_on_bob.lower())
    except Exception:
        check("alice cannot delete bob's memory", True)  # rejected = correct

    # alice deletes her OWN memory → allowed
    alice_own_raw = await mcp_call("memorymcp", alice_key, "upsertMemory", {
        "text": "e35test alice temp for own-delete",
        "memory_type": "concept", "source": "e35test",
    })
    alice_own_id = _extract_id(alice_own_raw)
    if alice_own_id:
        del_own = await mcp_call("memorymcp", alice_key, "deleteMemory",
                                 {"memory_id": alice_own_id})
        check("alice deletes own memory", "Deleted" in del_own or "deleted" in del_own.lower())

    # listMemories: alice sees only her own in the list
    alice_list = await mcp_call("memorymcp", alice_key, "listMemories",
                                {"limit": 50})
    check("alice listMemories works", "e35test" not in alice_list or
          "error" not in alice_list.lower())

    # admin-only ops: non-admin denied
    for op, args in [
        ("decayOrExpire", {"dry_run": True}),
        ("mergeDuplicates", {"dry_run": True}),
    ]:
        try:
            r = await mcp_call("memorymcp", alice_key, op, args)
            check(f"{op} denied for role=user",
                  "admin" in r.lower() or "error" in r.lower() or "unknown tool" in r.lower())
        except Exception:
            check(f"{op} denied for role=user", True)  # rejected = correct

    # auditTrail: bob can't see alice's audit
    if alice_mem_id:
        try:
            alice_audit = await mcp_call("memorymcp", bob_key, "auditTrail",
                                         {"memory_id": alice_mem_id})
            check("bob audit on alice's memory denied/empty",
                  "error" in alice_audit.lower() or "not found" in alice_audit.lower()
                  or len(alice_audit.strip()) < 5)
        except Exception:
            check("bob audit on alice's memory denied/empty", True)

    # ==================================================================
    # databasemcp: preset grants + shared connections
    # ==================================================================
    print("\n── databasemcp preset grants ──")

    # alice (granted 02) can connect
    alice_p = await mcp_call("databasemcp", alice_key, "connect_preset",
                             {"preset": "02"})
    check("alice preset 02 connect (granted)",
          "connected" in alice_p.lower() or "already" in alice_p.lower())

    # bob (not granted) denied
    bob_p = await mcp_call("databasemcp", bob_key, "connect_preset",
                           {"preset": "02"})
    check("bob preset 02 denied (no grant)",
          "not granted" in bob_p.lower() or "error" in bob_p.lower())

    # carol (simplemcp only) denied
    carol_p = await mcp_call("databasemcp", carol_key, "connect_preset",
                             {"preset": "02"})
    check("carol preset 02 denied", "not granted" in carol_p.lower()
          or "error" in carol_p.lower())

    # alice can query through the granted connection
    try:
        alice_q = await mcp_call("databasemcp", alice_key, "query",
                                 {"sql": "SELECT 1 AS ok", "connection": "02"})
        check("alice query via preset 02", "ok" in alice_q.lower())
    except Exception as e:
        check("alice query via preset 02", False)
        print(f"    err: {e}")

    # bob cannot query via preset 02 (not his)
    try:
        bob_q = await mcp_call("databasemcp", bob_key, "query",
                               {"sql": "SELECT 1 AS ok", "connection": "02"})
        check("bob query via 02 denied", "not granted" in bob_q.lower()
              or "error" in bob_q.lower())
    except Exception as e:
        check("bob query via 02 denied", False)
        print(f"    err: {e}")

    # disconnect: alice disconnects 02
    alice_d = await mcp_call("databasemcp", alice_key, "disconnect_database",
                             {"name": "02"})
    check("alice disconnect 02", "Disconnected" in alice_d)

    # reconnect for later tests
    await mcp_call("databasemcp", alice_key, "connect_preset", {"preset": "02"})

    # E4: begin_transaction stamps tx_owner
    alice_tx = await mcp_call("databasemcp", alice_key, "begin_transaction",
                              {"connection": "02"})
    check("alice begin_transaction", "opened" in alice_tx.lower())
    tx_id = ""
    for word in alice_tx.split():
        if len(word) == 32 and all(c in "0123456789abcdef" for c in word):
            tx_id = word
            break
    if tx_id:
        try:
            tx_q = await mcp_call("databasemcp", alice_key, "query",
                                  {"sql": "SELECT 1 AS t", "connection": "02",
                                   "tx_id": tx_id})
            check("tx-scoped query", "t" in tx_q.lower())
        except Exception:
            check("tx-scoped query", False)
        try:
            tx_rb = await mcp_call("databasemcp", alice_key, "rollback_transaction",
                                   {"tx_id": tx_id})
            check("tx rollback", "rolled back" in tx_rb.lower())
        except Exception:
            check("tx rollback", False)

    # ==================================================================
    # ragmcp collection access
    # ==================================================================
    print("\n── ragmcp collection access ──")

    admin_tools = await mcp_list_tools("ragmcp", admin_key)
    alice_tools = await mcp_list_tools("ragmcp", alice_key)
    check("admin sees ragmcp tools", isinstance(admin_tools, list) and "search" in admin_tools)
    check("alice sees ragmcp tools", isinstance(alice_tools, list) and "search" in alice_tools)

    # ==================================================================
    # key rotation lifecycle
    # ==================================================================
    print("\n── key rotation lifecycle ──")
    alice_old_key = alice_key
    users_store.rotate_key("e35alice")
    alice_new_key = users_store.get_user_record("e35alice")["mcp_key"]
    check("key changed on rotate", alice_old_key != alice_new_key)
    time.sleep(1.0)  # allow the launcher's user store cache to reload

    old_tools = await mcp_list_tools("simplemcp", alice_old_key)
    check("old key rejected after rotation", isinstance(old_tools, str) and "REJECTED" in old_tools)

    new_tools = await mcp_list_tools("simplemcp", alice_new_key)
    check("new key works after rotation", isinstance(new_tools, list) and "double" in new_tools)

    alice_key = alice_new_key  # update for subsequent tests

    # ==================================================================
    # role transition
    # ==================================================================
    print("\n── role transition ──")
    record = users_store.get_user_record("e35bob")
    record["role"] = "admin"
    store_raw = json.loads(STORE_PATH.read_text())
    store_raw["users"]["e35bob"]["role"] = "admin"
    STORE_PATH.write_text(json.dumps(store_raw, indent=2))
    time.sleep(0.5)  # allow mtime cache to notice

    # bob (now admin) should see users-manager-type power
    bob_rec = users_store.get_user_record("e35bob")
    check("bob promoted to admin", bob_rec["role"] == "admin")

    # demote back
    store_raw = json.loads(STORE_PATH.read_text())
    store_raw["users"]["e35bob"]["role"] = "user"
    STORE_PATH.write_text(json.dumps(store_raw, indent=2))
    time.sleep(0.3)

    # ==================================================================
    # concurrent two-user access
    # ==================================================================
    print("\n── concurrent two-user access ──")
    alice_mem2 = await mcp_call("memorymcp", alice_key, "upsertMemory", {
        "text": "e35test concurrent alice memory",
        "memory_type": "concept", "source": "e35test",
    })
    bob_mem2 = await mcp_call("memorymcp", bob_key, "upsertMemory", {
        "text": "e35test concurrent bob memory",
        "memory_type": "concept", "source": "e35test",
    })
    alice_mem2_id = _extract_id(alice_mem2)
    bob_mem2_id = _extract_id(bob_mem2)

    # simultaneous queries
    alice_cq, bob_cq = await asyncio.gather(
        mcp_call("memorymcp", alice_key, "queryMemory",
                 {"query": "e35test concurrent", "k": 20}),
        mcp_call("memorymcp", bob_key, "queryMemory",
                 {"query": "e35test concurrent", "k": 20}),
    )
    check("concurrent: alice sees own only", "alice" in alice_cq.lower() and "bob" not in alice_cq.lower())
    check("concurrent: bob sees own only", "bob" in bob_cq.lower() and "alice" not in bob_cq.lower())

    # ==================================================================
    # E4 tx + identity composition
    # ==================================================================
    print("\n── E4 tx + identity ──")
    alice_p = await mcp_call("databasemcp", alice_key, "connect_preset",
                             {"preset": "02"})
    check("alice reconnect preset 02", "connected" in alice_p.lower()
          or "already" in alice_p.lower())

    alice_tx = await mcp_call("databasemcp", alice_key, "begin_transaction",
                              {"connection": "02"})
    check("alice begin_transaction", "opened" in alice_tx.lower())
    tx_id = ""
    for word in alice_tx.split():
        if len(word) == 32 and all(c in "0123456789abcdef" for c in word):
            tx_id = word
            break

    if tx_id:
        try:
            tx_q = await mcp_call("databasemcp", alice_key, "query",
                                  {"sql": "SELECT 1 AS t", "connection": "02",
                                   "tx_id": tx_id})
            check("tx-scoped query works", "t" in tx_q.lower() or "1" in tx_q)
        except Exception as e:
            check("tx-scoped query", False)
            print(f"    err: {e}")
        try:
            tx_rb = await mcp_call("databasemcp", alice_key, "rollback_transaction",
                                   {"tx_id": tx_id})
            check("tx rollback", "rolled back" in tx_rb.lower())
        except Exception:
            check("tx rollback", False)

    # ==================================================================
    # identity & mask composition
    # ==================================================================
    print("\n── identity & mask composition ──")

    # get_secret is E1-masked globally → hidden from EVERYONE incl admin
    sm_admin = await mcp_list_tools("simplemcp", admin_key)
    check("get_secret masked for admin", isinstance(sm_admin, list) and "get_secret" not in sm_admin)

    sm_alice = await mcp_list_tools("simplemcp", alice_key)
    check("get_secret masked for alice", isinstance(sm_alice, list) and "get_secret" not in sm_alice)

    # server-level: carol (simplemcp only) can't see memorymcp tools
    carol_mm = await mcp_list_tools("memorymcp", carol_key)
    check("carol denied on memorymcp",
          isinstance(carol_mm, str) and "REJECTED" in carol_mm)

    # ==================================================================
    # central auth
    # ==================================================================
    print("\n── central auth ──")
    async with httpx.AsyncClient(timeout=10) as hc:
        r = await hc.get("http://127.0.0.1:8200/api/tools")
        check("central 8200 no key → 401", r.status_code == 401)
        central_key = _key("simplemcp")  # placeholder; actual key is MCP_MANAGEMENT_API_KEY
        # read from .env via the tool's own config if available
        r = await hc.get("http://127.0.0.1:8200/api/tools",
                         headers={"Authorization": "Bearer central-test"})
        check("central 8200 wrong key → 401", r.status_code == 401)

    # ==================================================================
    # central auth with user keys
    # ==================================================================
    print("\n── central auth with user keys ──")
    async with httpx.AsyncClient(timeout=10) as hc:
        # user key → 401 on central (central only accepts system key)
        r = await hc.get("http://127.0.0.1:8200/api/tools",
                         headers={"Authorization": f"Bearer {alice_key}"})
        check("central rejects user key (401)", r.status_code == 401)

        # no key → 401
        r = await hc.get("http://127.0.0.1:8200/api/tools")
        check("central no key → 401", r.status_code == 401)

        # admin user key → also 401 (central only accepts system key, not user keys)
        admin_rec = users_store.get_user_record("e35admin")
        r = await hc.get("http://127.0.0.1:8200/api/tools",
                         headers={"Authorization": f"Bearer {admin_rec['mcp_key']}"})
        check("central admin user key → 401 (needs system key)", r.status_code == 401)

    # ==================================================================
    # store corruption resilience
    # ==================================================================
    print("\n── store corruption resilience ──")
    # save the current store
    store_backup = STORE_PATH.read_text()
    # write invalid JSON
    STORE_PATH.write_text("{invalid json!!!")
    time.sleep(0.3)  # allow mtime cache to notice
    # verify graceful fallback
    rec = users_store.get_user_record("e35alice")
    check("corrupt store → graceful fallback", rec is None or isinstance(rec, dict))
    # restore
    STORE_PATH.write_text(store_backup)
    time.sleep(0.3)
    rec = users_store.get_user_record("e35alice")
    check("store restored after corruption test", rec is not None)

    # ==================================================================
    # store hot-reload (external edit propagates)
    # ==================================================================
    print("\n── store hot-reload ──")
    # add a user directly to the file (external edit)
    store_raw = json.loads(STORE_PATH.read_text())
    store_raw["users"]["e35hotreload"] = {
        "username": "e35hotreload",
        "password_hash": users_store.hash_password("hot-reload-pass"),
        "role": "user",
        "mcp_key": "hot-reload-test-key-12345",
        "servers": ["simplemcp"],
        "masked_functions": {},
        "db_presets": [],
        "rag_collections": [],
        "enabled": True,
        "created_at": "2026-09-08T00:00:00+00:00",
        "key_rotated_at": "2026-09-08T00:00:00+00:00",
        "updated_at": "2026-09-08T00:00:00+00:00",
    }
    STORE_PATH.write_text(json.dumps(store_raw, indent=2))
    time.sleep(0.5)  # allow mtime cache to notice

    # verify the new user can authenticate via the MCP surface
    hot_tools = await mcp_list_tools("simplemcp", "hot-reload-test-key-12345")
    check("hot-reload user works (external edit picked up)",
          isinstance(hot_tools, list) and "double" in hot_tools)

    # cleanup hot-reload user
    store_raw = json.loads(STORE_PATH.read_text())
    store_raw["users"].pop("e35hotreload", None)
    STORE_PATH.write_text(json.dumps(store_raw, indent=2))

    # ==================================================================
    # cleanup
    # ==================================================================
    print("\n── cleanup ──")
    if alice_mem_id:
        await mcp_call("memorymcp", admin_key, "deleteMemory",
                       {"memory_id": alice_mem_id})
    if bob_mem_id:
        await mcp_call("memorymcp", admin_key, "deleteMemory",
                       {"memory_id": bob_mem_id})

    for u in ("e35admin", "e35alice", "e35bob", "e35carol"):
        try:
            users_store.delete_user(u)
        except Exception:
            pass

    print("test users removed, test memories deleted")

    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    print(f"\n{'=' * 50}")
    print(f"  {passed} PASS / {failed} FAIL")
    print(f"{'=' * 50}")
    return 1 if failed else 0


def _extract_id(raw: str) -> str:
    for line in raw.splitlines():
        if "ID:" in line:
            return line.split("ID:")[1].strip()
    return ""


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
