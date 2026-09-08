#!/usr/bin/env python3
"""E3.5 — per-user data-plane integration test with real test data.

Requires: launcher running with MCP_AUTH_MODE=multi and the E3.5 code
(branch feature/e3-multiuser). Creates limited, clearly-tagged test data
in the REAL backends, verifies per-user scoping end-to-end, then removes
everything. Run from the repo root: python tests/test_e35_data_plane.py
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

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


async def mcp_call(tool: str, key: str, mcp_tool: str, args: dict) -> str:
    async with Client(_url(tool), auth=BearerAuth(key)) as c:
        r = await c.call_tool(mcp_tool, args)
        return r.content[0].text if getattr(r, "content", None) else ""


async def mcp_list_tools(tool: str, key: str) -> list[str]:
    async with Client(_url(tool), auth=BearerAuth(key)) as c:
        return sorted(t.name for t in await c.list_tools())


async def main() -> int:
    from tools.shared import users_store

    # ── setup: two test users ─────────────────────────────────────────
    for u in ("e35alice", "e35bob", "e35admin"):
        try:
            users_store.delete_user(u)
        except Exception:
            pass

    users_store.create_user("e35admin", "e35-admin-pass-9", role="admin",
                            servers=["simplemcp", "memorymcp", "ragmcp", "databasemcp"])
    users_store.create_user("e35alice", "e35-alice-pass-9", role="user",
                            servers=["memorymcp", "ragmcp", "databasemcp"],
                            db_presets=["02"])
    users_store.create_user("e35bob", "e35-bob-pass-9", role="user",
                            servers=["memorymcp", "ragmcp", "databasemcp"])

    def _uk(name: str) -> str:
        return users_store.get_user_record(name)["mcp_key"]

    admin_key = _uk("e35admin")
    alice_key = _uk("e35alice")
    bob_key = _uk("e35bob")

    print("\n=== E3.5 data-plane integration ===\n")

    # ── memorymcp: owner scoping with REAL memories ──────────────────
    print("-- memorymcp owner scoping --")
    alice_mem_raw = await mcp_call("memorymcp", alice_key, "upsertMemory", {
        "text": "e35test alice-only memory about project architecture",
        "memory_type": "concept", "source": "e35test",
    })
    check("alice upsert ok", len(alice_mem_raw.strip()) > 0 and "Error" not in alice_mem_raw)
    alice_mem_id = ""
    for line in alice_mem_raw.splitlines():
        if "ID:" in line:
            alice_mem_id = line.split("ID:")[1].strip()
            break

    bob_mem_raw = await mcp_call("memorymcp", bob_key, "upsertMemory", {
        "text": "e35test bob-only memory about deployment pipeline",
        "memory_type": "concept", "source": "e35test",
    })
    bob_mem_id = ""
    for line in bob_mem_raw.splitlines():
        if "ID:" in line:
            bob_mem_id = line.split("ID:")[1].strip()
            break

    # alice queries → sees only her own
    alice_q = await mcp_call("memorymcp", alice_key, "queryMemory",
                             {"query": "e35test", "k": 20})
    check("alice sees own memory", "alice-only" in alice_q)
    check("alice does NOT see bob's", "bob-only" not in alice_q)

    bob_q = await mcp_call("memorymcp", bob_key, "queryMemory",
                           {"query": "e35test", "k": 20})
    check("bob sees own memory", "bob-only" in bob_q)
    check("bob does NOT see alice's", "alice-only" not in bob_q)

    # alice tries to delete bob's memory → rejected (default mask + owner check)
    try:
        del_try = await mcp_call("memorymcp", alice_key, "deleteMemory",
                                 {"memory_id": bob_mem_id})
        check("alice cannot delete bob's",
              "not found" in del_try.lower() or "error" in del_try.lower())
    except Exception:
        check("alice cannot delete bob's", True)  # rejected = correct

    # ── databasemcp: preset grants ─────────────────────────────────────
    print("\n-- databasemcp preset grants --")
    alice_p = await mcp_call("databasemcp", alice_key, "connect_preset",
                             {"preset": "02"})
    check("alice preset 02 (granted)", "connected" in alice_p.lower()
          or "already" in alice_p.lower())

    bob_p = await mcp_call("databasemcp", bob_key, "connect_preset",
                           {"preset": "02"})
    check("bob preset 02 denied (no grant)", "not granted" in bob_p.lower()
          or "error" in bob_p.lower())

    # cleanup databasemcp connections
    await mcp_call("databasemcp", admin_key, "disconnect_database",
                   {"name": "02"})

    # ── ragmcp: tool access (both users have ragmcp in servers) ────────
    print("\n-- ragmcp tool access --")
    admin_tools = await mcp_list_tools("ragmcp", admin_key)
    alice_tools = await mcp_list_tools("ragmcp", alice_key)
    check("admin sees ragmcp tools", "search" in admin_tools)
    check("alice sees ragmcp tools", "search" in alice_tools)

    # ── cleanup ────────────────────────────────────────────────────────
    print("\n-- cleanup --")
    if alice_mem_id:
        await mcp_call("memorymcp", admin_key, "deleteMemory",
                       {"memory_id": alice_mem_id})
    if bob_mem_id:
        await mcp_call("memorymcp", admin_key, "deleteMemory",
                       {"memory_id": bob_mem_id})

    for u in ("e35alice", "e35bob", "e35admin"):
        try:
            users_store.delete_user(u)
        except Exception:
            pass

    print("test users removed, test memories deleted")

    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    print(f"\n=== {passed} PASS / {failed} FAIL ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
