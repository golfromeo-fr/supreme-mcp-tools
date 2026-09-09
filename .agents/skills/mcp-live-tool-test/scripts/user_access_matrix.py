#!/usr/bin/env python3
"""E3 — MCP client access matrix per user ("mimics an MCP client").

For every ENABLED user in the store, attempt an authenticated tools/list
against every configured MCP server and print the resulting access matrix,
plus an optional tools/call probe. This is exactly what a real MCP client
with that user's key would see — server-side identity gating included.

Usage:
  python scripts/user_access_matrix.py                     # all users, all servers
  python scripts/user_access_matrix.py --users admin,tester
  python scripts/user_access_matrix.py --call double       # also probe one tool

Exit code = number of unexpected failures (connection errors are expected
when a tool is down and are shown as DOWN).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fastmcp import Client  # noqa: E402
from fastmcp.client.auth import BearerAuth  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _servers() -> dict[str, tuple[str, str]]:
    """{server: (url, system_key)} from ports.json + tool configs."""
    ports = json.loads((ROOT / "config" / "ports.json").read_text())
    out = {}
    for name, port in ports.get("assignments", {}).get("mcp", {}).items():
        cfg = ROOT / "tools" / name / "config.json"
        try:
            key = json.loads(cfg.read_text())["auth"]["api_key"]
        except Exception:
            key = ""
        out[name] = (f"http://127.0.0.1:{port}/mcp", key)
    return out


async def _probe(url: str, key: str) -> tuple[str, list[str]]:
    try:
        async with Client(url, auth=BearerAuth(key)) as client:
            tools = [t.name for t in await client.list_tools()]
            return "OK", sorted(tools)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_token" in msg or "Authentication" in msg:
            return "DENIED", []
        if "404" in msg:
            return "DOWN", []
        return f"ERR:{type(e).__name__}", []


async def main(users: list[str], call_tool: str | None) -> int:
    from tools.shared import users_store

    users_store._ensure_seeded()
    servers = _servers()
    failures = 0
    for username in users:
        record = users_store.get_user_record(username)
        if record is None or not record.get("enabled", True):
            print(f"\n### {username}: disabled or missing — skipped")
            continue
        key = record["mcp_key"]
        print(f"\n### {username} (role={record.get('role', 'user')})")
        for server, (url, _syskey) in sorted(servers.items()):
            if server not in (record.get("servers") or []):
                print(f"  {server:<14} ✗ not granted (no key for this server)")
                continue
            status, tools = await _probe(url, key)
            if status == "OK":
                extra = ""
                if call_tool:
                    if call_tool in tools:
                        try:
                            async with Client(url, auth=BearerAuth(key)) as client:
                                r = await client.call_tool(call_tool, {})
                                extra = " | call ✓"
                        except Exception as e:
                            extra = f" | call ✗ ({type(e).__name__})"
                    else:
                        extra = f" | call {call_tool}: ✗ not visible"
                print(f"  {server:<14} ✓ {len(tools)} tool(s): {', '.join(tools) or '(none)'}{extra}")
            elif status == "DENIED":
                print(f"  {server:<14} ✗ DENIED (auth)")
            elif status == "DOWN":
                print(f"  {server:<14} — server down (not counted as failure)")
            else:
                failures += 1
                print(f"  {server:<14} ! {status}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", default="", help="comma list; default = all enabled")
    parser.add_argument("--call", default=None, help="also attempt this tool on visible servers")
    args = parser.parse_args()

    from tools.shared import users_store

    users_store._ensure_seeded()
    usernames = ([u.strip() for u in args.users.split(",") if u.strip()]
                 if args.users else
                 [r["username"] for r in users_store.list_users()])

    failures = asyncio.run(main(usernames, args.call))
    sys.exit(1 if failures else 0)
