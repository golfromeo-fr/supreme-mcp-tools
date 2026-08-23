#!/usr/bin/env python
"""Compare NEW-era (2026-07-28) and legacy handshake-era clients on a live server.

This is how you simulate a future MCP client against the current stack:
fastmcp's official client in its default mode IS a newest-era client.
The legacy control uses mode="legacy" (the 4.0 upgrade-guide knob).

Usage:
    python .agents/skills/mcp-live-tool-test/scripts/probe_eras.py [server] [--wire]

    server   any tool name with an entry in config/ports.json (default: simplemcp)
    --wire   additionally show raw HTTP behavior: whether initialize issues a
             Mcp-Session-Id and whether session-less tools/call is accepted

Requires fastmcp >= 4 in the interpreter and the launcher running.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

ROOT = Path(__file__).resolve().parents[4]


async def era_probe(url, key, mode, tool, args):
    kw = {} if mode == "new" else {"mode": "legacy"}
    async with Client(url, auth=BearerAuth(key), **kw) as c:
        n = len(await c.list_tools())
        res = await c.call_tool(tool, args)
        out = res.content[0].text if getattr(res, "content", None) else "?"
        label = "NEW-era" if mode == "new" else "OLD-era"
        print(f"[{label}] {tool} -> {' '.join(str(out).split())[:60]!r} "
              f"tools={n} proto={c.protocol_version}")


def wire_probe(url, key):
    import httpx

    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    with httpx.Client(timeout=15) as c:
        r = c.post(url, headers=h, json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2026-07-28", "capabilities": {},
                               "clientInfo": {"name": "wire-probe", "version": "1.0"}}})
        print(f"[wire] initialize -> {r.status_code} | "
              f"Mcp-Session-Id issued: {bool(r.headers.get('mcp-session-id'))}")
        r2 = c.post(url, headers=h, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "square", "arguments": {"value": 9}}})
        stateless = r2.status_code == 200
        print(f"[wire] tools/call WITHOUT session id -> {r2.status_code} "
              f"({'accepted — transport is stateless' if stateless else 'rejected — transport is stateful'})")


def main():
    server = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "simplemcp"
    ports = json.load(open(ROOT / "config/ports.json"))["assignments"]["mcp"]
    key = json.load(open(ROOT / f"tools/{server}/config.json"))["auth"]["api_key"]
    url = f"http://127.0.0.1:{ports[server]}/mcp"
    asyncio.run(era_probe(url, key, "new", "square", {"value": 7}))
    asyncio.run(era_probe(url, key, "legacy", "square", {"value": 6}))
    if "--wire" in sys.argv:
        wire_probe(url, key)


if __name__ == "__main__":
    main()
