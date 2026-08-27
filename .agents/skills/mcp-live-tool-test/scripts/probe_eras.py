#!/usr/bin/env python
"""Compare NEW-era (2026-07-28) and legacy handshake-era clients on a live server.

This is how you simulate a future MCP client against the current stack:
fastmcp's official client in its default mode IS a newest-era client.
The legacy control uses mode="legacy" (the 4.0 upgrade-guide knob).
The round-trip tool is picked per server from sweep_all.CALLS (first entry),
so this works on every server, not just simplemcp.

Usage:
    python .agents/skills/mcp-live-tool-test/scripts/probe_eras.py [server] [--wire]

    server   any tool name with an entry in config/ports.json (default: simplemcp)
    --wire   additionally show raw HTTP behavior: whether initialize issues a
             Mcp-Session-Id, and whether a session-less tools/list is accepted
             (universal probe — works for every server)

Requires fastmcp >= 4 in the interpreter and the launcher running.

Exit codes: 0 both eras OK; 1 an era probe failed; 2 bad usage/config.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sweep_all import CALLS
except ImportError:  # run standalone without sweep_all present
    CALLS = {}


def _era_call(server):
    """(tool, args) to round-trip with, or (None, None) if none is tabled."""
    entries = CALLS.get(server) or []
    return entries[0] if entries else (None, None)


async def era_probe(url, key, mode, server):
    tool, args = _era_call(server)
    kw = {} if mode == "new" else {"mode": "legacy"}
    label = "NEW-era" if mode == "new" else "OLD-era"
    try:
        async with Client(url, auth=BearerAuth(key), **kw) as c:
            n = len(await c.list_tools())
            called = ""
            if tool:
                res = await c.call_tool(tool, args)
                out = res.content[0].text if getattr(res, "content", None) else "?"
                err = " ERROR-FLAG" if getattr(res, "is_error", False) else ""
                called = f" | {tool} -> {' '.join(str(out).split())[:45]!r}{err}"
            print(f"[{label}] tools={n} proto={c.protocol_version}{called}")
            return True
    except Exception as e:
        print(f"[{label}] FAILED: {type(e).__name__}: {e}")
        return False


def _json_of(resp):
    """Parse a JSON or SSE-encoded JSON-RPC response body."""
    if "text/event-stream" in resp.headers.get("content-type", ""):
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                try:
                    return json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": f"unparseable body (HTTP {resp.status_code})"}}


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

        # Session-less tools/list (NOT a hardcoded tool call): works as a
        # statelessness probe on every server. 200 + a real result and no
        # error == accepted; anything else prints why.
        r2 = c.post(url, headers=h, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        body = _json_of(r2)
        accepted = (r2.status_code == 200
                    and isinstance(body.get("result"), dict)
                    and "tools" in body["result"])
        detail = "" if accepted else \
            f" | {str(body.get('error') or body)[:90]}"
        print(f"[wire] session-less tools/list -> {r2.status_code} "
              f"({'accepted — transport is stateless' if accepted else 'rejected — transport is stateful'}"
              f"{detail})")


def main() -> int:
    argv = sys.argv[1:]
    unknown_flags = [a for a in argv if a.startswith("-") and a != "--wire"]
    if unknown_flags:
        print(f"error: unknown flag(s) {unknown_flags} (supported: --wire)", file=sys.stderr)
        return 2
    positional = [a for a in argv if not a.startswith("-")]
    if len(positional) > 1:
        print(f"error: expected at most one server, got {positional}", file=sys.stderr)
        return 2

    try:
        ports = json.loads((ROOT / "config/ports.json").read_text())["assignments"]["mcp"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read config/ports.json — run from the repo checkout "
              f"(skill must live at .agents/skills/mcp-live-tool-test): {e}", file=sys.stderr)
        return 2

    server = positional[0] if positional else "simplemcp"
    if server not in ports:
        print(f"error: unknown server {server!r}; known: {', '.join(sorted(ports))}", file=sys.stderr)
        return 2
    try:
        key = json.loads((ROOT / f"tools/{server}/config.json").read_text())["auth"]["api_key"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: no api_key for {server!r} in tools/{server}/config.json: {e}", file=sys.stderr)
        return 2
    if not _era_call(server)[0]:
        print(f"note: no entry for {server!r} in sweep_all.CALLS — era probes will "
              f"compare tools/list + protocol only, no round-trip call")

    url = f"http://127.0.0.1:{ports[server]}/mcp"
    ok_new = asyncio.run(era_probe(url, key, "new", server))
    ok_old = asyncio.run(era_probe(url, key, "legacy", server))
    if "--wire" in argv:
        wire_probe(url, key)
    return 0 if (ok_new and ok_old) else 1


if __name__ == "__main__":
    sys.exit(main())
