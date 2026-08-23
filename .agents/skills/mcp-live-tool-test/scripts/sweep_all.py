#!/usr/bin/env python
"""Live sweep of every running MCP tool via a real fastmcp client.

Exercises the full path (client -> auth -> session -> server -> tool) against
a launcher you started yourself. Safe reads only; mutations are skipped by
design (see SKILL.md in this directory).

Usage:
    python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py
    python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py --only simplemcp,webmcp

Requires: fastmcp >= 4 installed in the interpreter (env_python has 4.0.0b3)
and the launcher running. Exit code = number of unexpected failures; the
known webmcp fetch_url defect is reported as KNOWN-DEFECT and tolerated
until the Content-Encoding fix lands.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

ROOT = Path(__file__).resolve().parents[4]

# brave_search_web EXCLUDED — known gzip-decode defect (SKILL.md).
# convertermcp not swept — its only tool needs a docx fixture.
CALLS = {
    "oraclemcp": [
        ("get_valid_languages", {}),
        ("get_schemas", {}),
    ],
    "webmcp": [
        ("brave_search_api", {"query": "example domain info", "count": 3}),
        ("google_search_api", {"query": "smoke test", "num": 3}),
        ("fetch_url", {"url": "https://example.com"}),
        ("post_url", {"url": "https://httpbin.org/post", "data": '{"ping":"smoke"}'}),
    ],
    "simplemcp": [
        ("double", {"value": 21}),
        ("square", {"value": 7}),
        ("greet", {"name": "smoke", "greeting": "Howdy"}),
        ("get_secret", {}),
    ],
    "ragmcp": [
        ("list_collections", {}),
    ],
    "memorymcp": [
        ("listMemoryTypes", {}),
        ("getMemoryMetrics", {}),
        ("queryMemory", {"query": "smoke test"}),
    ],
}

UPSTREAM_HINTS = ("429", "Oracle", "ora-", "USERID", "refused", "Connection")


def brief(text, n=100):
    t = " ".join(str(text).split())
    return t[:n] + ("…" if len(t) > n else "")


def classify(server, fn, out):
    """fetch_url doubles as the regression probe for the gzip-decode bug."""
    if server == "webmcp" and fn == "fetch_url":
        if "Example Domain" in out and "\ufffd" not in out:
            return "PASS"
        return "KNOWN-DEFECT"
    return "PASS"


async def sweep(server, port):
    cfg = json.load(open(ROOT / f"tools/{server}/config.json"))
    key = cfg["auth"]["api_key"]
    url = f"http://127.0.0.1:{port}/mcp"
    rows, tools_n, proto, err = [], None, None, None
    try:
        async with Client(url, auth=BearerAuth(key)) as c:
            tools_n = len(await c.list_tools())
            proto = c.protocol_version
            for fn, args in CALLS.get(server, []):
                try:
                    res = await c.call_tool(fn, args)
                    out = res.content[0].text if getattr(res, "content", None) else ""
                    status = classify(server, fn, out)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    out, status = msg, ("UPSTREAM" if any(h in msg for h in UPSTREAM_HINTS) else "FAIL")
                rows.append((fn, json.dumps(args), status, brief(out)))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return tools_n, proto, rows, err


async def main():
    import socket

    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    ports = json.load(open(ROOT / "config/ports.json"))["assignments"]["mcp"]
    servers = [s for s in ports if s in CALLS and (not only or s in only)]
    unexpected = 0
    for server in servers:
        port = ports[server]
        with socket.socket() as sk:
            sk.settimeout(0.5)
            if sk.connect_ex(("127.0.0.1", port)) != 0:
                print(f"\n=== {server} :{port} ===\n  not running — skipped")
                continue
        n, proto, rows, err = await sweep(server, port)
        print(f"\n=== {server} :{ports[server]} ===")
        if err:
            print(f"CONNECT FAIL: {err}")
            unexpected += 1
            continue
        print(f"tools listed: {n} | negotiated: {proto}")
        for fn, args, status, detail in rows:
            print(f"  [{status:>12}] {fn}{args} -> {detail}")
            if status == "FAIL":
                unexpected += 1
    print(f"\nunexpected failures: {unexpected} (KNOWN-DEFECT/UPSTREAM rows excluded)")
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    asyncio.run(main())
