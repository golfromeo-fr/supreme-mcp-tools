#!/usr/bin/env python
"""Live sweep of every running MCP tool via a real fastmcp client.

Exercises the full path (client -> auth -> session -> server -> tool) against
a launcher you started yourself. Safe reads only; mutations are skipped by
design (see SKILL.md in this directory).

Usage:
    python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py
    python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py --only simplemcp,webmcp

Requires: fastmcp >= 4 installed in the interpreter (env_python has 4.0.0b3)
and the launcher running.

Exit codes:
    0  every swept function behaved (UPSTREAM rows are allowed and tallied,
       never silent)
    1  at least one unexpected failure
    2  nothing was swept (no server reachable, or --only matched nothing) --
       a no-op must never read as all-clear
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

ROOT = Path(__file__).resolve().parents[4]

# brave_search_web NOT SWEPT — user directive 2026-08-27: the function is not
# meant to be used (it can't be hidden server-side); never call it in sweeps.
# fetch_url asserts clean decode via classify() (decode bug fixed 2026-08-27).
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

# Errors matching these are the server's *dependencies* failing (search quota,
# Oracle not configured, DB down) — not defects in this repo's code. They are
# still printed per-row and tallied; anything unmatched shows as FAIL, which
# is the safe direction (false alarm beats hidden failure).
UPSTREAM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b429\b",
        r"\bORA-\d{5}\b",
        r"\bUSERID\b",
        r"oracle.*(not configured|credentials)",
        r"connection (refused|reset|timed out|closed)",
        r"errno 111",
    )
]


def is_upstream(msg: str) -> bool:
    return any(p.search(msg) for p in UPSTREAM_PATTERNS)


def brief(text, n=100):
    t = " ".join(str(text).split())
    return t[:n] + ("…" if len(t) > n else "")


def port_open(port: float) -> bool:
    import socket

    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", int(port))) == 0


def classify(server, fn, out):
    """fetch_url doubles as the Content-Encoding regression probe (bug fixed
    2026-08-27): mojibake or missing body text means the decode bug is back."""
    if server == "webmcp" and fn == "fetch_url":
        if "Example Domain" in out and "\ufffd" not in out:
            return "PASS"
        return "FAIL"
    return "PASS"


async def sweep(server, port):
    cfg = json.loads((ROOT / f"tools/{server}/config.json").read_text())
    key = cfg["auth"]["api_key"]
    url = f"http://127.0.0.1:{port}/mcp"
    rows, tools_n, proto, err = [], None, None, None
    try:
        async with Client(url, auth=BearerAuth(key)) as c:
            tools_n = len(await c.list_tools())
            proto = c.protocol_version
            for fn, args in CALLS[server]:
                try:
                    res = await c.call_tool(fn, args)
                    out = res.content[0].text if getattr(res, "content", None) else ""
                    # Seatbelt: call_tool raises on tool errors by default
                    # (raise_on_error=True), but don't let a non-raising
                    # error result ever read as PASS; it gets the same
                    # upstream tolerance as the exception path below.
                    if getattr(res, "is_error", False):
                        if not out:
                            out = "(error result, empty content)"
                        status = "UPSTREAM" if is_upstream(out) else "FAIL"
                    else:
                        status = classify(server, fn, out)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    out = msg
                    status = "UPSTREAM" if is_upstream(msg) else "FAIL"
                rows.append((fn, json.dumps(args), status, brief(out)))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return tools_n, proto, rows, err


async def main() -> int:
    argv = sys.argv[1:]
    unknown_flags = [a for a in argv if a.startswith("-") and a != "--only"]
    if unknown_flags:
        print(f"error: unknown flag(s) {unknown_flags} (supported: --only a,b)", file=sys.stderr)
        return 2
    if "--only" in argv:
        idx = argv.index("--only")
        if idx + 1 >= len(argv):
            print("error: --only needs a comma-separated list", file=sys.stderr)
            return 2
        only = argv[idx + 1].split(",")
    else:
        only = None

    try:
        ports = json.loads((ROOT / "config/ports.json").read_text())["assignments"]["mcp"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read repo config (run from a repo checkout where "
              f".agents/skills/mcp-live-tool-test lives in place): {e}", file=sys.stderr)
        return 2

    running = {s for s, p in ports.items() if port_open(p)}
    servers = [s for s in ports if s in CALLS and (not only or s in only)]
    unswept_running = sorted(running - set(servers))
    if unswept_running:
        print(f"note: running but NOT swept (add to CALLS or pass --only): "
              f"{', '.join(unswept_running)}")

    reachable, unexpected, upstream_n = 0, 0, 0
    for server in servers:
        port = int(ports[server])
        print(f"\n=== {server} :{port} ===")
        if server not in running:
            print("  not running — skipped")
            continue
        n, proto, rows, err = await sweep(server, port)
        reachable += 1
        if err:
            print(f"CONNECT FAIL: {err}")
            unexpected += 1
            continue
        print(f"tools listed: {n} | negotiated: {proto}")
        for fn, args, status, detail in rows:
            print(f"  [{status:>12}] {fn}{args} -> {detail}")
            if status == "FAIL":
                unexpected += 1
            elif status == "UPSTREAM":
                upstream_n += 1

    if reachable == 0:
        print("\nNOTHING WAS SWEPT — no target server reachable."
              + (f" (--only {only} matched nothing running)" if only else ""))
        return 2
    print(f"\nswept {reachable} server(s): {unexpected} unexpected failure(s), "
          f"{upstream_n} upstream-tolerated")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
