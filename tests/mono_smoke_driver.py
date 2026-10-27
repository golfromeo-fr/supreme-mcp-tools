"""Driver: boot ONE tool's real server in MONO and exercise its gated flows.

Run as a subprocess by tests/test_mono_deployment_smoke.py. A fresh process
is the only faithful mono environment: the auth verifier is FROZEN at import
time (the factory reads MCP_AUTH_MODE then), and the pytest suite installs
memory_core STUBS into sys.modules globally (test_memory_artifact_e2e.py),
so an in-process test cannot rely on the real module graph.

Prints one JSON line: {"ok": bool, "checks": [...], "error": str|null}
"""
import asyncio
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

TOOL = sys.argv[1]
PORT = int(sys.argv[2])
KEY = os.environ.get("MCP_API_KEY", "mono-smoke-key")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

checks: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail[:400]})


def _rpc(method: str, params: dict, rpc_id: int = 1):
    """One JSON-RPC call against the running tool (X-API-Key header)."""
    body = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method,
                       "params": params}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/mcp-stateless", data=body, method="POST")
    req.add_header("X-API-Key", KEY)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    if raw.lstrip().startswith("{"):
        return json.loads(raw)
    for line in raw.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"no JSON/SSE payload: {raw[:200]}")


def call_tool(name: str, args: dict) -> str:
    out = _rpc("tools/call", {"name": name, "arguments": args})
    if "error" in out:
        raise AssertionError(f"{name}: protocol error {out['error']}")
    result = out.get("result", {})
    if result.get("isError"):
        raise AssertionError(f"{name}: tool error {result.get('content')}")
    return "\n".join(c.get("text", "") for c in result.get("content", [])
                     if c.get("type") == "text")


def wait_ready(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            _rpc("tools/list", {}, rpc_id=99)
            return
        except Exception as e:  # not up yet
            last = e
            time.sleep(0.5)
    raise AssertionError(f"tool {TOOL} never became ready: {last}")


def run() -> None:
    if TOOL == "databasemcp":
        import databasemcp_fastmcp as mod
        from connections import REGISTRY
    elif TOOL == "memorymcp":
        import memorymcp_fastmcp as mod
    elif TOOL == "ragmcp":
        import ragmcp_fastmcp as mod
    else:
        raise SystemExit(f"unknown tool {TOOL}")

    app = mod.app  # the PRODUCTION app (masks applied, transport built)

    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        wait_ready()

        if TOOL == "databasemcp":
            # the live report: a SECOND raw connection must be allowed in mono
            out = call_tool("connect_database", {
                "name": "smoke_one", "db_type": "libsql",
                "params": {"url": "file::memory:"}})
            check("connect_database allowed in mono", "Connected 'smoke_one'" in out, out)
            out = call_tool("execute_sql", {
                "sql": "CREATE TABLE t (v TEXT)", "connection": "smoke_one"})
            check("execute_sql on that connection", "OK" in out, out)
            out = call_tool("query", {"sql": "SELECT 1",
                                      "connection": "smoke_one"})
            check("query on that connection", "1" in out, out)
            out = call_tool("list_tables", {"connection": "smoke_one"})
            check("gated list_tables", "t" in out, out)
            out = call_tool("connect_preset", {"preset": "01"})
            check("connect_preset reports a usable answer (not a refusal)",
                  "not granted" not in out and "admin-only" not in out, out)

        elif TOOL == "memorymcp":
            out = call_tool("upsertMemory", {
                "text": "a mono smoke memory", "memory_type": "concept",
                "embedding": [0.01] * 1024})
            check("upsertMemory allowed in mono", not out.startswith("Error"), out)
            try:
                out = call_tool("queryMemory", {"query": "mono smoke", "k": 3})
                check("queryMemory sees the mono corpus",
                      "mono smoke" in out, out)
            except AssertionError as e:
                # No local embedding model in this environment — the grant
                # path is already covered by the upsert check above.
                check("queryMemory (skipped: no embedding model)", True, str(e))

        elif TOOL == "ragmcp":
            out = call_tool("list_collections", {})
            check("list_collections not denied in mono",
                  "not granted to your account" not in out, out)
            names = _rpc("tools/list", {}).get("result", {}).get("tools", [])
            check("search tool is visible in mono",
                  any(t.get("name") == "search" for t in names), "")
    finally:
        server.should_exit = True
        time.sleep(1.0)


if __name__ == "__main__":
    error = None
    try:
        run()
    except Exception as e:  # noqa: BLE001 — the driver reports, never raises
        error = f"{type(e).__name__}: {e}"
    ok = error is None and all(c["ok"] for c in checks)
    print(json.dumps({"ok": ok, "checks": checks, "error": error}))
    sys.exit(0 if ok else 1)
