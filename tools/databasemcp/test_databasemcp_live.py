#!/usr/bin/env python3
"""Live-server tests for databasemcp (port 8000).

simplemcp-pattern: drives the RUNNING server over the real MCP surface and
skips the whole file (with a reason) when the launcher is down. Uses the
fastmcp Client (same as .agents/skills/mcp-live-tool-test/scripts/sweep_all.py).

Covers the connection-registry lifecycle on a throwaway libSQL file database,
the generic tools, error paths, and — when the running server has them and
.env defines presets — the preset tools and the lazy preset bypass.

Env overrides: DATABASEMCP_URL (default http://127.0.0.1:8000/mcp),
DATABASEMCP_API_KEY (default: from tools/databasemcp/config.json).
"""

import asyncio
import json
import os
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("DATABASEMCP_URL", "http://127.0.0.1:8000/mcp")
_cfg = json.loads((ROOT / "tools" / "databasemcp" / "config.json").read_text())
API_KEY = os.environ.get("DATABASEMCP_API_KEY", _cfg["auth"]["api_key"])


def _probe_target() -> tuple[str, int]:
    parsed = urlparse(BASE_URL)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname or "127.0.0.1", port


_PROBE_HOST, _PROBE_PORT = _probe_target()
_launcher_up: bool | None = None


def _launcher_reachable() -> bool:
    """One-shot TCP probe of the live server, cached for the whole run."""
    global _launcher_up
    if _launcher_up is None:
        try:
            with socket.create_connection((_PROBE_HOST, _PROBE_PORT), timeout=2):
                _launcher_up = True
        except OSError:
            _launcher_up = False
    return _launcher_up


pytestmark = pytest.mark.skipif(
    not _launcher_reachable(),
    reason=f"launcher not running on {_PROBE_HOST}:{_PROBE_PORT}",
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _call(tool: str, arguments: dict) -> str:
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    async with Client(BASE_URL, auth=BearerAuth(API_KEY)) as client:
        result = await client.call_tool(tool, arguments)
        if getattr(result, "content", None):
            return result.content[0].text
        return ""


async def _listed_tools() -> list[str]:
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    async with Client(BASE_URL, auth=BearerAuth(API_KEY)) as client:
        return [t.name for t in await client.list_tools()]


@pytest.fixture(scope="module")
def tool_names():
    return _run(_listed_tools())


@pytest.fixture()
def sweep_conn():
    """A uniquely-named connection to a throwaway libSQL file DB."""
    token = uuid.uuid4().hex[:8]
    name = f"livetest_{token}"
    path = Path("/tmp") / f"databasemcp_livetest_{token}.db"
    out = _run(_call("connect_database", {
        "name": name, "db_type": "libsql", "params": {"url": f"file:{path}"},
    }))
    assert out.startswith(f"Connected '{name}'"), out
    yield name, path
    _run(_call("disconnect_database", {"name": name}))
    path.unlink(missing_ok=True)


class TestToolsList:
    def test_core_tools_present(self, tool_names):
        for name in ("connect_database", "disconnect_database", "list_connections",
                     "query", "execute_sql", "get_schemas", "list_tables", "explain_plan"):
            assert name in tool_names, f"{name} missing from tools/list"

    def test_preset_tools_when_server_has_them(self, tool_names):
        if "list_presets" not in tool_names:
            pytest.skip("running server predates P7 (no preset tools) — restart to activate")
        assert "connect_preset" in tool_names


class TestConnectionLifecycle:
    def test_full_round_trip(self, sweep_conn):
        name, path = sweep_conn
        out = _run(_call("execute_sql", {
            "sql": "CREATE TABLE live_t (id INTEGER PRIMARY KEY, label TEXT)", "connection": name}))
        assert out.startswith("OK."), out
        out = _run(_call("execute_sql", {
            "sql": "INSERT INTO live_t VALUES (1, 'smoke')", "connection": name}))
        assert "Rows affected: 1" in out, out
        out = _run(_call("query", {"sql": "SELECT * FROM live_t", "connection": name}))
        assert "smoke" in out, out

    def test_query_truncation(self, sweep_conn):
        name, _ = sweep_conn
        _run(_call("execute_sql", {"sql": "CREATE TABLE nums (id INTEGER)", "connection": name}))
        for i in range(5):
            _run(_call("execute_sql", {"sql": f"INSERT INTO nums VALUES ({i})", "connection": name}))
        out = _run(_call("query", {"sql": "SELECT * FROM nums", "max_rows": 3, "connection": name}))
        assert "Truncated" in out and "3" in out, out

    def test_get_schemas_cached_second_call(self, sweep_conn):
        name, _ = sweep_conn
        _run(_call("execute_sql", {"sql": "CREATE TABLE sch_t (id INTEGER PRIMARY KEY)", "connection": name}))
        first = _run(_call("get_schemas", {"table_name": "sch_t", "connection": name}))
        assert "cached" not in first and "id" in first, first
        second = _run(_call("get_schemas", {"table_name": "sch_t", "connection": name}))
        assert "(cached)" in second, second

    def test_list_tables_and_explain(self, sweep_conn):
        name, _ = sweep_conn
        _run(_call("execute_sql", {"sql": "CREATE TABLE lt_t (x INTEGER)", "connection": name}))
        tables = _run(_call("list_tables", {"connection": name}))
        assert "lt_t" in tables, tables
        plan = _run(_call("explain_plan", {"sql": "SELECT * FROM lt_t", "connection": name}))
        assert isinstance(plan, str) and plan, plan

    def test_list_connections_shows_active(self, sweep_conn, tool_names):
        name, _ = sweep_conn
        out = _run(_call("list_connections", {}))
        assert name in out, out


class TestErrorPaths:
    def test_query_read_guard(self, sweep_conn):
        name, _ = sweep_conn
        out = _run(_call("query", {"sql": "DELETE FROM lt_t", "connection": name}))
        assert "read-only" in out, out

    def test_query_empty_rejected(self):
        out = _run(_call("query", {"sql": "   "}))
        assert "required" in out, out

    def test_unknown_connection_lists_available(self):
        out = _run(_call("query", {"sql": "SELECT 1", "connection": "no_such_conn"}))
        assert "Unknown connection" in out or "No database connection" in out, out

    def test_connect_bad_params_rejected(self):
        out = _run(_call("connect_database", {"name": "x1", "db_type": "postgres", "params": {"host": "h"}}))
        assert "Missing required params" in out, out


class TestPresets:
    """Live preset tests — only when the running server has P7 AND .env
    defines at least one preset (both checked at runtime)."""

    @pytest.fixture()
    def presets_available(self, tool_names):
        if "list_presets" not in tool_names:
            pytest.skip("running server predates P7 — restart the launcher to activate")
        out = _run(_call("list_presets", {}))
        if out.startswith("none"):
            pytest.skip("no DB_PRESET_<NN> entries in .env")
        return out

    def test_list_presets_masks_passwords(self, presets_available):
        assert "password=" not in presets_available.lower() or "***" in presets_available
        # masked URLs never leak a raw credential segment
        import re
        assert not re.search(r"://[^/\s@:]+:[^@\s*]+@", presets_available), presets_available

    def test_connect_preset_and_query(self, presets_available):
        # first preset line: "- 01 [oracle] ..." — take its number
        first_line = next(l for l in presets_available.splitlines() if l.startswith("- "))
        number = first_line.split()[1]
        out = _run(_call("connect_preset", {"preset": number}))
        # idempotent: a server that already has this preset connected
        # (e.g. the UI panel connected it) must not fail the test
        assert (
            out.startswith("Connected preset")
            or "already connected" in out
            or "already exists" in out
        ), out
        # the bypass: address the preset directly in a generic tool
        out = _run(_call("query", {"sql": "SELECT 1", "connection": number}))
        assert "Unknown connection" not in out and "Connect failed" not in out, out

    def test_bypass_without_connect_preset(self, presets_available):
        # lazy bypass straight from a generic tool
        first_line = next(l for l in presets_available.splitlines() if l.startswith("- "))
        number = first_line.split()[1]
        out = _run(_call("query", {"sql": "SELECT 1", "connection": number}))
        assert "Unknown connection" not in out, out


if __name__ == "__main__":
    sys_exit = pytest.main([__file__, "-v"])
    raise SystemExit(sys_exit)
