"""
Tests for the per-request access log middleware in server_factory.

Guards the debuggability of client connections: every HTTP request hitting
the tool app must leave one INFO line on the "mcp.access" logger — including
the initial POST /mcp handshake (session=NEW), calls on an existing session
(session=<id>), and auth-rejected requests — and MCP_DISABLE_REQUEST_LOGS
must silence the middleware entirely.

Uses Starlette's TestClient against a real FastMCP app with a dummy tool.
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

pytest.importorskip("starlette")
pytest.importorskip("httpx")
pytest.importorskip("fastmcp")


@pytest.fixture
def reqlog_app(monkeypatch):
    """Build a real FastMCP app with the request-log middleware wired."""
    monkeypatch.setenv("REQLOG_TEST_API_KEY", "test-key-123")
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app

    mcp = create_fastmcp_server("reqlog_test")

    @mcp.tool()
    def echo(x: int) -> int:
        return x

    return get_transport_app(mcp)


def _access_records(caplog):
    return [r for r in caplog.records if r.name == "mcp.access"]


def test_initialize_and_reuse_are_logged(reqlog_app, caplog):
    """The handshake logs session=NEW; a call on the issued session logs its ID."""
    from starlette.testclient import TestClient

    caplog.set_level(logging.INFO, logger="mcp.access")
    with TestClient(reqlog_app) as client:
        r = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer test-key-123",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "t", "version": "1"}},
            },
        )
        assert r.status_code == 200
        sid = r.headers.get("mcp-session-id")
        assert sid

        r2 = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer test-key-123",
                "Mcp-Session-Id": sid,
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        assert r2.status_code == 200

    records = _access_records(caplog)
    messages = [r.getMessage() for r in records]
    assert messages, "expected at least one mcp.access record"
    assert all(m.startswith("[reqlog_test]") for m in messages), messages
    assert any("POST /mcp" in m and "session=NEW" in m and "-> 200" in m
               for m in messages), messages
    assert any(f"session={sid[:8]}" in m for m in messages), messages
    assert any(" in " in m and m.rsplit(" in ", 1)[1].split("ms", 1)[0].isdigit()
               for m in messages), messages


def test_modern_protocol_dialect_is_logged(reqlog_app, caplog):
    """A 2026-07-28 single-exchange request (no session, modern header) logs v=... .

    Modern-revision requests skip the handshake entirely; the access line is
    what makes them visible at all, so it must carry the protocol version.
    """
    from starlette.testclient import TestClient

    caplog.set_level(logging.INFO, logger="mcp.access")
    with TestClient(reqlog_app) as client:
        r = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer test-key-123",
                "MCP-Protocol-Version": "2026-07-28",
                "mcp-method": "tools/call",
                "mcp-name": "echo",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                  "params": {"name": "echo", "arguments": {"x": 1},
                             "_meta": {
                                 "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                                 "io.modelcontextprotocol/clientCapabilities": {},
                             }}},
        )
        assert r.status_code == 200, r.text

    records = _access_records(caplog)
    assert any("v=2026-07-28" in r.getMessage() and "session=NEW" in r.getMessage()
               and "-> 200" in r.getMessage()
               and "tools/call echo" in r.getMessage()
               for r in records), [r.getMessage() for r in records]


def test_tool_level_jsonrpc_error_is_surfaced(reqlog_app, caplog):
    """A failing tool call rides in an HTTP 200 — the line must flag the failure.

    FastMCP 4 reports unknown tools as a successful result with
    ``"isError": true`` (no JSON-RPC error object), so the marker to expect
    here is ``tool_error``.
    """
    from starlette.testclient import TestClient

    caplog.set_level(logging.INFO, logger="mcp.access")
    with TestClient(reqlog_app) as client:
        r = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer test-key-123",
                "MCP-Protocol-Version": "2026-07-28",
                "mcp-method": "tools/call",
                "mcp-name": "no_such_tool",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                  "params": {"name": "no_such_tool", "arguments": {},
                             "_meta": {
                                 "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                                 "io.modelcontextprotocol/clientCapabilities": {},
                             }}},
        )
        assert r.status_code in (200, 400)

    records = _access_records(caplog)
    assert any("tool_error" in r.getMessage() or "rpc_err=" in r.getMessage()
               for r in records), [r.getMessage() for r in records]


def test_auth_rejected_request_is_logged(reqlog_app, caplog):
    """A request with no credentials still leaves an access line with its 401."""
    from starlette.testclient import TestClient

    caplog.set_level(logging.INFO, logger="mcp.access")
    with TestClient(reqlog_app) as client:
        r = client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert r.status_code == 401

    records = _access_records(caplog)
    assert any("POST /mcp" in r.getMessage() and "-> 401" in r.getMessage()
               for r in records), [r.getMessage() for r in records]


def test_request_logs_can_be_disabled(monkeypatch, caplog):
    """MCP_DISABLE_REQUEST_LOGS suppresses the middleware entirely."""
    monkeypatch.setenv("REQLOG_DIS_API_KEY", "k")
    monkeypatch.setenv("MCP_DISABLE_REQUEST_LOGS", "1")
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app
    from starlette.testclient import TestClient

    mcp = create_fastmcp_server("reqlog_dis")
    app = get_transport_app(mcp)

    caplog.set_level(logging.INFO, logger="mcp.access")
    with TestClient(app) as client:
        assert client.get("/admin/flush-sessions").status_code in (401, 405)

    assert _access_records(caplog) == []
