"""
Tests for the session-flush endpoint and idle-TTL wiring in server_factory.

These guard the recovery path for "Session not found" after a server restart:
- POST /admin/flush-sessions terminates all in-memory sessions, so stale client
  Mcp-Session-Id values get HTTP 404 and the client re-initializes.
- MCP_SESSION_IDLE_TIMEOUT applies a session idle TTL so sessions self-evict.

Uses Starlette's TestClient against a real FastMCP app with a dummy tool.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

# Starlette TestClient requires httpx, which is a project dependency.
pytest.importorskip("starlette")
pytest.importorskip("httpx")
pytest.importorskip("fastmcp")


@pytest.fixture
def flush_app(monkeypatch):
    """Build a real FastMCP app with the flush endpoint + TTL wired."""
    monkeypatch.setenv("FLUSHPROBE_TEST_API_KEY", "test-key-123")
    monkeypatch.setenv("MCP_SESSION_IDLE_TIMEOUT", "600")
    # Force a deterministic key via the <NAME>_API_KEY env path.
    monkeypatch.setenv("FLUSHPROBE_TEST_API_KEY", "test-key-123")
    # create_fastmcp_server resolves <NAME upper>_API_KEY.
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app, _get_session_manager

    mcp = create_fastmcp_server("flushprobe_test")

    @mcp.tool()
    def echo(x: int) -> int:
        return x

    app = get_transport_app(mcp)
    return app, _get_session_manager


def _init(client, key="test-key-123"):
    """Run an MCP initialize and return the issued Mcp-Session-Id."""
    r = client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}},
        },
    )
    assert r.status_code == 200, r.text
    return r.headers.get("mcp-session-id")


def test_idle_timeout_applied(flush_app):
    """MCP_SESSION_IDLE_TIMEOUT is set on the session manager after startup."""
    from starlette.testclient import TestClient
    app, get_sm = flush_app
    with TestClient(app) as client:
        sm = get_sm(app)
        assert sm is not None
        assert getattr(sm, "session_idle_timeout", None) == 600.0


def test_flush_route_requires_auth(flush_app):
    """The flush endpoint is protected by the same tool API key as /mcp."""
    from starlette.testclient import TestClient
    app, _ = flush_app
    with TestClient(app) as client:
        assert client.post("/admin/flush-sessions").status_code == 401
        assert client.post(
            "/admin/flush-sessions", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401


def test_flush_terminates_sessions(flush_app):
    """After flush, tracked sessions drop to 0 and a stale ID gets 404."""
    from starlette.testclient import TestClient
    app, get_sm = flush_app
    right = {"Authorization": "Bearer test-key-123"}
    with TestClient(app) as client:
        sm = get_sm(app)
        # Empty flush
        assert client.post("/admin/flush-sessions", headers=right).json() == {"flushed": 0}

        # Initialize a real session
        sid = _init(client)
        assert sid is not None
        assert len(getattr(sm, "_server_instances", {})) == 1

        # Flush it
        assert client.post("/admin/flush-sessions", headers=right).json() == {"flushed": 1}
        assert len(getattr(sm, "_server_instances", {})) == 0

        # The stale session ID is now rejected — the signal that forces a
        # compliant client to re-initialize.
        r = client.post(
            "/mcp",
            headers={**right, "Mcp-Session-Id": sid, "Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        assert r.status_code == 404


def test_flush_can_be_disabled(monkeypatch):
    """MCP_DISABLE_FLUSH_ENDPOINT suppresses the route but keeps the TTL."""
    monkeypatch.setenv("FLUSHPROBE_DIS_API_KEY", "k")
    monkeypatch.setenv("MCP_DISABLE_FLUSH_ENDPOINT", "1")
    monkeypatch.setenv("MCP_SESSION_IDLE_TIMEOUT", "300")
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app
    from starlette.testclient import TestClient

    mcp = create_fastmcp_server("flushprobe_dis")
    app = get_transport_app(mcp)
    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/admin/flush-sessions" not in paths
    assert "/mcp" in paths
