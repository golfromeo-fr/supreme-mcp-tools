"""
End-to-end integration test for the session flush endpoint.

Unlike test_session_flush.py (which uses Starlette's TestClient against a bare
FastMCP app), this test starts a real uvicorn server on a real port and makes
HTTP requests with httpx. This verifies:

- The flush endpoint is reachable over real HTTP (not just ASGI in-process)
- A stale Mcp-Session-Id gets HTTP 404 after flush (the signal that forces
  compliant clients to re-initialize)
- The client can re-initialize and get a new session after flush
- Auth is enforced on the real server

Requires: uvicorn, httpx, fastmcp — all project dependencies.
"""
import os
import sys
import socket
import time
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

pytest.importorskip("uvicorn")
pytest.importorskip("httpx")
pytest.importorskip("fastmcp")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)
    return False


@pytest.fixture
def server_url(monkeypatch):
    """Start a real FastMCP+uvicorn server on a free port and yield its base URL."""
    import uvicorn
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app

    port = _free_port()
    monkeypatch.setenv("FLUSHPROBE_E2E_API_KEY", "e2e-key-789")
    monkeypatch.setenv("MCP_SESSION_IDLE_TIMEOUT", "600")

    mcp = create_fastmcp_server("flushprobe_e2e")

    @mcp.tool()
    def ping(x: int) -> int:
        return x

    app = get_transport_app(mcp)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run uvicorn in a background thread (uvicorn.Server.serve is async,
    # but Server.run handles the event loop for us)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_for_server(port), f"Server did not start on port {port}"

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def _init_session(client, base_url, key="e2e-key-789"):
    """Run an MCP initialize and return the session ID."""
    r = client.post(
        f"{base_url}/mcp",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json, text/event-stream",
        },
        json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "1"},
            },
        },
    )
    assert r.status_code == 200, f"Initialize failed: {r.status_code} {r.text}"
    return r.headers.get("mcp-session-id")


class TestEndToEndFlush:
    """Full HTTP round-trip: init → flush → 404 → re-init."""

    def test_flush_terminates_session_over_http(self, server_url):
        """A real HTTP flush must terminate the session so the next call 404s."""
        import httpx

        base_url = server_url
        auth = {"Authorization": "Bearer e2e-key-789"}

        with httpx.Client(timeout=10) as client:
            sid = _init_session(client, base_url)
            assert sid is not None, "No session ID returned from initialize"

            r = client.post(f"{base_url}/admin/flush-sessions", headers=auth)
            assert r.status_code == 200
            assert r.json()["flushed"] >= 1

            # The stale session ID must now get 404
            r = client.post(
                f"{base_url}/mcp",
                headers={
                    **auth,
                    "Mcp-Session-Id": sid,
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert r.status_code == 404, \
                f"Expected 404 for stale session, got {r.status_code}"

    def test_reinitialize_after_flush(self, server_url):
        """After flush, the client must be able to re-initialize successfully."""
        import httpx

        base_url = server_url
        auth = {"Authorization": "Bearer e2e-key-789"}

        with httpx.Client(timeout=10) as client:
            sid1 = _init_session(client, base_url)
            client.post(f"{base_url}/admin/flush-sessions", headers=auth)

            sid2 = _init_session(client, base_url)
            assert sid2 is not None
            assert sid2 != sid1, "New session ID must differ from the flushed one"

            r = client.post(
                f"{base_url}/mcp",
                headers={
                    **auth,
                    "Mcp-Session-Id": sid2,
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            )
            assert r.status_code == 200, f"New session failed: {r.status_code}"

    def test_flush_requires_auth(self, server_url):
        """The flush endpoint must reject unauthenticated requests."""
        import httpx

        base_url = server_url

        with httpx.Client(timeout=10) as client:
            assert client.post(f"{base_url}/admin/flush-sessions").status_code == 401
            assert client.post(
                f"{base_url}/admin/flush-sessions",
                headers={"Authorization": "Bearer wrong-key"},
            ).status_code == 401

    def test_flush_empty_returns_zero(self, server_url):
        """Flushing when no sessions exist must return {"flushed": 0}."""
        import httpx

        base_url = server_url
        auth = {"Authorization": "Bearer e2e-key-789"}

        with httpx.Client(timeout=10) as client:
            r = client.post(f"{base_url}/admin/flush-sessions", headers=auth)
            assert r.status_code == 200
            assert r.json() == {"flushed": 0}
