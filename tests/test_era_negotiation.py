"""
Era-negotiation smoke test: one FastMCP 4 server serves BOTH protocol eras.

Locks in the "one server, both eras" guarantee described in
plans/mcp-2026-07-28-stateless-upgrade.md and validated by the Phase 0 spike
(plans/mcp-2026-07-28-phase0-verdict.md, Q8/Q9):

1. A modern client (fastmcp.Client default) negotiates the sessionless
   2026-07-28 era and authenticates via Authorization: Bearer.
2. A legacy handshake-era client (fastmcp.Client(mode="legacy"), the knob
   named in the 4.0 upgrade guide) negotiates the older era and ALSO
   authenticates via the same middleware stack (verdict Q9).
3. The negotiated protocolVersion differs between the two clients.
4. X-API-Key-only requests pass through the api_key_fallback normalizer
   (server_factory.py, Phase 2 port of the Phase 0 verdict Q9 fix) and are
   authenticated exactly like Bearer requests, on both eras.

Run: pytest tests/test_era_negotiation.py -v
Requires fastmcp >= 4 (the feature/fastmcp-4 beta pin); skipped otherwise.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import fastmcp
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth

    FASTMCP_MAJOR = int(fastmcp.__version__.split(".")[0])
except Exception:  # pragma: no cover - fastmcp missing entirely
    FASTMCP_MAJOR = 0

pytestmark = pytest.mark.skipif(
    FASTMCP_MAJOR < 4,
    reason=f"era negotiation targets FastMCP 4 symbols; installed fastmcp is {fastmcp.__version__ if FASTMCP_MAJOR else 'missing'}",
)

API_KEY = "era-test-key-42"


def _make_test_server():
    """Minimal FastMCP server with the repo's DualHeaderVerifier auth."""
    from tools.shared.server_factory import DualHeaderVerifier

    verifier = DualHeaderVerifier(
        tokens={API_KEY: {"client_id": "era-client", "scopes": ["mcp"]}},
    )
    mcp = fastmcp.FastMCP("test-era-server", auth=verifier)

    @mcp.tool()
    def echo(message: str) -> str:
        return message

    return mcp


class _ServerHandle:
    """Run a FastMCP ASGI app on a random port in a background thread."""

    def __init__(self, app, host="127.0.0.1", port=0):
        self.app = app
        self.host = host
        self.port = port
        self._thread = None
        self._server = None

    def start(self):
        import uvicorn

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self._server.started:
                break
            time.sleep(0.05)
        for listener in self._server.servers[0].sockets if self._server.servers else []:
            self.port = listener.getsockname()[1]
            break

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/mcp"


@pytest.fixture(scope="module")
def server_url():
    from mcp.types import LATEST_PROTOCOL_VERSION  # noqa: F401 - guards mcp>=2 imports

    from tools.shared.server_factory import get_transport_app

    # Production factory wiring: includes the api_key_fallback normalizer.
    handle = _ServerHandle(get_transport_app(_make_test_server()))
    handle.start()
    yield handle.url
    handle.stop()


def _call_echo_and_version(url, **client_kwargs):
    async def _run():
        async with Client(url, **client_kwargs) as client:
            result = await client.call_tool("echo", {"message": "ping"})
            text = result.content[0].text if getattr(result, "content", None) else None
            return text, client.protocol_version

    # Local event loop, never asyncio.run(): asyncio.run() leaves
    # set_event_loop(None) behind, which makes legacy get_event_loop()
    # helpers in later test files raise under Python 3.13 (prior art:
    # tests/test_memory_autouse.py::_run).
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class TestEraNegotiation:
    """Both eras authenticate and negotiate different protocol versions."""

    def test_modern_client_authenticates_and_calls_tool(self, server_url):
        """Modern client (sessionless era) authenticates via Bearer and calls a tool."""
        from mcp.types import LATEST_PROTOCOL_VERSION

        text, version = _call_echo_and_version(server_url, auth=BearerAuth(API_KEY))
        assert text == "ping"
        assert version == LATEST_PROTOCOL_VERSION

    def test_legacy_client_authenticates_and_calls_tool(self, server_url):
        """Legacy handshake-era client (mode='legacy') shares the same auth stack."""
        text, version = _call_echo_and_version(
            server_url, auth=BearerAuth(API_KEY), mode="legacy"
        )
        assert text == "ping"
        assert version != ""

    def test_negotiated_versions_differ(self, server_url):
        """The two eras must negotiate DIFFERENT protocolVersions on one server."""
        _, modern = _call_echo_and_version(server_url, auth=BearerAuth(API_KEY))
        _, legacy = _call_echo_and_version(
            server_url, auth=BearerAuth(API_KEY), mode="legacy"
        )
        assert modern != legacy

    def test_x_api_key_only_request_accepted(self, server_url):
        """X-API-Key-only request passes like Bearer (C-1 parity, verdict Q9 fix)."""
        r = httpx.post(
            server_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "era-test", "version": "1.0"},
                },
            },
            headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        )
        assert r.status_code not in (401, 403)
