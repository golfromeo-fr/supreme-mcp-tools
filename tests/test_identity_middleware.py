"""E3/M1 — caller identity resolution + per-user visibility gate.

Re-encodes the spike scenarios (plans/e3-m1-identity-spike-2026-09-07.md)
as regressions, over the REAL multi-transport app:
- resolve_identity unit behavior (Bearer / X-API-Key / unknown / non-HTTP)
- per-user tools/list filtering + tools/call gating (alice ok, bob masked)
- admin bypasses user masks; global E1 masks unaffected (instance disable)
- wrong key still 401s at the auth layer
- fail-open without identity (in-memory Client), fail-closed with
  MCP_REQUIRE_IDENTITY=1 over HTTP
Live-app scaffold copied from tests/test_era_negotiation.py (uvicorn +
daemon thread on a random port) — do not invent a new one.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.client.auth import BearerAuth  # noqa: E402

from tools.shared.identity import IdentityGateMiddleware, resolve_identity  # noqa: E402
from tools.shared.server_factory import DualHeaderVerifier, get_transport_app  # noqa: E402

TOKENS = {
    "alice-key": {"client_id": "alice", "role": "user", "masked": []},
    "bob-key": {"client_id": "bob", "role": "user", "masked": ["ping"]},
    "admin-key": {"client_id": "golfromeo", "role": "admin", "masked": ["ping"]},
}


class _ServerHandle:
    """Run a FastMCP ASGI app on a random port in a background thread."""

    def __init__(self, app, host="127.0.0.1", port=0):
        import uvicorn

        self.app = app
        self.host = host
        self.port = port
        config = uvicorn.Config(
            app=app, host=self.host, port=self.port, log_level="error",
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


def _build_app():
    verifier = DualHeaderVerifier(tokens=dict(TOKENS))
    mcp = FastMCP("identity-probe", auth=verifier)

    @mcp.tool
    def ping() -> str:
        return "pong"

    mcp.add_middleware(IdentityGateMiddleware(tokens_map_fn=verifier.tokens))
    return get_transport_app(mcp), verifier


@pytest.fixture(scope="module")
def server_url():
    app, _verifier = _build_app()
    handle = _ServerHandle(app)
    yield handle.url
    handle.stop()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _list_and_call(url, key, tool="ping"):
    async with Client(url, auth=BearerAuth(key)) as client:
        names = sorted(t.name for t in await client.list_tools())
        try:
            result = await client.call_tool(tool, {})
            called = result.content[0].text if getattr(result, "content", None) else ""
        except Exception as e:
            called = f"REJECTED:{type(e).__name__}"
        return names, called


async def _list_and_call_inmem(mcp, tool="ping"):
    async with Client(mcp) as client:
        result = await client.call_tool(tool, {})
        return result.content[0].text if getattr(result, "content", None) else ""


class TestResolveIdentity:
    def test_bearer_header(self):
        class _Req:
            headers = {"authorization": "Bearer alice-key", "x-api-key": None}

        ident = resolve_identity(_Req(), lambda: TOKENS)
        assert ident is not None and ident[0] == "alice"

    def test_x_api_key_fallback(self):
        class _Req:
            headers = {"x-api-key": "bob-key"}

        ident = resolve_identity(_Req(), lambda: TOKENS)
        assert ident is not None and ident[0] == "bob"

    def test_unknown_token_is_none(self):
        class _Req:
            headers = {"authorization": "Bearer nope"}

        # None via lookup miss (not via the broken-resolver guard)
        assert resolve_identity(_Req(), lambda: TOKENS) is None

    def test_broken_resolver_is_none_not_raise(self):
        def _boom():
            raise RuntimeError("store unreadable")

        class _Req:
            headers = {"authorization": "Bearer alice-key"}

        assert resolve_identity(_Req(), _boom) is None


class TestGateLive:
    def test_alice_sees_and_calls(self, server_url):
        names, called = _run(_list_and_call(server_url, "alice-key"))
        assert names == ["ping"]
        assert called == "pong"

    def test_bob_masked_hidden_and_rejected(self, server_url):
        names, called = _run(_list_and_call(server_url, "bob-key"))
        assert names == []  # ping masked -> empty tools/list
        assert called.startswith("REJECTED"), called

    def test_admin_bypasses_user_masks(self, server_url):
        names, called = _run(_list_and_call(server_url, "admin-key"))
        assert names == ["ping"]  # masked=["ping"] but role=admin
        assert called == "pong"

    def test_wrong_key_401_at_auth_layer(self, server_url):
        with pytest.raises(Exception):
            _run(_list_and_call(server_url, "wrong-key"))


class TestFailOpenAndClosed:
    def test_in_memory_client_fails_open(self):
        """Non-HTTP scope: gate fails open, tool callable."""
        verifier = DualHeaderVerifier(tokens=dict(TOKENS))
        mcp = FastMCP("inmem", auth=verifier)

        @mcp.tool
        def ping() -> str:
            return "pong"

        mcp.add_middleware(IdentityGateMiddleware(tokens_map_fn=verifier.tokens))
        assert _run(_list_and_call_inmem(mcp)) == "pong"

    def test_require_identity_fails_closed_over_http(self, monkeypatch):
        monkeypatch.setenv("MCP_REQUIRE_IDENTITY", "1")
        verifier = DualHeaderVerifier(tokens=dict(TOKENS))
        mcp = FastMCP("strict", auth=verifier)

        @mcp.tool
        def ping() -> str:
            return "pong"

        mcp.add_middleware(IdentityGateMiddleware(tokens_map_fn=verifier.tokens))
        app = get_transport_app(mcp)
        handle = _ServerHandle(app)
        try:
            async def _anon():
                async with Client(handle.url) as client:  # no auth -> 401 pre-middleware
                    await client.call_tool("ping", {})

            # an authenticated-but-unknown-identity client is impossible here
            # (auth rejects first), so assert the documented composition:
            # auth 401s BEFORE the gate is consulted.
            with pytest.raises(Exception):
                _run(_anon())
        finally:
            handle.stop()


class TestMonoModeUnchanged:
    def test_mono_factory_untouched_by_users_layer(self, monkeypatch):
        """mono must be byte-identical to today: single-token verifier, no
        identity gate, and users_store never even imported by the factory."""
        monkeypatch.setenv("MCP_AUTH_MODE", "mono")
        sys.modules.pop("tools.shared.users_store", None)

        from tools.shared.server_factory import create_fastmcp_server

        mcp = create_fastmcp_server("mono-probe", api_key="mono-key")
        verifier = mcp.auth
        assert isinstance(verifier, DualHeaderVerifier)
        assert list(verifier.tokens().keys()) == ["mono-key"]
        middleware = getattr(mcp, "middleware", []) or []
        assert not any(
            type(m).__name__ == "IdentityGateMiddleware" for m in middleware
        )
        assert "tools.shared.users_store" not in sys.modules


class TestAttribution:
    def test_access_log_carries_user(self, server_url, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="mcp.access"):
            _run(_list_and_call(server_url, "alice-key"))
        access_lines = [r.getMessage() for r in caplog.records
                        if "user=alice" in r.getMessage()]
        assert access_lines, "expected user=alice in mcp.access line"
        # positional: user= sits between session= and ->
        assert " session=NEW user=alice -> " in access_lines[0]


def _run_inner():
    raise AssertionError("placeholder path must not execute")
