"""E3/M2+M3 — multi-user integration over the REAL factory path.

MCP_AUTH_MODE=multi + a temp users.json: the factory builds
UserStoreVerifier + IdentityGateMiddleware; alice (user, one server) sees
only her tools; the system key acts as admin (bob-role behavior is covered
at store level in test_users_store.py); attribution lands in mcp.access.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import Client  # noqa: E402
from fastmcp.client.auth import BearerAuth  # noqa: E402


@pytest.fixture()
def multi_env(tmp_path, monkeypatch):
    """Temp users store + mono kill; returns (store module, factory kwargs)."""
    store_path = tmp_path / "users.json"
    monkeypatch.setenv("MCP_USERS_STORE", str(store_path))
    monkeypatch.setenv("MCP_AUTH_MODE", "multi")
    monkeypatch.delenv("MCP_REQUIRE_IDENTITY", raising=False)

    from tools.shared import users_store as us

    monkeypatch.setattr(us, "USERS_PATH", store_path)
    monkeypatch.setattr(us, "_store_cache", {"mtime": None, "store": None})
    return us


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _ServerHandle:
    def __init__(self, app, host="127.0.0.1", port=0):
        import uvicorn

        self.app = app
        config = uvicorn.Config(
            app=app, host=host, port=port, log_level="error"
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
        self._server.should_exit = True
        self._thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"


@pytest.fixture(scope="module")
def live_multi():
    """Factory-built app in multi mode with two identities; one server."""
    os.environ["MCP_AUTH_MODE"] = "multi"  # module-scoped: set/restore here
    from tools.shared.server_factory import create_fastmcp_server

    us_mod = __import__("tools.shared.users_store", fromlist=["users_store"])
    store_path = Path("/tmp") / f"e3_integration_{id(object()):x}.json"
    us_mod.USERS_PATH = store_path
    us_mod._store_cache = {"mtime": None, "store": None}

    admin = us_mod.create_user("rootadmin", "password-1", role="admin",
                               servers=["probe", "other"])
    alice = us_mod.create_user("alice", "password-1", role="user",
                               servers=["probe"],
                               masked_functions={"probe": ["secret_tool"]})

    mcp = create_fastmcp_server("probe", api_key="system-key-probe")

    @mcp.tool
    def ping() -> str:
        return "pong"

    @mcp.tool
    def secret_tool() -> str:
        return "hidden"

    app = __import__("tools.shared.server_factory",
                     fromlist=["get_transport_app"]).get_transport_app(mcp)

    import uvicorn
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8755, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    url = "http://127.0.0.1:8755/mcp"

    yield {
        "url": url,
        "admin_key": admin["mcp_key"],
        "system_key": "system-key-probe",
        "alice_key": alice["mcp_key"],
        "users_store": us_mod,
    }
    server.should_exit = True
    thread.join(timeout=5)
    os.environ.pop("MCP_AUTH_MODE", None)
    store_path.unlink(missing_ok=True)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _list(url, key):
    async with Client(url, auth=BearerAuth(key)) as client:
        try:
            return sorted(t.name for t in await client.list_tools())
        except Exception as e:
            return f"ERROR: {type(e).__name__} {e} data={getattr(e, 'data', None)}"


async def _call(url, key, tool):
    async with Client(url, auth=BearerAuth(key)) as client:
        result = await client.call_tool(tool, {})
        return result.content[0].text if getattr(result, "content", None) else ""


class TestMultiUserIntegration:
    def test_admin_sees_everything_and_calls(self, live_multi):
        names = _run(_list(live_multi["url"], live_multi["admin_key"]))
        assert names == ["ping", "secret_tool"]

    def test_system_key_is_admin_too(self, live_multi):
        names = _run(_list(live_multi["url"], live_multi["system_key"]))
        assert names == ["ping", "secret_tool"]

    def test_user_sees_allowed_server_minus_masks(self, live_multi):
        names = _run(_list(live_multi["url"], live_multi["alice_key"]))
        assert names == ["ping"]  # secret_tool per-user masked

    def test_user_cannot_call_masked_tool(self, live_multi):
        with pytest.raises(Exception, match="[Uu]nknown tool"):
            _run(_call(live_multi["url"], live_multi["alice_key"], "secret_tool"))

    def test_user_can_call_allowed_tool(self, live_multi):
        out = _run(_call(live_multi["url"], live_multi["alice_key"], "ping"))
        assert out == "pong"

    def test_unknown_key_rejected_at_auth(self, live_multi):
        with pytest.raises(Exception):
            _run(_list(live_multi["url"], "bogus-key"))

    def test_hot_disable_user(self, live_multi):
        us_mod = live_multi["users_store"]
        us_mod.set_enabled("alice", False)
        with pytest.raises(Exception):
            _run(_list(live_multi["url"], live_multi["alice_key"]))
        us_mod.set_enabled("alice", True)
        names = _run(_list(live_multi["url"], live_multi["alice_key"]))
        assert "ping" in names
