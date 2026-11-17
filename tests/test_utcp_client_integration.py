"""UTCP client integration tests — the real reference client (utcp 1.2.0)
against a real HTTP server serving the multi-transport app.

Unlike test_utcp_endpoint.py (Starlette TestClient — synthetic HTTP), this
boots uvicorn on a real port and drives the endpoint with the official
Python UTCP client, i.e. the same path real users take:

- manual registration from GET /utcp (auth template + ${VAR} placeholder)
- discovery: search_tools listing + word-match search
- calls: wrapped {"body": ...} args, str and dict results
- honest error surfacing (validation error → client-side exception)
- bad key → manual is not registered (or create() fails loudly)

Config written to disk mirrors scripts/setup_utcp_client.sh's generated
shape: manual_call_templates + dotenv load_variables_from with BOTH the
plain and the provider-namespaced variable names.

Requires: utcp, utcp-http, uvicorn, fastmcp — all project dependencies
(utcp pins live in requirements.txt; servers never import it, client-only).
"""
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("uvicorn")
pytest.importorskip("httpx")
pytest.importorskip("fastmcp")
pytest.importorskip("utcp")
pytest.importorskip("utcp_http")

from utcp.utcp_client import UtcpClient

API_KEY = "utcpx-real-key-1"
# NOTE: underscore-free on purpose — the reference client doubles underscores
# when deriving variable scopes (utcpx_probe would resolve utcpx__probe_...)
SERVER_NAME = "utcpxprobe"
VAR = "UTCP_UTCPXPROBE_API_KEY"  # UTCP_<NAME>_API_KEY, upper-cased


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


@pytest.fixture(scope="module")
def server_url():
    """Production multi-transport app with the UTCP dialect on a real port."""
    import uvicorn
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app

    port = _free_port()
    mcp = create_fastmcp_server(SERVER_NAME, api_key=API_KEY)

    @mcp.tool()
    def ping() -> str:
        return "pong"

    @mcp.tool()
    def echo(message: str) -> str:
        return f"echo: {message}"

    @mcp.tool()
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    app = get_transport_app(mcp)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_for_server(port), f"Server did not start on port {port}"

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def _write_client_config(tmp_path: Path, base_url: str, api_key: str) -> str:
    """Client config + dotenv env mirroring setup_utcp_client.sh's output."""
    env = tmp_path / "utcp-client.env"
    env.write_text(
        f"{VAR}={api_key}\n"
        # clients resolve manual placeholders under a <manual>_ scope too
        f"{SERVER_NAME}_{VAR}={api_key}\n"
    )
    cfg = {
        "load_variables_from": [
            {"variable_loader_type": "dotenv", "env_file_path": str(env)},
        ],
        "manual_call_templates": [
            {
                "name": SERVER_NAME,
                "call_template_type": "http",
                "http_method": "GET",
                "url": f"{base_url}/utcp",
                "content_type": "application/json",
                "auth": {
                    "auth_type": "api_key",
                    "api_key": "${" + VAR + "}",
                    "var_name": "X-API-Key",
                    "location": "header",
                },
            },
        ],
        "tool_repository": {"tool_repository_type": "in_memory"},
        "tool_search_strategy": {
            "tool_search_strategy_type": "tag_and_description_word_match",
        },
    }
    path = tmp_path / "utcp-client.json"
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)


async def _all_tools(client) -> set:
    tools = await client.search_tools("", 200)
    return {t.name for t in tools}


def test_real_client_discovery_calls_and_error(server_url, tmp_path):
    """Register the manual, discover tools, call them, surface errors."""
    cfg_path = _write_client_config(tmp_path, server_url, API_KEY)

    async def scenario():
        client = await UtcpClient.create(config=cfg_path)
        try:
            # discovery: all three probe tools registered, namespaced
            names = {t for t in await _all_tools(client) if t.startswith(SERVER_NAME + ".")}
            assert names == {
                f"{SERVER_NAME}.ping",
                f"{SERVER_NAME}.echo",
                f"{SERVER_NAME}.add",
            }, names

            # word-match search finds echo
            hits = await client.search_tools("echo", 10)
            assert f"{SERVER_NAME}.echo" in {t.name for t in hits}

            # calls: str result, dict result (structured content), zero-arg
            r = await client.call_tool(f"{SERVER_NAME}.echo",
                                       {"body": {"message": "hi"}})
            assert r["result"] == "echo: hi"
            r = await client.call_tool(f"{SERVER_NAME}.add",
                                       {"body": {"a": 2, "b": 3}})
            assert r == {"sum": 5} or r == {"result": {"sum": 5}}
            r = await client.call_tool(f"{SERVER_NAME}.ping", {"body": {}})
            assert r["result"] == "pong"

            # honest error: wrong arg type → 400 surfaces client-side
            with pytest.raises(Exception) as exc:
                await client.call_tool(f"{SERVER_NAME}.echo",
                                       {"body": {"message": 123}})
            assert "400" in str(exc.value)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_real_client_bad_key_not_registered(server_url, tmp_path):
    """A wrong key must not yield a usable tool set (401 → unregistered)."""
    cfg_path = _write_client_config(tmp_path, server_url, "utcpx-wrong-key")

    async def scenario():
        # Client 1.2.0: a 401 yields success=False (not a raise) → create()
        # returns a client with ZERO tools (probe-proven 2026-11-18). Tolerate
        # a future client that raises instead, but ONLY for an auth/variable
        # reason — a config bug must not masquerade as a passing rejection.
        from utcp.exceptions import UtcpVariableNotFound
        try:
            client = await UtcpClient.create(config=cfg_path)
        except UtcpVariableNotFound:
            return
        except Exception as e:
            assert "401" in str(e) or "unauthorized" in str(e).lower(), \
                f"create() failed for a non-auth reason: {e}"
            return
        try:
            assert await _all_tools(client) == set()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_generator_output_drives_real_client_underscored_name(tmp_path):
    """F1+F2 regression: run the REAL setup_utcp_client.sh against a tmp tree
    with an UNDERSCORED tool name, then feed its generated config to the real
    client against a matching ephemeral server.

    Closes two gaps the hand-written-config test above cannot:
    - the generator's provider-namespaced underscore-doubling is exercised
      end-to-end (the six production tools are underscore-free, so this path
      is otherwise never run); reverting the doubling line makes the client
      fail to resolve the scoped variable → this test fails.
    - the generator's ACTUAL output (not a hand-written mirror) is proven
      client-usable, guarding against config-shape drift (e.g. a missing
      auth block on the manual template, a historical generator bug).
    """
    import shutil
    import subprocess

    import uvicorn
    from tools.shared.server_factory import create_fastmcp_server, get_transport_app

    name = "my_probe"
    key = "utcpx-gen-key-1"
    var = "UTCP_MY_PROBE_API_KEY"
    port = _free_port()

    # tmp repo tree; common.sh derives SMC_ROOT from its own copied location,
    # so the generator reads THIS tree's ports.json + tools/<name>/config.json
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ports.json").write_text(
        json.dumps({"assignments": {"mcp": {name: port}}}))
    (tmp_path / "tools" / name).mkdir(parents=True)
    (tmp_path / "tools" / name / "config.json").write_text(
        json.dumps({"auth": {"api_key": key}}))
    (tmp_path / "scripts").mkdir()
    for f in ("setup_utcp_client.sh", "common.sh"):
        shutil.copy(PROJECT_ROOT / "scripts" / f, tmp_path / "scripts" / f)

    cfg_path = tmp_path / "client.json"
    env_path = tmp_path / "client.env"
    subprocess.run(
        ["bash", str(tmp_path / "scripts" / "setup_utcp_client.sh"),
         "--tools", name, "--config", str(cfg_path), "--env", str(env_path)],
        check=True, capture_output=True, text=True)

    # the fix itself: doubled namespaced line + plain line both present
    env_text = env_path.read_text()
    assert f"my__probe_{var}={key}" in env_text, env_text
    assert f"{var}={key}" in env_text, env_text

    # server on the SAME port the generator wrote into the config URL
    mcp = create_fastmcp_server(name, api_key=key)

    @mcp.tool()
    def ping() -> str:
        return "pong"

    @mcp.tool()
    def echo(message: str) -> str:
        return f"echo: {message}"

    app = get_transport_app(mcp)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_for_server(port), f"Server did not start on port {port}"

    try:
        async def scenario():
            client = await UtcpClient.create(config=str(cfg_path))
            try:
                names = {t.name for t in await client.search_tools("", 200)}
                assert {f"{name}.ping", f"{name}.echo"} <= names, names
                r = await client.call_tool(f"{name}.echo",
                                           {"body": {"message": "hi"}})
                assert r["result"] == "echo: hi"
            finally:
                await client.close()

        asyncio.run(scenario())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
