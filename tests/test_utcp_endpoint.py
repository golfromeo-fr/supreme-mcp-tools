"""UTCP endpoint tests — manual shape, plain-HTTP calls, knobs, masks, E3.

Covers the fourth dialect wired by server_factory.get_transport_app:
manual generation (schema + call-template shape, auth placeholder, Host-
derived URLs), the call route (wrapped {"body": ...} and flat bodies,
zero-arg tools, error mapping), the disable knobs, and E3 per-user mask
enforcement (route-level, mirroring IdentityGateMiddleware).
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient

from tools.shared.server_factory import create_fastmcp_server, get_transport_app

API_KEY = "utcp-probe-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _probe_mcp(name="utcpprobe_test", api_key=API_KEY):
    """A probe FastMCP instance with one zero-arg, one str, one dict tool."""
    mcp = create_fastmcp_server(name, api_key=api_key)

    @mcp.tool()
    def ping() -> str:
        return "pong"

    @mcp.tool()
    def echo(message: str) -> str:
        return f"echo: {message}"

    @mcp.tool()
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    return mcp


# ── Manual shape ──────────────────────────────────────────────────────────


def test_manual_shape_and_auth_placeholder():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = client.get("/utcp")
        assert r.status_code == 200
        manual = r.json()
        assert manual["utcp_version"] == "1.0"
        assert manual["manual_version"] == "1.0"
        tools = {t["name"]: t for t in manual["tools"]}
        assert set(tools) == {"ping", "echo", "add"}
        # Host-derived template URL points back at the serving host
        t = tools["echo"]
        assert t["tool_call_template"]["url"] == "http://testserver/utcp/call/echo"
        tpl = t["tool_call_template"]
        assert tpl["http_method"] == "POST"
        assert tpl["content_type"] == "application/json"
        assert tpl["body_field"] == "body"
        auth = tpl["auth"]
        assert auth["auth_type"] == "api_key"
        assert auth["var_name"] == "X-API-Key"
        assert auth["location"] == "header"
        # client-side variable placeholder — the real key never appears
        assert auth["api_key"] == "${UTCP_UTCPPROBE_TEST_API_KEY}"
        assert API_KEY not in r.text
        # inputs wrap the real schema under "body"
        assert t["inputs"]["properties"]["body"]["properties"]["message"] == {"type": "string"}
        assert t["inputs"]["required"] == ["body"]
        # zero-arg tool: no wrapper requirement
        assert "required" not in tools["ping"]["inputs"]


# ── Call route ────────────────────────────────────────────────────────────


def _post(client, tool, payload):
    return client.post(f"/utcp/call/{tool}", json=payload)


def test_call_wrapped_and_flat_bodies():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = _post(client, "echo", {"body": {"message": "hi"}})
        assert r.status_code == 200
        assert r.json() == {"result": "echo: hi"}
        # flat body (hand-rolled clients) is accepted too
        r = _post(client, "echo", {"message": "yo"})
        assert r.status_code == 200
        assert r.json() == {"result": "echo: yo"}



def test_call_dict_result_uses_structured_content():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = _post(client, "add", {"body": {"a": 2, "b": 3}})
        assert r.status_code == 200
        body = r.json()
        # dict-returning tool: structured content path (fastmcp may wrap it
        # under "result" per its output-schema policy)
        assert body == {"sum": 5} or body == {"result": {"sum": 5}}


def test_call_zero_arg_tool_empty_request():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = client.post("/utcp/call/ping")
        assert r.status_code == 200
        assert r.json() == {"result": "pong"}


def test_unknown_tool_404_no_leak():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = _post(client, "definitely_not_here", {"body": {}})
        assert r.status_code == 404
        assert r.json() == {"error": "Unknown tool"}


def test_bad_json_and_bad_body_shapes_400():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        r = client.post("/utcp/call/echo", content=b"not json",
                        headers={**HEADERS, "Content-Type": "application/json"})
        assert r.status_code == 400
        r = _post(client, "echo", {"body": "a string"})
        assert r.status_code == 400


def test_validation_error_maps_400():
    app = get_transport_app(_probe_mcp())
    with TestClient(app, headers=HEADERS) as client:
        # echo requires message: int (str) → fastmcp ValidationError
        r = _post(client, "echo", {"body": {"message": 123}})
        assert r.status_code == 400


def test_unauthenticated_401():
    app = get_transport_app(_probe_mcp())
    with TestClient(app) as client:
        assert client.get("/utcp").status_code == 401
        r = client.post("/utcp/call/echo", json={"body": {"message": "x"}})
        assert r.status_code == 401


# ── Knobs ─────────────────────────────────────────────────────────────────


def test_disable_env_suppresses_routes(monkeypatch):
    monkeypatch.setenv("MCP_DISABLE_UTCP_ENDPOINT", "1")
    app = get_transport_app(_probe_mcp())
    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/utcp" not in paths
    with TestClient(app, headers=HEADERS) as client:
        assert client.get("/utcp").status_code == 404


def test_utcp_enabled_env_truthy_values(monkeypatch):
    from tools.shared import utcp_endpoint as ue

    monkeypatch.delenv("MCP_DISABLE_UTCP_ENDPOINT", raising=False)
    assert ue.utcp_enabled("simplemcp") is True
    for v in ("1", "true", "yes"):
        monkeypatch.setenv("MCP_DISABLE_UTCP_ENDPOINT", v)
        assert ue.utcp_enabled("simplemcp") is False


def test_config_key_false_disables(tmp_path):
    from tools.shared import utcp_endpoint as ue

    tools = tmp_path / "tools"
    (tools / "knobtool").mkdir(parents=True)
    (tools / "knobtool" / "config.json").write_text('{"utcp": false}')
    assert ue.utcp_enabled("knobtool", config_dir=tools) is False
    (tools / "knobtool" / "config.json").write_text('{"utcp": true}')
    assert ue.utcp_enabled("knobtool", config_dir=tools) is True


def test_config_invalid_value_fails_loudly(tmp_path):
    from tools.shared import utcp_endpoint as ue

    tools = tmp_path / "tools"
    (tools / "knobtool").mkdir(parents=True)
    (tools / "knobtool" / "config.json").write_text('{"utcp": "yes"}')
    with pytest.raises(ValueError, match="must be a boolean"):
        ue.utcp_enabled("knobtool", config_dir=tools)


# ── Global masks (E1) ─────────────────────────────────────────────────────


def test_masked_tool_hidden_from_manual_and_call_rejected(tmp_path):
    cfg = tmp_path / "tools_config.json"
    cfg.write_text(json.dumps(
        {"disabled_tools": {"utcpprobe_mask": ["echo"]}}))
    mcp = create_fastmcp_server("utcpprobe_mask", api_key=API_KEY)

    @mcp.tool()
    def echo(message: str) -> str:
        return f"echo: {message}"

    @mcp.tool()
    def ping() -> str:
        return "pong"

    from tools.shared import function_masks as fm

    fm.apply_function_masks(mcp, "utcpprobe_mask", config_path=cfg)
    app = get_transport_app(mcp)
    with TestClient(app, headers=HEADERS) as client:
        names = {t["name"] for t in client.get("/utcp").json()["tools"]}
        assert names == {"ping"}
        r = client.post("/utcp/call/echo", json={"body": {"message": "hi"}})
        assert r.status_code == 404
        assert r.json() == {"error": "Unknown tool"}
        assert client.post("/utcp/call/ping").status_code == 200


# ── E3 multi-user (per-user masks at route level) ─────────────────────────


def test_multi_user_mask_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_AUTH_MODE", "multi")
    store_path = tmp_path / "users.json"
    monkeypatch.setenv("MCP_USERS_STORE", str(store_path))
    from tools.shared import users_store as us

    monkeypatch.setattr(us, "USERS_PATH", store_path)
    monkeypatch.setattr(us, "_store_cache", {"mtime": None, "store": None})
    us.create_user("rootadmin", "password-1", role="admin",
                   servers=["utcpprobe_multi"])
    alice = us.create_user("alice", "password-1", role="user",
                           servers=["utcpprobe_multi"],
                           masked_functions={"utcpprobe_multi": ["echo"]})
    mcp = create_fastmcp_server("utcpprobe_multi", api_key="system-key-multi")

    @mcp.tool()
    def echo(message: str) -> str:
        return f"echo: {message}"

    @mcp.tool()
    def ping() -> str:
        return "pong"

    app = get_transport_app(mcp)
    # system key: admin — sees and calls everything
    with TestClient(app, headers={"Authorization": "Bearer system-key-multi"}) as c:
        names = {t["name"] for t in c.get("/utcp").json()["tools"]}
        assert names == {"echo", "ping"}
        assert c.post("/utcp/call/echo", json={"body": {"message": "hi"}}).status_code == 200
    # alice: user role with echo masked — echo gone from manual, 404 on call
    alice_headers = {"Authorization": f"Bearer {alice['mcp_key']}"}
    with TestClient(app, headers=alice_headers) as c:
        names = {t["name"] for t in c.get("/utcp").json()["tools"]}
        assert names == {"ping"}
        r = c.post("/utcp/call/echo", json={"body": {"message": "hi"}})
        assert r.status_code == 404
        assert r.json() == {"error": "Unknown tool"}
        assert c.post("/utcp/call/ping").status_code == 200
