"""E1 — runtime Function-Mask toggling.

Masks must apply to the RUNNING server, not only after a restart:
- apply_mask_at_runtime toggles a live FastMCP instance (disable/enable),
- the per-tool 81xx mgmt server exposes POST /admin/function-masks
  (file-first: persist, then apply),
- server_manager wires the tool's mcp into its registry for both,
- central /api/disabled-tools/... pushes the change after persisting.
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.tool_extensions.registry import ExtensionRegistry  # noqa: E402


@pytest.fixture()
def mask_env(tmp_path, monkeypatch):
    """A probe FastMCP instance + a REDIRECTED tools_config.json (tests must
    never touch the real user config)."""
    import launcher.tools_config as tc
    from tools.shared.server_factory import create_fastmcp_server

    cfg = tmp_path / "tools_config.json"
    monkeypatch.setattr(tc, "_DEFAULT_CONFIG_FILE", cfg)

    mcp = create_fastmcp_server("maskprobe-e1")

    @mcp.tool
    async def double(value: int) -> int:
        return value * 2

    return mcp, cfg, tc


def _list_names(mcp) -> list[str]:
    async def _run():
        from fastmcp import Client

        async with Client(mcp) as client:
            return sorted(t.name for t in await client.list_tools())

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class TestRuntimeHelper:
    def test_disable_then_enable_hides_and_restores(self, mask_env):
        from tools.shared.function_masks import apply_mask_at_runtime

        mcp, _cfg, _tc = mask_env
        assert _list_names(mcp) == ["double"]

        result = apply_mask_at_runtime(mcp, "maskprobe-e1", "double", True)
        assert result == {"applied": True, "reason": None}
        assert _list_names(mcp) == []  # hidden from tools/list immediately

        result = apply_mask_at_runtime(mcp, "maskprobe-e1", "double", False)
        assert result["applied"] is True
        assert _list_names(mcp) == ["double"]  # restored without restart

    def test_masked_call_fails_with_unknown_tool(self, mask_env):
        from tools.shared.function_masks import apply_mask_at_runtime

        mcp, _cfg, _tc = mask_env
        apply_mask_at_runtime(mcp, "maskprobe-e1", "double", True)
        # calling a runtime-masked tool raises (fastmcp native disable)
        with pytest.raises(Exception, match="[Uu]nknown tool|not found"):
            _run_call(mcp, "double", {"value": 2})

    def test_no_instance_reports_not_applied(self):
        from tools.shared.function_masks import apply_mask_at_runtime

        result = apply_mask_at_runtime(None, "srv", "tool", True)
        assert result["applied"] is False
        assert "instance" in result["reason"]


def _run_call(mcp, tool: str, arguments: dict) -> str:
    async def _run():
        from fastmcp import Client

        async with Client(mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.content[0].text if getattr(result, "content", None) else ""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


class TestMgmtEndpoint81xx:
    def test_endpoint_persists_then_applies(self, mask_env):
        from fastapi.testclient import TestClient

        from launcher.tool_extensions.http_server import ExtensionHTTPServer

        mcp, cfg, _tc = mask_env
        registry = ExtensionRegistry()
        registry.mcp_instance = mcp
        server = ExtensionHTTPServer(
            tool_name="maskprobe-e1", registry=registry, port=8199,
            host="127.0.0.1", api_key="probe-key",
        )
        client = TestClient(server.app)

        resp = client.post(
            "/admin/function-masks",
            json={"tool": "double", "masked": True},
            headers={"X-API-Key": "probe-key"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["runtime_applied"] is True, body
        assert body["disabled"] == ["double"]
        # file-first: the persisted config now carries the mask
        assert json.loads(cfg.read_text())["disabled_tools"]["maskprobe-e1"] == ["double"]
        # and the live server hides the tool
        assert _list_names(mcp) == []

        resp = client.post(
            "/admin/function-masks",
            json={"tool": "double", "masked": False},
            headers={"X-API-Key": "probe-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["runtime_applied"] is True
        assert json.loads(cfg.read_text())["disabled_tools"]["maskprobe-e1"] == []
        assert _list_names(mcp) == ["double"]

    def test_endpoint_requires_auth(self, mask_env):
        from fastapi.testclient import TestClient

        from launcher.tool_extensions.http_server import ExtensionHTTPServer

        mcp, _cfg, _tc = mask_env
        registry = ExtensionRegistry()
        registry.mcp_instance = mcp
        server = ExtensionHTTPServer(
            tool_name="maskprobe-e1", registry=registry, port=8199,
            host="127.0.0.1", api_key="probe-key",
        )
        resp = TestClient(server.app).post(
            "/admin/function-masks", json={"tool": "double", "masked": True},
        )
        assert resp.status_code in (401, 403)


class TestCentralPush:
    def test_push_runtime_mask_uses_tool_registry(self, mask_env, monkeypatch):
        import launcher.tool_extensions.registry as reg_mod
        from launcher.management_server import _push_runtime_mask

        mcp, _cfg, _tc = mask_env
        registry = ExtensionRegistry()
        registry.mcp_instance = mcp
        monkeypatch.setitem(reg_mod._global_registries, "maskprobe-e1", registry)

        result = asyncio.run(_push_runtime_mask("maskprobe-e1", "double", True))
        assert result["runtime_applied"] is True
        assert _list_names(mcp) == []

    def test_push_without_registry_reports_not_applied(self, monkeypatch):
        import launcher.tool_extensions.registry as reg_mod
        from launcher.management_server import _push_runtime_mask

        monkeypatch.setitem(reg_mod._global_registries, "ghost-srv", None)
        result = asyncio.run(_push_runtime_mask("ghost-srv", "double", True))
        assert result["runtime_applied"] is False


class TestWiring:
    def test_server_manager_wires_mcp_instance(self):
        from launcher.server_manager import _wire_mcp_instance
        from launcher.tool_extensions.registry import ExtensionRegistry

        registry = ExtensionRegistry()
        assert registry.mcp_instance is None  # default

        _wire_mcp_instance(registry, types.SimpleNamespace(mcp="FAKE_MCP"))
        assert registry.mcp_instance == "FAKE_MCP"

        _wire_mcp_instance(registry, object())  # module without mcp → None, no raise
        assert registry.mcp_instance is None


class TestBootSyncRegression:
    def test_cleanup_keeps_masks_for_existing_tools(self, mask_env):
        """launchmcp rewrites the `tools` section at boot and runs
        validate_and_cleanup_config — it must prune only masks whose tool no
        longer exists, never the live masks (E1 restart-stability)."""
        from launcher.tools_config import validate_and_cleanup_config, load_tools_config

        _mcp, cfg, _tc = mask_env
        cfg.write_text(json.dumps({
            "tools": {"webmcp": ["brave_search_web", "fetch_url"]},
            "disabled_tools": {
                "webmcp": ["brave_search_web", "ghost_tool"],  # one valid, one stale
            },
            "version": 1,
        }))
        results = validate_and_cleanup_config(cfg)
        assert results["removed_invalid"] == ["ghost_tool"]
        assert load_tools_config(cfg)["disabled_tools"]["webmcp"] == ["brave_search_web"]
