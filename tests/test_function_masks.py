"""D4 — server-side Function Mask enforcement.

Masks (``disabled_tools`` in tools_config.json) must be enforced at the tool
server: a masked tool disappears from ``tools/list`` and direct calls fail
with "Unknown tool" — this is what the user expects ("the MCP client cannot
see the masked function"), verified against fastmcp 4.0.0's native disable
semantics. Before D4 the masks were UI-cosmetic only (zero serving-path
readers — see plans/launcher-persistence-design-2026-09-05.md §P0).
"""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.shared.function_masks import apply_function_masks, masked_tools


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _write_config(tmp_path, disabled: dict) -> Path:
    path = tmp_path / "tools_config.json"
    path.write_text(json.dumps({"disabled_tools": disabled, "version": 1}))
    return path


def _make_mcp():
    from fastmcp import FastMCP

    mcp = FastMCP("maskprobe")

    @mcp.tool
    def visible() -> str:
        return "v"

    @mcp.tool
    def secret_sauce() -> str:
        return "s"

    return mcp


class TestMaskedToolsReader:
    def test_reads_disabled_tools_for_server(self, tmp_path):
        path = _write_config(tmp_path, {"srv": ["tool_a", "tool_b"], "other": ["x"]})
        assert masked_tools("srv", config_path=path) == ["tool_a", "tool_b"]

    def test_missing_file_means_no_masks(self, tmp_path):
        assert masked_tools("srv", config_path=tmp_path / "absent.json") == []

    def test_corrupt_file_means_no_masks(self, tmp_path):
        path = tmp_path / "tools_config.json"
        path.write_text("{not json")
        assert masked_tools("srv", config_path=path) == []

    def test_non_dict_config_means_no_masks(self, tmp_path):
        path = tmp_path / "tools_config.json"
        path.write_text("[1, 2, 3]")
        assert masked_tools("srv", config_path=path) == []

    def test_non_string_entries_dropped(self, tmp_path):
        path = _write_config(tmp_path, {"srv": ["keep", 42, None, ""]})
        assert masked_tools("srv", config_path=path) == ["keep"]


class TestApplyFunctionMasks:
    def test_masked_tool_invisible_and_uncallable(self, tmp_path):
        path = _write_config(tmp_path, {"maskprobe": ["secret_sauce"]})
        mcp = _make_mcp()

        applied = apply_function_masks(mcp, "maskprobe", config_path=path)

        assert applied == ["secret_sauce"]
        listed = [t.name for t in _run(mcp.list_tools())]
        assert "secret_sauce" not in listed
        assert "visible" in listed
        with pytest.raises(Exception, match="[Uu]nknown tool"):
            _run(mcp.call_tool("secret_sauce", {}))

    def test_empty_masks_is_noop(self, tmp_path):
        path = _write_config(tmp_path, {"other": ["x"]})
        mcp = _make_mcp()
        assert apply_function_masks(mcp, "maskprobe", config_path=path) == []
        assert len(_run(mcp.list_tools())) == 2

    def test_ghost_mask_names_tolerated(self, tmp_path):
        path = _write_config(tmp_path, {"maskprobe": ["ghost_tool", "visible"]})
        mcp = _make_mcp()
        assert apply_function_masks(mcp, "maskprobe", config_path=path) == [
            "ghost_tool",
            "visible",
        ]
        assert [t.name for t in _run(mcp.list_tools())] == ["secret_sauce"]


class TestMaskEnforcementOverTransport:
    """The client-facing proof: tools/list over the multi-transport app."""

    def test_client_tools_list_hides_masked_tool(self, tmp_path):
        from fastmcp import Client

        from tools.shared.function_masks import DEFAULT_CONFIG_PATH
        from tools.shared.server_factory import get_transport_app

        path = _write_config(tmp_path, {"maskprobe": ["secret_sauce"]})
        mcp = _make_mcp()
        apply_function_masks(mcp, "maskprobe", config_path=path)

        handle = _ServerHandle(get_transport_app(mcp))
        handle.start()
        try:
            async def _list():
                async with Client(handle.url) as client:
                    return [t.name for t in await client.list_tools()]

            assert "secret_sauce" not in _run(_list())
        finally:
            handle.stop()
        # sanity: the real config file was never touched by this test
        assert not DEFAULT_CONFIG_PATH.exists() or "maskprobe" not in (
            DEFAULT_CONFIG_PATH.read_text()
            if DEFAULT_CONFIG_PATH.exists()
            else ""
        )


class _ServerHandle:
    """Run a FastMCP ASGI app on a random port in a background thread
    (same pattern as tests/test_era_negotiation.py)."""

    def __init__(self, app, host="127.0.0.1", port=0):
        self.app = app
        self.host = host
        self.port = port
        self._thread = None
        self._server = None

    def start(self):
        import uvicorn

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="error")
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
        if self._server.servers:
            self.port = self._server.servers[0].sockets[0].getsockname()[1]

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/mcp"
