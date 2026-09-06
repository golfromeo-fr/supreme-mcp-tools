"""P5 — databasemcp wiring: ports, rename completeness, function masks."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


class TestWiring:
    def test_ports_json_assignments(self):
        ports = json.loads((PROJECT_ROOT / "config" / "ports.json").read_text())
        assert ports["assignments"]["mcp"]["databasemcp"] == 8000
        assert ports["assignments"]["mgmt"]["databasemcp"] == 8110

    def test_tracked_configs_have_no_oraclemcp(self):
        # Intentional historical references excluded (rename spec, P1 gate)
        excluded = ("ex-oraclemcp", "as oraclemcp", "former oraclemcp", "oraclemcp-test-key")
        hits = []
        for base in ("config", "launcher"):
            for f in (PROJECT_ROOT / base).rglob("*"):
                if f.is_file() and f.suffix in (".json", ".py") and "__pycache__" not in str(f):
                    text = f.read_text(errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if "oraclemcp" in line and not any(x in line for x in excluded):
                            hits.append(f"{f}:{i}")
        assert hits == [], f"stale oraclemcp references: {hits}"

    def test_tool_dirs_discoverable_via_ports(self):
        # entry module resolves ports from ports.json (same lookup as the launcher)
        import databasemcp_fastmcp as m

        assert m.TOOL_NAME == "databasemcp"
        assert m.MCP_PORT == 8000 and m.MGMT_PORT == 8110

    def test_function_mask_hides_tool(self, tmp_path, monkeypatch):
        from fastmcp import Client

        from tools.shared.function_masks import apply_function_masks
        from tools.shared.server_factory import create_fastmcp_server

        cfg = tmp_path / "tools_config.json"
        cfg.write_text(json.dumps({
            "disabled_tools": {"databasemcp": ["query"]},
            "version": 1,
        }))
        # fresh FastMCP instance to avoid mutating the shared one
        mcp = create_fastmcp_server("databasemcp-maskprobe")

        @mcp.tool
        async def query(sql: str, max_rows: int = 100, connection: str | None = None) -> str:
            return "probe"

        applied = apply_function_masks(mcp, "databasemcp", config_path=cfg)
        assert applied == ["query"]

        async def _list():
            async with Client(mcp) as client:  # in-memory transport, instance-level disable honored
                return [t.name for t in await client.list_tools()]

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            names = loop.run_until_complete(_list())
        finally:
            loop.close()
        assert "query" not in names
