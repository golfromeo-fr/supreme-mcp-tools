"""Memory Explorer's MCP client — the UI talks to memorymcp through its MCP
surface only (never the backend directly; E2 principle).

One short-lived fastmcp Client per call: management usage is low-frequency
and this keeps error handling per-call trivial.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parents[1]


def _defaults() -> tuple[str, str]:
    """(base_url, api_key) from ports.json + the tool's config.json."""
    port = 8005
    try:
        ports = json.loads((_PROJECT / "config" / "ports.json").read_text())
        port = ports["assignments"]["mcp"]["memorymcp"]
    except Exception as e:
        logger.warning(f"ports.json unreadable, defaulting memorymcp port: {e}")
    api_key = ""
    try:
        cfg = json.loads((_PROJECT / "tools" / "memorymcp" / "config.json").read_text())
        api_key = cfg["auth"]["api_key"]
    except Exception as e:
        logger.warning(f"memorymcp config.json unreadable: {e}")
    return f"http://127.0.0.1:{port}/mcp", api_key


class MemoryMcpError(RuntimeError):
    pass


class MemoryMcpClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        if base_url is None or api_key is None:
            d_url, d_key = _defaults()
            self.base_url = base_url or d_url
            self.api_key = api_key or d_key
        else:
            self.base_url = base_url
            self.api_key = api_key

    async def _call(self, tool: str, arguments: dict) -> dict | str:
        from fastmcp import Client
        from fastmcp.client.auth import BearerAuth

        try:
            async with Client(self.base_url, auth=BearerAuth(self.api_key)) as client:
                result = await client.call_tool(tool, arguments)
        except Exception as e:
            raise MemoryMcpError(f"memorymcp unreachable at {self.base_url}: {e}") from e
        if getattr(result, "content", None):
            text = result.content[0].text
        else:
            text = ""
        # structured-capable tools (listMemories) return JSON-parseable data
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    async def list_memories(self, limit: int = 20, offset: int = 0,
                            tag: str | None = None) -> dict:
        args: dict = {"limit": limit, "offset": offset}
        if tag:
            args["tag"] = tag
        out = await self._call("listMemories", args)
        return out if isinstance(out, dict) else {"error": str(out)}

    async def query(self, query_text: str, k: int = 10) -> str:
        return await self._call("queryMemory", {"query": query_text, "k": k})

    async def get(self, memory_id: str) -> str:
        return await self._call("getMemory", {"memory_id": memory_id})

    async def audit(self, memory_id: str, limit: int = 20) -> str:
        return await self._call("auditTrail", {"memory_id": memory_id, "limit": limit})

    async def delete(self, memory_id: str) -> str:
        return await self._call("deleteMemory", {"memory_id": memory_id})


_client: MemoryMcpClient | None = None


def get_memory_client() -> MemoryMcpClient:
    global _client
    if _client is None:
        _client = MemoryMcpClient()
    return _client
