"""Host-mono deployment smoke (2026-10-27).

Mono mode = no ``MCP_AUTH_MODE=multi``: one static system key per tool, no
user store, no identity gate. It is the primary deployment shape (a plain
host launcher with a fresh checkout), and every mono bug found so far was
invisible to the multi-mode test suite:

  2026-10-26  ragmcp: every collection denied to the mono operator
  2026-10-26  memorymcp: the legacy ownerless corpus hidden
  2026-10-26  databasemcp: every connect refused ("you are not admin")

This test closes that gap: it builds each tool's PRODUCTION ASGI app (the
same factory the launcher serves), runs the app's lifespan, drives the real
MCP wire protocol over the ASGI transport with the system key, and
exercises the flows that broke — so a mono regression fails the suite
instead of waiting to be found in the field.

Stores are faked (the live turso/PG stores must not be touched); the AUTH
path, the identity resolvers, the grant gates and the tool bodies are the
real code.
"""
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "tools" / "databasemcp",
           PROJECT_ROOT / "tools" / "memorymcp", PROJECT_ROOT / "tools" / "ragmcp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

httpx = pytest.importorskip("httpx")
pytest.importorskip("pytest_asyncio")


# ---------------------------------------------------------------------------
# Helpers: talk MCP over the ASGI transport (no ports, no launcher)
# ---------------------------------------------------------------------------

def _system_key(mcp) -> str:
    """The key the app was actually built with (env-first, then config.json)."""
    tokens = getattr(mcp.auth, "tokens", lambda: {})() or {}
    assert tokens, "the tool app has no auth token map"
    return next(iter(tokens))


def _headers(key: str) -> dict:
    return {"X-API-Key": key, "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}


def _parse(resp) -> dict:
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise AssertionError(f"no SSE data frame in: {resp.text[:400]}")
    return resp.json()


async def _rpc(client, path, key, method, params=None, rpc_id=1) -> dict:
    resp = await client.post(
        path, headers=_headers(key),
        json={"jsonrpc": "2.0", "id": rpc_id, "method": method,
              "params": params or {}})
    return _parse(resp)


async def _call_tool(client, path, key, name, args) -> str:
    """Call a tool and return its TEXT result (the tool contract)."""
    out = await _rpc(client, path, key, "tools/call",
                      {"name": name, "arguments": args})
    if "error" in out:
        raise AssertionError(f"{name} -> protocol error: {out['error']}")
    result = out.get("result", {})
    if result.get("isError"):
        raise AssertionError(f"{name} -> tool error: {result.get('content')}")
    parts = [c.get("text", "") for c in result.get("content", [])
             if c.get("type") == "text"]
    return "\n".join(parts)


async def _tool_names(client, path, key) -> set[str]:
    out = await _rpc(client, path, key, "tools/list", {})
    return {t["name"] for t in out.get("result", {}).get("tools", [])}


@pytest.fixture()
def mono_env(monkeypatch):
    """Force MONO regardless of the repo .env (which sets multi at home)."""
    monkeypatch.setenv("MCP_AUTH_MODE", "mono")
    monkeypatch.setenv("MCP_USERS_STORE_DISABLE", "1")
    monkeypatch.setenv("DB_PRESET_AUTOCONNECT", "0")
    monkeypatch.setenv("MCP_API_KEY", "mono-smoke-key")
    return "mono-smoke-key"


# ---------------------------------------------------------------------------
# databasemcp — the flow that failed live ("you are not admin")
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_databasemcp_mono_connect_and_query(mono_env):
    import databasemcp_fastmcp
    from tools.shared.server_factory import get_transport_app
    from connections import REGISTRY

    REGISTRY._entries.clear()
    REGISTRY._active = None
    app = get_transport_app(databasemcp_fastmcp.mcp)
    key = _system_key(databasemcp_fastmcp.mcp)

    try:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://mono.test",
                                         timeout=30.0) as client:
                # 1) a SECOND raw connection must be allowed in mono
                out = await _call_tool(client, "/mcp-stateless", key,
                                       "connect_database",
                                       {"name": "smoke_one", "db_type": "libsql",
                                        "params": {"url": "file::memory:"}})
                assert "Connected 'smoke_one'" in out, out

                # 2) and it must be usable
                out = await _call_tool(client, "/mcp-stateless", key,
                                       "execute_sql",
                                       {"sql": "CREATE TABLE t (v TEXT)",
                                        "connection": "smoke_one"})
                assert "OK" in out, out

                out = await _call_tool(client, "/mcp-stateless", key, "query",
                                      {"sql": "SELECT 1",
                                       "connection": "smoke_one"})
                assert "1" in out, out

                # 3) the gated metadata tools must work in mono too
                out = await _call_tool(client, "/mcp-stateless", key,
                                       "list_tables",
                                       {"connection": "smoke_one"})
                assert "t" in out, out
    finally:
        REGISTRY._entries.clear()
        REGISTRY._active = None


# ---------------------------------------------------------------------------
# memorymcp — the twin bug (legacy corpus hidden by owner scoping)
# ---------------------------------------------------------------------------

class _MemStore:
    """Minimal in-memory VectorStore for the memorymcp smoke.

    Applies the neutral Filter faithfully (owner equality) — a fake that
    ignored filters would hide the very owner-scoping regression this
    smoke exists to catch.
    """

    def __init__(self):
        self.points: dict[str, object] = {}

    def upsert(self, collection, points):
        for p in points:
            self.points[str(p.id)] = p

    def retrieve(self, collection, ids, *, with_payload=True, with_vectors=False):
        class _P:
            def __init__(self, pid, payload, vector=None):
                self.id, self.payload, self.vector = pid, payload, vector
        return [_P(i, self.points[i].payload) for i in ids if i in self.points]

    @staticmethod
    def _matches(payload, flt) -> bool:
        if flt is None:
            return True
        for cond in (flt.must or []):
            key = cond.key
            match = cond.match
            value = match.value if hasattr(match, "value") else None
            if payload.get(key) != value:
                return False
        return True

    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        out = []
        for p in list(self.points.values()):
            if not self._matches(p.payload, filter):
                continue
            out.append(type("Q", (), {"id": p.id, "payload": p.payload,
                                     "score": 0.9})())
            if len(out) >= limit:
                break
        return out

    def set_payload(self, collection, payload, *, ids):
        for i in ids:
            if i in self.points:
                self.points[i].payload.update(payload)

    def get_collection(self, name):
        class _C:
            points_count = 0
            named_vectors = {"": 1024}
            dim = 1024
            has_sparse = False
            distance = "Cosine"
        return _C()

    def list_collections(self):
        return []


@pytest.mark.asyncio
async def test_memorymcp_mono_upsert_and_query(mono_env, monkeypatch):
    import memory_tools
    import memory_core
    from tools.shared.server_factory import get_transport_app

    store = _MemStore()
    # The LEGACY corpus: a memory with NO owner (pre-E3). Mono must see it —
    # the twin bug scoped the operator to owner="memorymcp" and hid it.
    legacy = type("L", (), {"id": "legacy-1", "vector": [0.2] * 1024})()
    legacy.payload = {
        "id": "legacy-1", "text": "a legacy ownerless memory",
        "memory_type": "concept", "source": "agent_action", "tags": [],
        "usage_count": 1, "created_at": "2026-01-01T00:00:00Z",
        "last_accessed": "2026-01-01T00:00:00Z",
    }
    store.points["legacy-1"] = legacy

    monkeypatch.setattr(memory_tools, "vector_store", store)
    monkeypatch.setattr(memory_tools, "sql_store",
                        type("S", (), {"is_available": False})())
    monkeypatch.setattr(memory_tools, "generate_embedding",
                        lambda text: [0.1] * 1024)
    app = get_transport_app(memory_core.mcp)
    key = _system_key(memory_core.mcp)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://mono.test",
                                     timeout=30.0) as client:
            out = await _call_tool(client, "/mcp-stateless", key, "queryMemory",
                                   {"query": "legacy ownerless", "k": 5})
            # the mono operator must SEE the ownerless legacy memory (the
            # twin bug scoped them to owner="memorymcp" and hid it)
            assert "legacy ownerless" in out, out

            out = await _call_tool(client, "/mcp-stateless", key, "upsertMemory",
                                   {"text": "a fresh mono memory",
                                    "memory_type": "concept"})
            assert not out.startswith("Error"), out


# ---------------------------------------------------------------------------
# ragmcp — the original report (every collection denied)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ragmcp_mono_search_not_denied(mono_env, monkeypatch):
    import ragmcp_fastmcp
    from tools.shared.server_factory import get_transport_app

    class _Coll:
        points_count = 0
        named_vectors = {"dense": 1024}
        dim = 1024
        has_sparse = False
        distance = "Cosine"

    class _RagStore:
        def list_collections(self):
            return ["code_index"]

        def get_collection(self, name):
            return _Coll()

        def query_dense(self, *a, **kw):
            return []

        def query_sparse(self, *a, **kw):
            return []

    monkeypatch.setattr(ragmcp_fastmcp, "vector_store", _RagStore())
    app = get_transport_app(ragmcp_fastmcp.mcp)
    key = _system_key(ragmcp_fastmcp.mcp)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://mono.test",
                                     timeout=30.0) as client:
            # the tool must be VISIBLE (mono has no identity gate) and callable
            names = await _tool_names(client, "/mcp-stateless", key)
            assert "search" in names, sorted(names)

            out = await _call_tool(client, "/mcp-stateless", key, "search",
                                   {"query": "anything",
                                    "collection_name": "code_index",
                                    "mode": "dense"})
            assert "not granted to your account" not in out, out
