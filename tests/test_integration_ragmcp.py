"""
Integration test for ragmcp search through the tool layer with a Turso backend.

Exercises the full path: MCP tool function → _do_sparse_search →
vector_store.query_sparse(query_text=...) → Turso FTS5 MATCH.

This verifies that query_text flows end-to-end from the search tool down to the
store layer, and that lexical search actually returns hits on Turso (the M5 fix).
No embedding API or Qdrant server needed — uses in-memory libSQL + mode="sparse".
"""
import sys
import asyncio
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")

DUMMY_VEC = [0.0, 0.0, 0.0, 0.0]
COLL = "test-code-search"


@pytest.fixture
def seeded_store():
    """A TursoVectorStore with code chunks indexed for FTS5 search."""
    from shared.impls.turso_vector import TursoVectorStore
    from shared.store_models import PointStruct

    store = TursoVectorStore(url="file::memory:")
    store.ensure_collection(COLL, dense_dim=4, sparse=True)
    store.upsert(COLL, [
        PointStruct(
            id="c1", vector=DUMMY_VEC,
            payload={
                "filePath": "src/movement.c",
                "startLine": 10, "endLine": 50,
                "fileType": "c",
                "functionName": "get_movement_type",
                "codeChunk": "int get_movement_type(STOMVT *mv) { return mv->type; }",
            },
        ),
        PointStruct(
            id="c2", vector=DUMMY_VEC,
            payload={
                "filePath": "src/orders.c",
                "startLine": 5, "endLine": 30,
                "fileType": "c",
                "functionName": "process_order",
                "codeChunk": "void process_order(ORDER *o) { execute(o); }",
            },
        ),
        PointStruct(
            id="c3", vector=DUMMY_VEC,
            payload={
                "filePath": "src/inventory.c",
                "startLine": 100, "endLine": 150,
                "fileType": "c",
                "functionName": "check_stock",
                "codeChunk": "int check_stock(ITEM *item) { return item->qty > 0; }",
            },
        ),
    ])
    return store


@pytest.fixture
def ragmcp_with_store(seeded_store, monkeypatch):
    """Import ragmcp_fastmcp and inject the test store."""
    import ragmcp_fastmcp
    monkeypatch.setattr(ragmcp_fastmcp, "vector_store", seeded_store)
    return ragmcp_fastmcp


class TestRagmcpSparseSearch:
    """Sparse search through the ragmcp tool layer must find code via FTS5."""

    @pytest.mark.asyncio
    async def test_sparse_search_finds_function(self, ragmcp_with_store):
        """search(mode='sparse') must find the right code chunk by keyword."""
        result = await ragmcp_with_store.search(
            query="movement_type",
            limit=5,
            collection_name=COLL,
            mode="sparse",
        )
        assert "get_movement_type" in result
        assert "src/movement.c" in result

    @pytest.mark.asyncio
    async def test_sparse_search_finds_different_function(self, ragmcp_with_store):
        """A different query must find a different chunk."""
        result = await ragmcp_with_store.search(
            query="process_order",
            limit=5,
            collection_name=COLL,
            mode="sparse",
        )
        assert "process_order" in result
        assert "src/orders.c" in result

    @pytest.mark.asyncio
    async def test_sparse_search_no_match_returns_empty(self, ragmcp_with_store):
        """A query with no matching tokens must return an empty result."""
        result = await ragmcp_with_store.search(
            query="nonexistent_xyzzy_42",
            limit=5,
            collection_name=COLL,
            mode="sparse",
        )
        # Should not contain any of the indexed function names
        assert "get_movement_type" not in result
        assert "process_order" not in result
        assert "check_stock" not in result

    @pytest.mark.asyncio
    async def test_query_text_flows_to_store(self, ragmcp_with_store, seeded_store, monkeypatch):
        """Verify query_text is actually passed to vector_store.query_sparse."""
        captured = {}
        original = seeded_store.query_sparse

        def spy(collection, sparse, *, limit=10, filter=None, query_text=None):
            captured["query_text"] = query_text
            captured["collection"] = collection
            return original(collection, sparse, limit=limit, filter=filter, query_text=query_text)

        monkeypatch.setattr(seeded_store, "query_sparse", spy)

        await ragmcp_with_store.search(
            query="check_stock",
            limit=5,
            collection_name=COLL,
            mode="sparse",
        )
        assert captured.get("query_text") == "check_stock"
        assert captured.get("collection") == COLL

    @pytest.mark.asyncio
    async def test_file_type_filter(self, ragmcp_with_store):
        """file_type filter must narrow results to matching files."""
        result = await ragmcp_with_store.search(
            query="movement_type",
            limit=5,
            collection_name=COLL,
            mode="sparse",
            file_type="c",
        )
        assert "get_movement_type" in result

        # Non-matching file type should return nothing
        result_empty = await ragmcp_with_store.search(
            query="movement_type",
            limit=5,
            collection_name=COLL,
            mode="sparse",
            file_type="py",
        )
        assert "get_movement_type" not in result_empty
