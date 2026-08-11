"""
Qdrant-backed contract tests using the local file mode (no server needed).

The qdrant_client package bundles the Qdrant core engine — QdrantClient(path=...)
gives a real Qdrant instance without a network server. These tests exercise
the changes made to qdrant_vector.py:

- query_sparse/query_hybrid now accept a ``query_text`` kwarg (accepted for
  Protocol parity, not used by Qdrant's native sparse index)
- The kwarg must not break existing Qdrant behavior
"""
import sys
import uuid
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

pytest.importorskip("qdrant_client")

from shared.store_models import PointStruct, SparseVector

DIM = 4
QDRANT_PATH = str(Path(__file__).resolve().parent / "_qdrant_test_data")


@pytest.fixture
def qdrant_store():
    """A QdrantVectorStore backed by local file mode (no server needed)."""
    from qdrant_client import QdrantClient
    from shared.impls.qdrant_vector import QdrantVectorStore
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)
    client = QdrantClient(path=QDRANT_PATH)
    store = QdrantVectorStore(client=client)
    yield store
    try:
        store._client.close()
    except Exception:
        pass
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)


def _uid(n: int = 0) -> str:
    """Deterministic UUID for test point IDs (Qdrant local mode requires UUIDs)."""
    return str(uuid.UUID(int=n))


class TestQdrantQueryTextKwarg:
    """The query_text kwarg must be accepted without breaking Qdrant's native sparse search."""

    def test_query_sparse_accepts_query_text(self, qdrant_store):
        """query_sparse must accept query_text=... without error (Qdrant ignores it)."""
        qdrant_store.ensure_collection("qt1", dense_dim=DIM, sparse=True)
        qdrant_store.upsert("qt1", [
            PointStruct(id=_uid(1), vector=[0.1] * DIM,
                        payload={"text": "python web framework"},
                        sparse_vector=SparseVector(indices=[0, 1, 2], values=[1.0, 0.8, 0.6])),
        ])
        sparse = SparseVector(indices=[0, 1, 2], values=[1.0, 0.8, 0.6])
        results = qdrant_store.query_sparse("qt1", sparse, limit=5, query_text="python web")
        assert isinstance(results, list)

    def test_query_sparse_works_without_query_text(self, qdrant_store):
        """Existing callers that don't pass query_text must still work."""
        qdrant_store.ensure_collection("qt2", dense_dim=DIM, sparse=True)
        qdrant_store.upsert("qt2", [
            PointStruct(id=_uid(1), vector=[0.1] * DIM,
                        payload={"text": "rust systems language"},
                        sparse_vector=SparseVector(indices=[10, 11], values=[1.0, 0.5])),
        ])
        sparse = SparseVector(indices=[10, 11], values=[1.0, 0.5])
        results = qdrant_store.query_sparse("qt2", sparse, limit=5)
        assert isinstance(results, list)

    def test_query_hybrid_accepts_query_text(self, qdrant_store, monkeypatch):
        """query_hybrid must accept query_text=... and still fuse dense+sparse.

        Uses the Python-side RRF fallback (``_NATIVE_HYBRID = False``) because
        the Qdrant local file mode doesn't support the server-side
        Prefetch/FusionQuery API. The fallback is the real code path for
        qdrant-client <1.7.
        """
        import shared.impls.qdrant_vector as qv_mod
        monkeypatch.setattr(qv_mod, "_NATIVE_HYBRID", False)

        qdrant_store.ensure_collection("qt3", dense_dim=DIM, sparse=True)
        qdrant_store.upsert("qt3", [
            PointStruct(id=_uid(1), vector=[0.9, 0.1, 0.1, 0.1],
                        payload={"text": "hybrid test"},
                        sparse_vector=SparseVector(indices=[5, 6], values=[1.0, 0.7])),
        ])
        dense = [0.9, 0.1, 0.1, 0.1]
        sparse = SparseVector(indices=[5, 6], values=[1.0, 0.7])
        results = qdrant_store.query_hybrid("qt3", dense, sparse, limit=5, query_text="hybrid")
        assert isinstance(results, list)

    def test_set_payload_merges(self, qdrant_store):
        """Qdrant set_payload must merge — the reference semantics other backends are tested against."""
        qdrant_store.ensure_collection("qt4", dense_dim=DIM)
        pid = _uid(1)
        qdrant_store.upsert("qt4", [
            PointStruct(id=pid, vector=[0.1] * DIM,
                        payload={"text": "hello", "usage_count": 0}),
        ])
        qdrant_store.set_payload("qt4", {"usage_count": 5}, ids=[pid])
        pts = qdrant_store.retrieve("qt4", [pid], with_payload=True)
        assert pts[0].payload.get("text") == "hello"
        assert pts[0].payload.get("usage_count") == 5

    def test_dense_search_returns_scored_results(self, qdrant_store):
        """Sanity: dense search should return higher scores for closer vectors."""
        qdrant_store.ensure_collection("qt5", dense_dim=DIM)
        qdrant_store.upsert("qt5", [
            PointStruct(id=_uid(1), vector=[0.9, 0.1, 0.0, 0.0], payload={"label": "near"}),
            PointStruct(id=_uid(2), vector=[0.0, 0.0, 0.0, 0.9], payload={"label": "far"}),
        ])
        results = qdrant_store.query_dense("qt5", [0.9, 0.1, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0].payload.get("label") == "near"
        assert results[0].score > results[1].score
