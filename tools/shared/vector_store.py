"""
VectorStore — abstract interface for vector index backends.

Defines the VectorStore Protocol, a factory (get_vector_store), and a no-op fallback
(NullVectorStore). Concrete impls live in tools/shared/impls/.

Phase 2 of the backend abstraction plan.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, Iterator, Iterable, runtime_checkable

from shared.store_models import (
    PointStruct, ScoredPoint, Filter, CollectionInfo, SparseVector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStore(Protocol):
    """
    Vector index store for ANN query, hybrid search, and payload filtering.

    Implementations: QdrantVectorStore, TursoVectorStore, PostgresVectorStore.
    """

    # lifecycle
    def ensure_collection(
        self, name: str, *,
        dense_dim: int | None = None,
        sparse: bool = False,
        distance: str = "Cosine",
    ) -> None: ...

    # writes
    def upsert(self, collection: str, points: list[PointStruct]) -> None: ...
    def set_payload(self, collection: str, payload: dict, *, ids: list[str]) -> None: ...
    def delete(self, collection: str, *,
               ids: list[str] | None = None,
               filter: Filter | None = None) -> None: ...
    def delete_collection(self, name: str) -> None: ...

    # reads — three named methods, not one with ``using``
    def query_dense(self, collection: str, vec: list[float], *,
                    limit: int = 10, filter: Filter | None = None,
                    using: str | None = None) -> list[ScoredPoint]: ...
    def query_sparse(self, collection: str, sparse: SparseVector, *,
                     limit: int = 10, filter: Filter | None = None) -> list[ScoredPoint]: ...
    def query_hybrid(self, collection: str, dense: list[float],
                     sparse: SparseVector, *, limit: int = 10,
                     filter: Filter | None = None) -> list[ScoredPoint]: ...

    def retrieve(self, collection: str, ids: list[str], *,
                 with_payload: bool = True, with_vectors: bool = False) -> list[PointStruct]: ...

    def scroll(self, collection: str, *, limit: int = 1000, offset=None,
               with_payload: bool = True,
               filter: Filter | None = None,
    ) -> tuple[list[PointStruct], "Any"]: ...

    def get_collection(self, name: str) -> CollectionInfo: ...
    def list_collections(self) -> list[str]: ...

    # migration — MUST stream without materializing full dataset (R1: default no vectors)
    def iter_all(self, collection: str, *, with_vectors: bool = False) -> Iterator[PointStruct]: ...
    def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int: ...


# ---------------------------------------------------------------------------
# No-op fallback
# ---------------------------------------------------------------------------

class NullVectorStore:
    """No-op store for when no vector backend is configured."""

    def ensure_collection(self, name, *, dense_dim=None, sparse=False, distance="Cosine"):
        pass

    def upsert(self, collection, points):
        pass

    def set_payload(self, collection, payload, *, ids):
        pass

    def delete(self, collection, *, ids=None, filter=None):
        pass

    def delete_collection(self, name):
        pass

    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        return []

    def query_sparse(self, collection, sparse, *, limit=10, filter=None):
        return []

    def query_hybrid(self, collection, dense, sparse, *, limit=10, filter=None):
        return []

    def retrieve(self, collection, ids, *, with_payload=True, with_vectors=False):
        return []

    def scroll(self, collection, *, limit=1000, offset=None, with_payload=True, filter=None):
        return [], None

    def get_collection(self, name):
        from shared.store_models import CollectionInfo
        return CollectionInfo(name=name, points_count=0)

    def list_collections(self):
        return []

    def iter_all(self, collection, *, with_vectors=False):
        return
        yield  # type: ignore[unreachable]

    def bulk_upsert(self, collection, points):
        return 0


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Return the process-scoped VectorStore singleton.

    Resolves backend from: config.json storage.vector > env vars > None.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    from shared.store_factory import resolve_vector_backend
    backend = resolve_vector_backend()

    if backend is None:
        _singleton = NullVectorStore()
        return _singleton

    _singleton = make_vector_store(backend["name"], backend)
    return _singleton


def make_vector_store(backend_name: str, config: dict | None = None) -> VectorStore:
    """Create a VectorStore by name. Used by factory and migration tool."""
    config = config or {}

    if backend_name == "qdrant":
        from shared.impls.qdrant_vector import QdrantVectorStore
        return QdrantVectorStore(
            host=config.get("host", "qdrant"),
            port=config.get("port", 6333),
        )

    if backend_name == "turso":
        from shared.impls.turso_vector import TursoVectorStore
        url = config.get("url") or os.getenv("TURSO_DATABASE_URL", "file::memory:")
        auth_token = config.get("auth_token") or os.getenv("TURSO_AUTH_TOKEN")
        return TursoVectorStore(url=url, auth_token=auth_token)

    if backend_name == "postgres":
        from shared.impls.postgres_vector import PostgresVectorStore
        dsn = config.get("dsn")
        if not dsn:
            from shared.store_factory import pg_dsn_from_env
            dsn = pg_dsn_from_env()
        return PostgresVectorStore(dsn=dsn)

    if backend_name == "none":
        return NullVectorStore()

    raise ValueError(f"Unknown vector backend: {backend_name}")
