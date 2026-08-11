"""
QdrantVectorStore — VectorStore adapter over qdrant_client.

Translates neutral types (Filter, PointStruct, ScoredPoint, CollectionInfo)
to/from qdrant_client.models at the boundary. Tool code never imports
qdrant_client directly.

Phase 2 of the backend abstraction plan.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Iterable

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from shared.store_models import (
    PointStruct, ScoredPoint, Filter, FieldCondition,
    MatchValue, MatchText, MatchAny, MatchContains, Range,
    CollectionInfo, SparseVector,
)

logger = logging.getLogger(__name__)

# S1: Prefetch/FusionQuery added in qdrant-client 1.7+. Fall back to Python RRF.
try:
    from qdrant_client.models import FusionQuery, Prefetch, Fusion
    _NATIVE_HYBRID = True
except ImportError:
    _NATIVE_HYBRID = False
    logger.warning("qdrant-client <1.7: hybrid search falls back to Python RRF")


# ---------------------------------------------------------------------------
# Translation helpers (neutral → qdrant)
# ---------------------------------------------------------------------------

def _to_qdrant_filter(f: Filter | None) -> qm.Filter | None:
    """Translate neutral Filter → qdrant_client.models.Filter."""
    if f is None:
        return None
    if not f.must and not f.should and not f.must_not:
        return None
    return qm.Filter(
        must=[_to_qdrant_condition(c) for c in f.must] or None,
        should=[_to_qdrant_condition(c) for c in f.should] or None,
        must_not=[_to_qdrant_condition(c) for c in f.must_not] or None,
    )


def _to_qdrant_condition(c: FieldCondition) -> qm.FieldCondition:
    """Translate one FieldCondition.
    MatchContains → MatchValue (Qdrant payload arrays already do array-contains on MatchValue).
    """
    match = None
    if isinstance(c.match, MatchValue):
        match = qm.MatchValue(value=c.match.value)
    elif isinstance(c.match, MatchText):
        match = qm.MatchText(text=c.match.text)
    elif isinstance(c.match, MatchAny):
        match = qm.MatchAny(any=c.match.values)
    elif isinstance(c.match, MatchContains):
        match = qm.MatchValue(value=c.match.value)
    elif c.match is not None:
        match = qm.MatchValue(value=c.match.value)

    range_ = None
    if c.range_:
        range_ = qm.Range(
            gt=c.range_.gt, gte=c.range_.gte,
            lt=c.range_.lt, lte=c.range_.lte,
        )

    return qm.FieldCondition(key=c.key, match=match, range=range_)


def _to_qdrant_point(p: PointStruct) -> qm.PointStruct:
    """Translate neutral PointStruct → qdrant_client.models.PointStruct."""
    vector = p.vector
    if p.sparse_vector is not None:
        if isinstance(vector, list):
            vector = {"dense": vector}
        elif not isinstance(vector, dict):
            vector = {"dense": vector}
        vector["sparse"] = qm.SparseVector(
            indices=p.sparse_vector.indices,
            values=p.sparse_vector.values,
        )
    return qm.PointStruct(id=p.id, vector=vector, payload=p.payload or {})


# ---------------------------------------------------------------------------
# Translation helpers (qdrant → neutral)
# ---------------------------------------------------------------------------

def _from_qdrant_scored(sp) -> ScoredPoint:
    """Translate qdrant ScoredPoint → neutral ScoredPoint."""
    return ScoredPoint(
        id=str(sp.id),
        score=float(sp.score),
        payload=sp.payload,
        vector=getattr(sp, 'vector', None),
    )


def _from_qdrant_collection_info(name: str, info) -> CollectionInfo:
    """Translate qdrant collection info → neutral CollectionInfo."""
    vc = info.config.params.vectors
    if isinstance(vc, dict):
        named_vectors = {}
        for k, v in vc.items():
            if hasattr(v, 'size'):
                named_vectors[k] = v.size
    elif vc and hasattr(vc, 'size'):
        named_vectors = {"": vc.size}
    else:
        named_vectors = {}

    has_sparse = bool(info.config.params.sparse_vectors)

    distance = "Cosine"
    if vc and hasattr(vc, 'distance'):
        distance = str(vc.distance).capitalize()
    elif isinstance(vc, dict):
        for v in vc.values():
            if hasattr(v, 'distance'):
                distance = str(v.distance).capitalize()
                break

    return CollectionInfo(
        name=name,
        points_count=info.points_count or 0,
        named_vectors=named_vectors,
        has_sparse=has_sparse,
        distance=distance,
    )


# ---------------------------------------------------------------------------
# Python-side RRF fusion (fallback for qdrant-client < 1.7)
# ---------------------------------------------------------------------------

def _rrf_fuse(dense_hits: list[ScoredPoint], sparse_hits: list[ScoredPoint],
              limit: int = 10, k: int = 60) -> list[ScoredPoint]:
    """Reciprocal Rank Fusion — combines two ranked lists into one."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
    return [ScoredPoint(id=id, score=score, payload=payloads.get(id))
            for id, score in ranked]


# ---------------------------------------------------------------------------
# QdrantVectorStore
# ---------------------------------------------------------------------------

class QdrantVectorStore:
    """Thin adapter: delegates to qdrant_client, translates types at the boundary."""

    def __init__(self, host: str = "qdrant", port: int = 6333, timeout: int = 30,
                 client: QdrantClient | None = None):
        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(host=host, port=port, timeout=timeout)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ensure_collection(
        self, name: str, *,
        dense_dim: int | None = None,
        sparse: bool = False,
        distance: str = "Cosine",
    ) -> None:
        try:
            self._client.get_collection(name)
            return
        except Exception:
            pass

        dist = qm.Distance.COSINE if distance == "Cosine" else qm.Distance.EUCLID

        if dense_dim is None and sparse:
            # Sparse-only collection
            self._client.create_collection(
                collection_name=name,
                vectors_config={},
                sparse_vectors_config={
                    "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams())
                },
            )
        elif dense_dim is not None and sparse:
            # Hybrid: named dense + named sparse
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": qm.VectorParams(size=dense_dim, distance=dist)
                },
                sparse_vectors_config={
                    "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams())
                },
            )
        elif dense_dim is not None:
            # Dense-only
            self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=dense_dim, distance=dist),
            )
        else:
            raise ValueError("ensure_collection requires at least dense_dim or sparse")

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[PointStruct]) -> None:
        self._client.upsert(
            collection_name=collection,
            points=[_to_qdrant_point(p) for p in points],
        )

    def set_payload(self, collection: str, payload: dict, *, ids: list[str]) -> None:
        self._client.set_payload(
            collection_name=collection, payload=payload, points=ids,
        )

    def delete(self, collection: str, *,
               ids: list[str] | None = None,
               filter: Filter | None = None) -> None:
        if ids:
            self._client.delete(collection_name=collection, points_selector=ids)
        elif filter:
            self._client.delete(
                collection_name=collection,
                points_selector=_to_qdrant_filter(filter),
            )

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    # ------------------------------------------------------------------
    # reads — three query methods
    # ------------------------------------------------------------------

    def query_dense(self, collection: str, vec: list[float], *,
                    limit: int = 10, filter: Filter | None = None,
                    using: str | None = None) -> list[ScoredPoint]:
        res = self._client.query_points(
            collection_name=collection,
            query=vec,
            using=using,
            query_filter=_to_qdrant_filter(filter),
            limit=limit,
            with_payload=True,
        )
        return [_from_qdrant_scored(p) for p in res.points]

    def query_sparse(self, collection: str, sparse: SparseVector, *,
                     limit: int = 10, filter: Filter | None = None,
                     query_text: str | None = None) -> list[ScoredPoint]:
        # Qdrant matches via the native sparse vector index (integer term hashes
        # shared between index and query within a process). ``query_text`` is
        # accepted for Protocol parity but not used here.
        qv = qm.SparseVector(indices=sparse.indices, values=sparse.values)
        res = self._client.query_points(
            collection_name=collection,
            query=qv,
            using="sparse",
            query_filter=_to_qdrant_filter(filter),
            limit=limit,
            with_payload=True,
        )
        return [_from_qdrant_scored(p) for p in res.points]

    def query_hybrid(self, collection: str, dense: list[float],
                     sparse: SparseVector, *, limit: int = 10,
                     filter: Filter | None = None,
                     query_text: str | None = None) -> list[ScoredPoint]:
        qf = _to_qdrant_filter(filter)

        if _NATIVE_HYBRID:
            qv = qm.NamedVector(name="dense", vector=dense)
            qsv = qm.NamedSparseVector(
                name="sparse",
                vector=qm.SparseVector(indices=sparse.indices, values=sparse.values),
            )
            res = self._client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(query=qv, using="dense", limit=limit * 2),
                    Prefetch(query=qsv, using="sparse", limit=limit * 2),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=qf,
                limit=limit,
                with_payload=True,
            )
            return [_from_qdrant_scored(p) for p in res.points]
        else:
            # S1 fallback: run both queries, fuse with Python RRF
            dense_hits = self.query_dense(collection, dense, limit=limit * 2, filter=filter, using="dense")
            sparse_hits = self.query_sparse(collection, sparse, limit=limit * 2, filter=filter)
            return _rrf_fuse(dense_hits, sparse_hits, limit=limit)

    def retrieve(self, collection: str, ids: list[str], *,
                 with_payload: bool = True, with_vectors: bool = False) -> list[PointStruct]:
        results = self._client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return [
            PointStruct(
                id=str(r.id),
                vector=getattr(r, 'vector', None) if with_vectors else None,
                payload=r.payload,
            )
            for r in results
        ]

    def scroll(self, collection: str, *, limit: int = 1000, offset=None,
               with_payload: bool = True,
               filter: Filter | None = None,
    ) -> tuple[list[PointStruct], Any]:
        results, next_offset = self._client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
            scroll_filter=_to_qdrant_filter(filter),
        )
        points = [
            PointStruct(id=str(r.id), vector=None, payload=r.payload)
            for r in results
        ]
        return points, next_offset

    def get_collection(self, name: str) -> CollectionInfo:
        info = self._client.get_collection(name)
        return _from_qdrant_collection_info(name, info)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    # ------------------------------------------------------------------
    # migration (R1: default with_vectors=False)
    # ------------------------------------------------------------------

    def iter_all(self, collection: str, *, with_vectors: bool = False) -> Iterator[PointStruct]:
        """Stream all points. Backends MUST yield one at a time."""
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for r in results:
                yield PointStruct(
                    id=str(r.id),
                    vector=getattr(r, 'vector', None) if with_vectors else None,
                    payload=r.payload,
                )
            if not offset or not results:
                break

    def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int:
        batch: list = []
        count = 0
        for p in points:
            batch.append(_to_qdrant_point(p))
            if len(batch) >= 100:
                self._client.upsert(collection_name=collection, points=batch)
                count += len(batch)
                batch = []
        if batch:
            self._client.upsert(collection_name=collection, points=batch)
            count += len(batch)
        return count
