"""
Backend-neutral types for the storage abstraction layer.

These types are the ONLY shapes that tool code (memory_tools.py, ragmcp_fastmcp.py, etc.)
should import. Concrete backends (qdrant_client, libsql, psycopg) translate to/from
these types at the impl boundary.

Phase 0 of the backend abstraction plan — no behavior change, pure addition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Iterable, Literal


# ---------------------------------------------------------------------------
# Type aliases for backend names
# ---------------------------------------------------------------------------

SQLBackend = Literal["postgres", "turso", "none"]
VectorBackend = Literal["qdrant", "turso", "postgres", "none"]


# ---------------------------------------------------------------------------
# Sparse vector
# ---------------------------------------------------------------------------

@dataclass
class SparseVector:
    """Sparse vector as {indices, values}. Matches Qdrant's representation."""
    indices: list[int]
    values: list[float]


# ---------------------------------------------------------------------------
# Collection metadata (read schema back)
# ---------------------------------------------------------------------------

@dataclass
class CollectionInfo:
    """Schema/capability info for a vector collection."""
    name: str
    points_count: int
    named_vectors: dict[str, int] = field(default_factory=dict)
    has_sparse: bool = False
    distance: str = "Cosine"

    @property
    def dim(self) -> int | None:
        """Convenience: dimension of the single (or first) dense vector."""
        return next(iter(self.named_vectors.values()), None)

    @property
    def has_dense(self) -> bool:
        """True if any dense vector is configured."""
        return len(self.named_vectors) > 0


# ---------------------------------------------------------------------------
# Write shapes
# ---------------------------------------------------------------------------

@dataclass
class PointStruct:
    """
    A point to upsert into a vector collection.

    vector: list[float] for unnamed dense, dict[str, list[float]] for named dense.
    sparse_vector: optional SparseVector for hybrid collections.
    """
    id: str | int
    vector: list[float] | dict[str, list[float]]
    sparse_vector: SparseVector | None = None
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Query result shape
# ---------------------------------------------------------------------------

@dataclass
class ScoredPoint:
    """
    A ranked result from a vector query.

    CONTRACT: ``score`` is ALWAYS similarity (higher = better), NEVER distance.
    All impls MUST normalize: cosine similarity, negated bm25, negated distance, etc.
    Callers can safely sort by ``score`` descending.
    """
    id: str | int
    score: float
    payload: dict | None = None
    vector: list[float] | dict[str, list[float]] | None = None


# ---------------------------------------------------------------------------
# Filter shapes
# ---------------------------------------------------------------------------

@dataclass
class Range:
    """Numeric range filter on a payload field."""
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None


@dataclass
class MatchValue:
    """Exact equality on a scalar field."""
    value: Any


@dataclass
class MatchText:
    """Full-text / substring match on a string field."""
    text: str


@dataclass
class MatchAny:
    """Any-of match (field value is in the given list)."""
    values: list[Any]


@dataclass
class MatchContains:
    """Array-contains-scalar match (field is a JSON array containing this value)."""
    value: Any


@dataclass
class FieldCondition:
    """A single filter condition on a payload field."""
    key: str
    match: MatchValue | MatchText | MatchAny | MatchContains | None = None
    range_: Range | None = None


@dataclass
class Filter:
    """A boolean filter combining multiple conditions."""
    must: list[FieldCondition] = field(default_factory=list)
    should: list[FieldCondition] = field(default_factory=list)
    must_not: list[FieldCondition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connection info
# ---------------------------------------------------------------------------

@dataclass
class BackendInfo:
    """Info about the active backend, for logging and diagnostics."""
    sql_backend: SQLBackend
    vector_backend: VectorBackend
    sql_available: bool = False
    vector_available: bool = False
