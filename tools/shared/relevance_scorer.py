"""
Relevance Scorer - Computes relevance scores combining semantic similarity,
recency decay, and usage frequency.
"""

import math
import os
import re
from datetime import datetime, timezone

class ScoringWeights:
    """Configurable weights for the relevance scoring formula."""

    def __init__(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        recency_half_life_days: float | None = None,
        max_usage: int | None = None,
    ):
        # Semantic similarity weight (default 0.5)
        self.alpha = alpha if alpha is not None else float(os.getenv("MEMORY_ALPHA", "0.5"))
        # Recency decay weight (default 0.3)
        self.beta = beta if beta is not None else float(os.getenv("MEMORY_BETA", "0.3"))
        # Usage boost weight (default 0.2)
        self.gamma = gamma if gamma is not None else float(os.getenv("MEMORY_GAMMA", "0.2"))
        # Recency half-life in days (default 14)
        self.recency_half_life_days = recency_half_life_days if recency_half_life_days is not None else float(
            os.getenv("MEMORY_RECENCY_HALF_LIFE_DAYS", "14")
        )
        self.max_usage = max_usage if max_usage is not None else int(os.getenv("MEMORY_MAX_USAGE", "100"))

    def __repr__(self) -> str:
        return (
            f"ScoringWeights(α={self.alpha}, β={self.beta}, γ={self.gamma}, "
            f"half_life={self.recency_half_life_days}d, max_usage={self.max_usage})"
        )


def compute_recency_decay(
    last_accessed: str | None,
    now: str | None = None,
    half_life_days: float = 14.0,
) -> float:
    """
    Compute recency decay score using exponential decay.

    Formula: exp(-days_since_access / half_life)

    Args:
        last_accessed: ISO timestamp of last access
        now: Current timestamp (defaults to now)
        half_life_days: Days until score drops to ~0.37

    Returns:
        Float between 0 and 1, where 1 = just accessed
    """
    if not last_accessed:
        return 1.0

    try:
        last_time = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
        current_time = datetime.fromisoformat(now.replace("Z", "+00:00")) if now else datetime.now(timezone.utc)

        days_since = (current_time - last_time).total_seconds() / (24 * 3600)
        decay = math.exp(-days_since / half_life_days)
        return min(1.0, max(0.0, decay))
    except Exception:
        return 0.0


def compute_usage_boost(usage_count: int, max_usage: int = 100) -> float:
    """
    Compute usage boost score via min-max normalization.

    Formula: min(usage_count / max_usage, 1.0)

    Args:
        usage_count: Number of times memory was accessed
        max_usage: Maximum usage for normalization

    Returns:
        Float between 0 and 1, where 1 = frequently used
    """
    if usage_count <= 0:
        return 0.0
    if max_usage <= 0:
        return 1.0
    return min(usage_count / max_usage, 1.0)


def compute_relevance_score(
    semantic_similarity: float,
    last_accessed: str | None,
    usage_count: int,
    weights: ScoringWeights | None = None,
    now: str | None = None,
) -> float:
    """
    Compute combined relevance score.

    Formula: α * semantic_sim + β * recency_decay + γ * usage_boost

    Args:
        semantic_similarity: Vector similarity score (0-1)
        last_accessed: ISO timestamp of last access
        usage_count: Number of times accessed
        weights: Scoring weights (uses defaults if None)
        now: Current timestamp

    Returns:
        Combined relevance score (0-1)
    """
    if weights is None:
        weights = ScoringWeights()

    recency = compute_recency_decay(
        last_accessed,
        now=now,
        half_life_days=weights.recency_half_life_days,
    )

    usage = compute_usage_boost(usage_count, weights.max_usage)

    score = (
        weights.alpha * semantic_similarity +
        weights.beta * recency +
        weights.gamma * usage
    )

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# P3 hybrid recall — keyword lane (plans/odysseus-reuse-2026-10-06.md)
# ---------------------------------------------------------------------------

_MEMORY_TYPE_BOOST_TOKENS = {
    "architectural_decision": {"architectural", "architecture", "decision"},
    "decision": {"decision", "decide"},
    "code_pattern": {"pattern", "code", "coding"},
    "trick": {"trick", "tip", "hack"},
    "lesson": {"lesson", "learned", "learning"},
    "plan": {"plan", "planning", "planned"},
    "idea": {"idea"},
}


def type_boost(memory_type: str | None, query: str) -> float:
    """
    Additive keyword-lane boost when query tokens overlap the memory-type
    label (odysseus's category boost, adapted to this repo's memory types).
    Capped by MEMORY_TYPE_BOOST_CAP (default 0.10); 0.0 without overlap.
    """
    try:
        cap = float(os.getenv("MEMORY_TYPE_BOOST_CAP", "0.10"))
    except ValueError:
        cap = 0.10
    if not memory_type or not query or cap <= 0:
        return 0.0
    mtype = memory_type.strip().lower()
    tokens = set(_MEMORY_TYPE_BOOST_TOKENS.get(mtype, {mtype.replace("_", " ")}))
    qtokens = set(re.findall(r"[a-z]+", query.lower()))
    return cap if tokens & qtokens else 0.0


def normalize_keyword_rows(rows: list[dict]) -> dict[str, float]:
    """
    Map SqlStore.search_text rows to {id: kw_score in [0,1]}.

    Two score shapes exist across the impls (OQ5): PG pg_trgm returns a
    native 0-1 similarity (higher = better) — used directly. Turso FTS5
    bm25() returns negative lower-is-better scores, and the PG ILIKE degrade
    path returns all zeros — those are rank-normalized (best row = 1.0).
    Rows must arrive best-first, as both impls order them.
    """
    if not rows:
        return {}
    scores = [float(r.get("sim_score") or 0.0) for r in rows]
    lo, hi = min(scores), max(scores)
    if 0.0 <= lo and hi <= 1.0 and hi > 0.0:
        return {str(r.get("id")): s for r, s in zip(rows, scores)}
    n = len(rows)
    if n == 1:
        return {str(rows[0].get("id")): 1.0}
    return {str(r.get("id")): 1.0 - (i / n) for i, r in enumerate(rows)}


def blend_hybrid(current_blend: float, kw_norm: float, kw_weight: float) -> float:
    """
    Final hybrid score: (1 - kw_weight) * existing α/β/γ blend + kw_weight *
    boosted keyword score. kw_weight=0 reproduces the pre-P3 score exactly.
    """
    kw = min(1.0, max(0.0, kw_norm))
    w = min(1.0, max(0.0, kw_weight))
    return (1.0 - w) * current_blend + w * kw


def score_relevance(
    memory_payload: dict,
    query_vector: list[float] | None = None,
    weights: ScoringWeights | None = None,
    now: str | None = None,
    semantic_score: float | None = None,
) -> float:
    """
    Score a memory's relevance to a query.

    This is the main entry point for scoring retrieved memories.

    Args:
        memory_payload: Qdrant point payload with metadata
        query_vector: Query embedding vector (optional if semantic_score provided)
        weights: Scoring weights (uses defaults if None)
        now: Current timestamp for recency calculation
        semantic_score: Pre-computed semantic similarity (e.g. from Qdrant)

    Returns:
        Combined relevance score (0-1)
    """
    if weights is None:
        weights = ScoringWeights()

    # Determine semantic similarity
    if semantic_score is not None:
        similarity = semantic_score
    else:
        stored_vector = memory_payload.get("embedding")
        if stored_vector and query_vector:
            similarity = cosine_similarity(query_vector, stored_vector)
        else:
            similarity = 0.5  # neutral fallback

    last_accessed = memory_payload.get("last_accessed")
    usage_count = memory_payload.get("usage_count", 0)

    return compute_relevance_score(
        semantic_similarity=similarity,
        last_accessed=last_accessed,
        usage_count=usage_count,
        weights=weights,
        now=now,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    try:
        import numpy as np
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    except ImportError:
        pass

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)