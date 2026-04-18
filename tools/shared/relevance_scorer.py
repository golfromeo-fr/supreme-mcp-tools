"""
Relevance Scorer - Computes relevance scores combining semantic similarity,
recency decay, and usage frequency.
"""

import math
import os
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
        self.recency_half_life_days = recency_half_life_days or float(
            os.getenv("MEMORY_RECENCY_HALF_LIFE_DAYS", "14")
        )
        # Max usage for normalization (default 100)
        self.max_usage = max_usage or int(os.getenv("MEMORY_MAX_USAGE", "100"))

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
        return 0.0

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

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)