"""
Unit tests for tools/shared/relevance_scorer.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


class TestScoringWeights(unittest.TestCase):
    """Test ScoringWeights defaults and parameter handling."""

    def test_defaults(self):
        from shared.relevance_scorer import ScoringWeights
        w = ScoringWeights()
        self.assertAlmostEqual(w.alpha, 0.5)
        self.assertAlmostEqual(w.beta, 0.3)
        self.assertAlmostEqual(w.gamma, 0.2)
        self.assertAlmostEqual(w.recency_half_life_days, 14.0)
        self.assertEqual(w.max_usage, 100)

    def test_explicit_values(self):
        from shared.relevance_scorer import ScoringWeights
        w = ScoringWeights(alpha=0.7, beta=0.2, gamma=0.1, recency_half_life_days=7.0, max_usage=50)
        self.assertAlmostEqual(w.alpha, 0.7)
        self.assertAlmostEqual(w.beta, 0.2)
        self.assertAlmostEqual(w.gamma, 0.1)
        self.assertAlmostEqual(w.recency_half_life_days, 7.0)
        self.assertEqual(w.max_usage, 50)

    def test_zero_recency_half_life(self):
        from shared.relevance_scorer import ScoringWeights
        w = ScoringWeights(recency_half_life_days=0.0)
        self.assertEqual(w.recency_half_life_days, 0.0)

    def test_zero_max_usage(self):
        from shared.relevance_scorer import ScoringWeights
        w = ScoringWeights(max_usage=0)
        self.assertEqual(w.max_usage, 0)

    def test_repr(self):
        from shared.relevance_scorer import ScoringWeights
        w = ScoringWeights()
        r = repr(w)
        self.assertIn("ScoringWeights", r)
        self.assertIn("0.5", r)


class TestComputeRecencyDecay(unittest.TestCase):
    """Test compute_recency_decay."""

    def test_none_last_accessed(self):
        from shared.relevance_scorer import compute_recency_decay
        self.assertEqual(compute_recency_decay(None), 0.0)

    def test_just_accessed(self):
        from shared.relevance_scorer import compute_recency_decay
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        score = compute_recency_decay(now)
        self.assertGreater(score, 0.9)

    def test_old_access(self):
        from shared.relevance_scorer import compute_recency_decay
        score = compute_recency_decay("2020-01-01T00:00:00+00:00")
        self.assertLess(score, 0.01)

    def test_custom_half_life(self):
        from shared.relevance_scorer import compute_recency_decay
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        score_14 = compute_recency_decay(recent, half_life_days=14)
        score_1 = compute_recency_decay(recent, half_life_days=1)
        self.assertGreater(score_14, score_1)

    def test_invalid_timestamp(self):
        from shared.relevance_scorer import compute_recency_decay
        score = compute_recency_decay("not-a-date")
        self.assertEqual(score, 0.0)


class TestComputeUsageBoost(unittest.TestCase):
    """Test compute_usage_boost."""

    def test_zero_usage(self):
        from shared.relevance_scorer import compute_usage_boost
        self.assertEqual(compute_usage_boost(0), 0.0)

    def test_negative_usage(self):
        from shared.relevance_scorer import compute_usage_boost
        self.assertEqual(compute_usage_boost(-5), 0.0)

    def test_max_usage(self):
        from shared.relevance_scorer import compute_usage_boost
        self.assertAlmostEqual(compute_usage_boost(100, max_usage=100), 1.0)

    def test_exceeds_max(self):
        from shared.relevance_scorer import compute_usage_boost
        self.assertAlmostEqual(compute_usage_boost(200, max_usage=100), 1.0)

    def test_half_usage(self):
        from shared.relevance_scorer import compute_usage_boost
        self.assertAlmostEqual(compute_usage_boost(50, max_usage=100), 0.5)

    def test_zero_max_usage_regression(self):
        from shared.relevance_scorer import compute_usage_boost
        result = compute_usage_boost(5, max_usage=0)
        self.assertEqual(result, 1.0)


class TestComputeRelevanceScore(unittest.TestCase):
    """Test compute_relevance_score."""

    def test_perfect_score(self):
        from shared.relevance_scorer import compute_relevance_score
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        score = compute_relevance_score(
            semantic_similarity=1.0,
            last_accessed=now,
            usage_count=100,
        )
        self.assertGreater(score, 0.9)

    def test_minimal_score(self):
        from shared.relevance_scorer import compute_relevance_score
        score = compute_relevance_score(
            semantic_similarity=0.0,
            last_accessed=None,
            usage_count=0,
        )
        self.assertAlmostEqual(score, 0.0)

    def test_clamps_to_one(self):
        from shared.relevance_scorer import compute_relevance_score, ScoringWeights
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        w = ScoringWeights(alpha=0.5, beta=0.3, gamma=0.2)
        score = compute_relevance_score(1.0, now, 1000, weights=w)
        self.assertLessEqual(score, 1.0)


class TestScoreRelevance(unittest.TestCase):
    """Test score_relevance main entry point."""

    def test_with_semantic_score(self):
        from shared.relevance_scorer import score_relevance
        payload = {"usage_count": 5, "last_accessed": "2025-01-01T00:00:00+00:00"}
        score = score_relevance(payload, semantic_score=0.8)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_without_semantic_score_no_vector(self):
        from shared.relevance_scorer import score_relevance
        payload = {"usage_count": 5, "last_accessed": "2025-01-01T00:00:00+00:00"}
        score = score_relevance(payload, semantic_score=None)
        self.assertGreater(score, 0.0)

    def test_with_vectors(self):
        from shared.relevance_scorer import score_relevance
        payload = {
            "usage_count": 5,
            "last_accessed": "2025-01-01T00:00:00+00:00",
            "embedding": [0.1] * 10,
        }
        query_vec = [0.1] * 10
        score = score_relevance(payload, query_vector=query_vec)
        self.assertGreater(score, 0.0)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine_similarity."""

    def test_identical_vectors(self):
        from shared.relevance_scorer import cosine_similarity
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        from shared.relevance_scorer import cosine_similarity
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self):
        from shared.relevance_scorer import cosine_similarity
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_different_lengths(self):
        from shared.relevance_scorer import cosine_similarity
        self.assertEqual(cosine_similarity([1, 2], [1, 2, 3]), 0.0)

    def test_zero_vector(self):
        from shared.relevance_scorer import cosine_similarity
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
