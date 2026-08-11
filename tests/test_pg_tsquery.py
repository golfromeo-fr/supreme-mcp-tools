"""
Tests for _build_tsquery — the pure function that builds a PostgreSQL tsquery.

This function is the core of query_sparse/query_hybrid on the PG backend.
It decides whether to use websearch_to_tsquery (when query_text is provided)
or fall back to synthetic terms from integer sparse indices. No DB connection
needed — the function returns (term, sql_expr) tuples.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from shared.store_models import SparseVector
from shared.impls.postgres_vector import _build_tsquery


class TestBuildTsquery:
    def test_with_query_text_uses_websearch(self):
        term, sql = _build_tsquery("python flask", SparseVector(indices=[], values=[]))
        assert term == "python flask"
        assert "websearch_to_tsquery" in sql
        assert "%s" in sql  # parameterized

    def test_without_query_text_uses_sparse_indices(self):
        sparse = SparseVector(indices=[101, 202, 303], values=[0.5, 0.3, 0.2])
        term, sql = _build_tsquery(None, sparse)
        assert "x101" in term
        assert "x202" in term
        assert "x303" in term
        assert " | " in term  # OR-joined
        assert "to_tsquery" in sql
        assert "websearch_to_tsquery" not in sql

    def test_without_query_text_caps_at_10_terms(self):
        indices = list(range(100, 120))
        sparse = SparseVector(indices=indices, values=[1.0] * 20)
        term, sql = _build_tsquery(None, sparse)
        # Only first 10 sparse indices are used
        parts = term.split(" | ")
        assert len(parts) == 10
        assert "x100" in parts[0]
        assert "x109" in parts[-1]
        assert "x110" not in term

    def test_empty_everything_falls_back_gracefully(self):
        """No query_text and no sparse indices → synthetic 'xnone' placeholder."""
        term, sql = _build_tsquery(None, SparseVector(indices=[], values=[]))
        assert term == "xnone"
        assert "to_tsquery" in sql

    def test_query_text_empty_string_falls_back_to_sparse(self):
        """Empty string query_text is falsy → should fall back to sparse path."""
        sparse = SparseVector(indices=[42], values=[1.0])
        term, _ = _build_tsquery("", sparse)
        assert "x42" in term

    def test_query_text_takes_precedence_over_sparse(self):
        """When both are provided, query_text wins (it's the real search string)."""
        sparse = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        term, sql = _build_tsquery("real query", sparse)
        assert term == "real query"
        assert "websearch_to_tsquery" in sql

    def test_sql_expr_always_parameterized(self):
        """The SQL expression must use %s — never inline the term."""
        for query_text, sparse in [
            ("hello", SparseVector(indices=[], values=[])),
            (None, SparseVector(indices=[1, 2], values=[1.0, 0.5])),
            (None, SparseVector(indices=[], values=[])),
        ]:
            _, sql = _build_tsquery(query_text, sparse)
            assert "%s" in sql, f"SQL must be parameterized, got: {sql}"
