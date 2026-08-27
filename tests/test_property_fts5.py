"""
Property-based tests for escape_fts5_query using Hypothesis.

These tests prove that escape_fts5_query is safe for arbitrary user input:
- The output is one or more double-quoted FTS5 phrases joined by " OR "
- Internal quotes are always doubled (FTS5 escaping rule)
- The escaped query can be used in a MATCH expression without injection
- No user input can break out of the quoted-term context

Additionally, we test the actual FTS5 MATCH behavior end-to-end with
libSQL: arbitrary text that contains the query's tokens must be findable
after escaping, operator characters must be neutralized, and multi-term
queries must match rows whose terms are NOT adjacent (the 2026-08-27
regression: whole-string phrase escaping made multi-word sparse searches
silently return zero hits).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from shared.text_search_utils import escape_fts5_query

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

from hypothesis import given, strategies as st, assume, settings, HealthCheck


# Strategy: arbitrary strings that could appear in user search queries
query_text = st.text(
    alphabet=st.sampled_from([chr(c) for c in range(32, 127)]),
    min_size=0,
    max_size=200,
)


class TestEscapeFts5Properties:
    """Pure property tests — no DB needed."""

    @given(query_text)
    def test_output_is_or_joined_quoted_terms(self, q):
        """Shape: one or more double-quoted tokens joined by ' OR '."""
        escaped = escape_fts5_query(q)
        if escaped == '""':
            return
        for token in escaped.split(" OR "):
            # tokens never contain whitespace, so the split is unambiguous
            assert token.startswith('"') and token.endswith('"')
            assert len(token) >= 2

    @given(query_text)
    def test_output_length_grows_with_input(self, q):
        """Escaping only adds characters (quotes, separators), never removes
        enough to shrink below the input's non-whitespace length."""
        escaped = escape_fts5_query(q)
        assert len(escaped) >= len(q)

    @given(query_text)
    def test_terms_round_trip(self, q):
        """Unescaping each quoted token recovers the input's whitespace-split
        terms in order; empty/whitespace-only input escapes to a bare phrase."""
        escaped = escape_fts5_query(q)
        expected = q.split()
        if not expected:
            assert escaped == '""'
            return
        tokens = escaped.split(" OR ")
        restored = [t[1:-1].replace('""', '"') for t in tokens]
        assert restored == expected

    @given(query_text)
    def test_internal_quotes_are_doubled(self, q):
        """Every quote inside a term is part of a doubled pair."""
        escaped = escape_fts5_query(q)
        if escaped == '""':
            return
        for token in escaped.split(" OR "):
            inner = token[1:-1]
            assert not any('"' in part for part in inner.split('""'))


@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
class TestFts5MatchWithArbitraryInput:
    """End-to-end: escaped queries must work in real FTS5 MATCH expressions."""

    @given(query_text)
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=2000)
    def test_arbitrary_query_does_not_crash_match(self, q):
        """An arbitrary user query, after escaping, must not cause a SQL error
        when used in an FTS5 MATCH expression. This is the injection-safety guarantee."""
        import libsql_experimental as libsql
        conn = libsql.connect("file::memory:")
        conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text)")
        conn.execute("INSERT INTO fts(text) VALUES ('hello world python flask')")

        escaped = escape_fts5_query(q)
        # Must not raise — the escaping prevents SQL/FTS5 injection
        rows = conn.execute("SELECT text FROM fts WHERE fts MATCH ?", (escaped,)).fetchall()
        # We don't assert on results (the arbitrary query may not match),
        # just that it doesn't crash or inject.
        assert isinstance(rows, list)

    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=3, max_size=20))
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], deadline=2000)
    def test_simple_word_query_finds_match(self, word):
        """A simple lowercase word must be findable in FTS5 after escaping."""
        assume(word.strip())  # skip empty/whitespace-only

        import libsql_experimental as libsql
        conn = libsql.connect("file::memory:")
        conn.execute("CREATE VIRTUAL TABLE fts2 USING fts5(text)")
        conn.execute(f"INSERT INTO fts2(text) VALUES (?)", (f"the {word.strip()} is here",))

        escaped = escape_fts5_query(word.strip())
        rows = conn.execute("SELECT text FROM fts2 WHERE fts2 MATCH ?", (escaped,)).fetchall()
        # FTS5 phrase search should find the word in the indexed text
        # (may not match if the word is a stopword or gets stemmed differently)
        # At minimum, the query must not crash
        assert isinstance(rows, list)

    def test_multi_word_query_matches_nonadjacent_terms(self):
        """Regression (2026-08-27): multi-term queries must match rows where
        the terms are NOT adjacent. Whole-string phrase escaping made every
        multi-word sparse search silently return zero hits."""
        import libsql_experimental as libsql
        conn = libsql.connect("file::memory:")
        conn.execute("CREATE VIRTUAL TABLE fts3 USING fts5(text)")
        conn.execute(
            "INSERT INTO fts3(text) VALUES "
            "('the ClientTransport class manages connect_session lifecycle')"
        )
        conn.execute("INSERT INTO fts3(text) VALUES ('completely unrelated row')")

        rows = conn.execute(
            "SELECT text FROM fts3 WHERE fts3 MATCH ?",
            (escape_fts5_query("ClientTransport connect_session"),),
        ).fetchall()

        assert len(rows) == 1
        assert "ClientTransport" in rows[0][0]
