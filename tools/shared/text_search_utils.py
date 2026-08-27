"""
Shared helpers for text-search backends (FTS5 / tsvector).

Phase 3 cleanup: extracted from the duplicated ``_escape_fts5_query`` that
lived in both turso_sql.py and turso_vector.py. Pure functions, no DB deps.
"""
from __future__ import annotations


def escape_fts5_query(query: str) -> str:
    """
    Escape a user query for an FTS5 MATCH expression as OR-joined terms.

    Each whitespace-separated term becomes a double-quoted FTS5 phrase with
    internal quotes doubled per the FTS5 escaping rule, joined with OR —
    BM25-style recall: rows matching ANY term are returned and bm25() ranks
    them. (Wrapping the whole string as one phrase silently returned zero
    hits whenever the words were not adjacent in the text, which broke every
    multi-term lexical search.) Quoting every term keeps operators and
    punctuation inert, preventing MATCH injection from untrusted query text.
    """
    terms = [t.replace('"', '""') for t in query.split() if t]
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)
