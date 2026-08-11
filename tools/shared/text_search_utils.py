"""
Shared helpers for text-search backends (FTS5 / tsvector).

Phase 3 cleanup: extracted from the duplicated ``_escape_fts5_query`` that
lived in both turso_sql.py and turso_vector.py. Pure functions, no DB deps.
"""
from __future__ import annotations


def escape_fts5_query(query: str) -> str:
    """
    Escape a user query for an FTS5 MATCH expression.

    Wraps the whole query in double quotes (so it is treated as a phrase and
    operators/punctuation are not interpreted), and doubles any internal double
    quotes per the FTS5 escaping rule. This also prevents MATCH injection from
    untrusted query text.
    """
    escaped = query.replace('"', '""')
    return f'"{escaped}"'
