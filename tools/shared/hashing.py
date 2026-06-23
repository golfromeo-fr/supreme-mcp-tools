"""
Deterministic text hashing for memory deduplication.

Extracted from pg_store.py — pure function, no DB dependencies.
Used by all SqlStore impls (PG, Turso) for dedup key generation.

Phase 0 of the backend abstraction plan — no behavior change, pure extraction.
"""
import hashlib


def text_hash(text: str) -> str:
    """
    Deterministic hash for deduplication.

    SHA256 of stripped+lowercased text, truncated to 40 chars.
    Same input always produces the same hash; whitespace and case
    variations are normalized before hashing.
    """
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:40]
