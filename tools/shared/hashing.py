"""
Deterministic text hashing for memory deduplication.

Extracted from pg_store.py — pure function, no DB dependencies.
Used by all SqlStore impls (PG, Turso) for dedup key generation.

Phase 0 of the backend abstraction plan — no behavior change, pure extraction.
"""
import hashlib


def text_hash(text: str, owner: str | None = None) -> str:
    """
    Deterministic hash for deduplication.

    SHA256 of stripped+lowercased text, truncated to 40 chars.
    Same input always produces the same hash; whitespace and case
    variations are normalized before hashing.

    owner=None (default) produces exactly the pre-E3.6 hash, so the mono-mode
    and legacy corpus keep their dedup keys untouched. With an owner, the
    owner is mixed into the hash (sha256("<owner>\\0<normalized>")) giving
    each E3 user their own dedup namespace — identical text+type from two
    users must not collapse onto one row (P1, plans/odysseus-reuse-2026-10-06.md;
    pattern re-derived from odysseus's owner-mixed ids, no code copied).
    """
    normalized = text.strip().lower()
    if owner:
        return hashlib.sha256(f"{owner}\0{normalized}".encode()).hexdigest()[:40]
    return hashlib.sha256(normalized.encode()).hexdigest()[:40]
