"""
SqlStore — abstract interface for relational/metadata backends.

Defines the SqlStore Protocol, a factory (get_sql_store), and a no-op fallback
(_NullSqlStore). Concrete impls live in tools/shared/impls/.

Phase 1 of the backend abstraction plan.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, Iterator, Iterable, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SqlStore(Protocol):
    """
    Relational/metadata/full-text store.

    Implementations: PostgresSqlStore, TursoSqlStore.
    Optional — if no SQL backend is configured, tools fall back to vector-only mode.
    """

    is_available: bool

    # CRUD
    def upsert_memory(self, memory_id: str, text: str, memory_type: str,
                      source: str, tags: list[str], path: str | None,
                      commit: str | None, agent_id: str | None,
                      sensitivity: str, retention_policy: str) -> str: ...
    def get_memory(self, memory_id: str) -> dict | None: ...
    def delete_memory(self, memory_id: str) -> bool: ...

    # Search
    def search_text(self, query: str, limit: int = 10, *,
                    memory_type: str | None = None,
                    tags: list[str] | None = None,
                    agent_id: str | None = None) -> list[dict]: ...

    # Analytics
    def get_metrics(self) -> dict: ...
    def decay_memories(self, ttl_days: int, min_usage_count: int,
                       retention_policy: str | None = None) -> int: ...
    def get_all_memory_ids(self) -> list[str]: ...

    # Migration support — MUST stream without materializing full dataset
    def iter_all(self) -> Iterator[dict]: ...
    def bulk_upsert(self, rows: Iterable[dict]) -> int: ...


# ---------------------------------------------------------------------------
# No-op fallback
# ---------------------------------------------------------------------------

class NullSqlStore:
    """No-op store for when no SQL backend is configured."""

    is_available = False

    def upsert_memory(self, memory_id, *a, **kw) -> str:
        return memory_id

    def get_memory(self, *a, **kw) -> dict | None:
        return None

    def delete_memory(self, *a, **kw) -> bool:
        return False

    def search_text(self, *a, **kw) -> list[dict]:
        return []

    def get_metrics(self, *a, **kw) -> dict:
        return {}

    def decay_memories(self, *a, **kw) -> int:
        return 0

    def get_all_memory_ids(self, *a, **kw) -> list[str]:
        return []

    def iter_all(self) -> Iterator[dict]:
        return
        yield  # type: ignore[unreachable]  # empty generator

    def bulk_upsert(self, rows) -> int:
        return 0


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_singleton: SqlStore | None = None


def get_sql_store() -> SqlStore:
    """
    Return the process-scoped SqlStore singleton.

    Resolves backend from: config.json storage.sql > env vars > None.
    Once resolved, the same instance is returned on every call.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    from shared.store_factory import resolve_sql_backend
    backend = resolve_sql_backend()

    if backend is None:
        _singleton = NullSqlStore()
        return _singleton

    _singleton = make_sql_store(backend["name"], backend)
    return _singleton


def make_sql_store(backend_name: str, config: dict | None = None) -> SqlStore:
    """
    Create a SqlStore by name. Used by the factory and the migration tool.

    Does NOT cache — each call creates a new instance.
    For the singleton, use get_sql_store().
    """
    config = config or {}

    if backend_name == "postgres":
        from shared.impls.postgres_sql import PostgresSqlStore
        dsn = config.get("dsn") or _pg_dsn_from_env()
        return PostgresSqlStore(dsn=dsn)

    if backend_name == "turso":
        from shared.impls.turso_sql import TursoSqlStore
        url = config.get("url") or os.getenv("TURSO_DATABASE_URL", "file::memory:")
        auth_token = config.get("auth_token") or os.getenv("TURSO_AUTH_TOKEN")
        return TursoSqlStore(url=url, auth_token=auth_token)

    if backend_name == "none":
        return NullSqlStore()

    raise ValueError(f"Unknown SQL backend: {backend_name}")


def _pg_dsn_from_env() -> str:
    """Build PostgreSQL DSN from env vars (shared with store_factory)."""
    from shared.store_factory import pg_dsn_from_env
    return pg_dsn_from_env()
