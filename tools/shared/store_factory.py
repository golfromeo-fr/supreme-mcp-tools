"""
Backend factory — resolves SQL + Vector backends from config.

Priority: explicit config.json > env vars > defaults.

Phase 1 of the backend abstraction plan.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_tool_config(tool_name: str) -> dict:
    """Load config.json for a given tool."""
    config_path = Path(__file__).parent.parent / tool_name / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load {config_path}: {e}")
    return {}


def resolve_sql_backend(tool_name: str = "memorymcp") -> dict | None:
    """
    Resolve SQL backend config.

    Priority: config.json storage.sql.backend > env POSTGRES_HOST > env TURSO_DATABASE_URL > None
    """
    cfg = _load_tool_config(tool_name).get("storage", {}).get("sql", {})
    backend = cfg.get("backend")

    # Explicit config
    if backend == "postgres":
        return {"name": "postgres", "dsn": pg_dsn_from_env()}
    if backend == "turso":
        return {
            "name": "turso",
            "url": cfg.get("url") or os.getenv("TURSO_DATABASE_URL"),
            "auth_token": os.getenv(cfg.get("token_env", "TURSO_AUTH_TOKEN")),
        }
    if backend == "none":
        return None

    # Env-var auto-detect
    if os.getenv("POSTGRES_HOST"):
        return {"name": "postgres", "dsn": pg_dsn_from_env()}
    if os.getenv("TURSO_DATABASE_URL"):
        return {
            "name": "turso",
            "url": os.getenv("TURSO_DATABASE_URL"),
            "auth_token": os.getenv("TURSO_AUTH_TOKEN"),
        }

    return None


def resolve_vector_backend(tool_name: str = "memorymcp") -> dict | None:
    """
    Resolve Vector backend config.

    Priority: config.json storage.vector.backend > env QDRANT_HOST > env TURSO_DATABASE_URL > None
    """
    cfg = _load_tool_config(tool_name).get("storage", {}).get("vector", {})
    backend = cfg.get("backend")

    if backend == "qdrant":
        return {
            "name": "qdrant",
            "host": os.getenv("QDRANT_HOST", "qdrant"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
        }
    if backend == "turso":
        return {
            "name": "turso",
            "url": cfg.get("url") or os.getenv("TURSO_DATABASE_URL"),
            "auth_token": os.getenv(cfg.get("token_env", "TURSO_AUTH_TOKEN")),
        }
    if backend == "postgres":
        return {"name": "postgres", "dsn": pg_dsn_from_env()}
    if backend == "none":
        return None

    # Auto-detect
    if os.getenv("QDRANT_HOST"):
        return {
            "name": "qdrant",
            "host": os.getenv("QDRANT_HOST", "qdrant"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
        }
    if os.getenv("TURSO_DATABASE_URL"):
        return {
            "name": "turso",
            "url": os.getenv("TURSO_DATABASE_URL"),
            "auth_token": os.getenv("TURSO_AUTH_TOKEN"),
        }

    return None


def pg_dsn_from_env() -> str:
    """Build PostgreSQL DSN from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "gr")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "memorymcp")
    return f"host={host} port={port} user={user} password={password} dbname={dbname}"
