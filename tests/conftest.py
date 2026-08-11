"""Project-wide pytest configuration.

Two jobs:

1. Load the project ``.env`` once before collection so tests see the same
   backend config the running tools use (e.g. ``POSTGRES_HOST``). Mirrors how
   every tool starts up via ``dotenv``.

2. Provide a ``pg_dsn`` session fixture that resolves a Postgres DSN *and probes
   reachability*, so the optional Postgres contract tests run automatically
   whenever a Postgres is configured and up — and skip cleanly (no failures)
   when it isn't. This is the "run if applicable" behaviour: no special
   ``POSTGRES_TEST_DSN`` var is required if the standard ``POSTGRES_*`` vars are
   present (e.g. the app's own config block in ``.env``).

DSN resolution priority:
    POSTGRES_TEST_DSN  >  assembled from POSTGRES_HOST/PORT/USER/PASSWORD/DB
"""
import os
from pathlib import Path

import pytest

# Load .env once, before collection. Best-effort: a missing/odd .env must never
# break test collection.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def _assemble_pg_dsn() -> str | None:
    """Resolve a PG DSN: explicit override wins, else build from POSTGRES_* vars."""
    explicit = os.getenv("POSTGRES_TEST_DSN")
    if explicit:
        return explicit
    host = os.getenv("POSTGRES_HOST")
    if not host:
        return None
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", user)
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"


@pytest.fixture(scope="session")
def pg_dsn():
    """Return a PG DSN if Postgres is configured AND reachable, else None.

    Session-scoped so the reachability probe runs exactly once for the whole
    suite (a 3s connect timeout keeps a down server from stalling collection).
    """
    dsn = _assemble_pg_dsn()
    if not dsn:
        return None
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            pass
    except Exception:
        return None
    return dsn
