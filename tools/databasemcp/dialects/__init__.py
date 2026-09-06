"""databasemcp dialect layer — one DbDialect implementation per database type.

Protocol (plans/databasemcp-overhaul-2026-09-06.md): each dialect owns the
database-specific SQL, bind style, catalog queries, and error formatting;
the connection registry and tools stay dialect-agnostic.
"""
from ._base import DbDialect
from .oracle import OracleDialect
from .postgres import PostgresDialect
from .libsql import LibsqlDialect

DIALECTS: dict[str, DbDialect] = {
    "oracle": OracleDialect(),
    "postgres": PostgresDialect(),
    "libsql": LibsqlDialect(),
}


def get_dialect(db_type: str) -> DbDialect:
    try:
        return DIALECTS[db_type]
    except KeyError:
        supported = ", ".join(sorted(DIALECTS))
        raise ValueError(
            f"Unsupported db_type '{db_type}'. Supported: {supported}"
        ) from None
