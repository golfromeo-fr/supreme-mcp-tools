"""DbDialect protocol — shared by all databasemcp dialects."""
from abc import ABC, abstractmethod
from typing import Any


class DbDialect(ABC):
    """One implementation per database type."""

    name: str = ""
    REQUIRED_PARAMS: tuple[str, ...] = ()

    @abstractmethod
    def connect(self, params: dict) -> Any:
        """Open a pool/connection handle; raise on failure."""

    @abstractmethod
    def close(self, handle) -> None:
        """Close the handle (best-effort)."""

    @abstractmethod
    def ping(self, handle) -> None:
        """Health-check; raise on failure."""

    @abstractmethod
    def run_select(self, handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        """Run a SELECT; returns (rows as dicts, truncated_flag)."""

    @abstractmethod
    def execute(self, handle, sql: str) -> int:
        """Run a statement, commit, return rowcount (-1 if unavailable)."""

    @abstractmethod
    def list_tables(self, handle) -> list[dict]:
        """[{"name": str, "comment": str}]"""

    @abstractmethod
    def describe_table(self, handle, table: str) -> dict:
        """{"columns": [{name,type,nullable,comment}],
        "constraints": [{"name","type"}],   # type: PRIMARY|UNIQUE|FOREIGN|CHECK
        "foreign_keys": [{"name","column","ref_table","ref_column"}]}"""

    @abstractmethod
    def explain(self, handle, sql: str) -> str:
        """Execution plan text for sql."""

    @abstractmethod
    def format_error(self, e: Exception) -> dict:
        """{"error": str, "code": str|None, "message": str, "offset": int|None}"""

    # ------------------------------------------------------------------
    # Transaction support (E4). A transaction pins a DEDICATED handle with
    # autocommit off — pool handles are never held for this (except Oracle,
    # whose session pool is the only connection source). Statements inside a
    # transaction NEVER commit implicitly; commit/rollback are explicit.
    # ------------------------------------------------------------------

    @abstractmethod
    def open_tx(self, handle, params: dict) -> Any:
        """Open a dedicated autocommit-off connection for a transaction.

        handle is the entry's pool/connection handle (Oracle acquires a
        session from it; Postgres/libsql ignore it and connect standalone).
        Returns the tx handle."""

    @abstractmethod
    def select_tx(self, tx_handle, sql: str, max_rows: int) -> tuple[list[dict], bool]:
        """Run a SELECT inside the transaction; (rows, truncated). No commit."""

    @abstractmethod
    def execute_tx(self, tx_handle, sql: str) -> int:
        """Run a statement inside the transaction WITHOUT committing; rowcount."""

    @abstractmethod
    def commit_tx(self, tx_handle) -> None:
        """Commit the transaction."""

    @abstractmethod
    def rollback_tx(self, tx_handle) -> None:
        """Roll back the transaction."""

    @abstractmethod
    def close_tx(self, handle, tx_handle) -> None:
        """Release the tx handle: rollback-if-open (harmless when clean),
        then close the connection / return the session to its pool."""
