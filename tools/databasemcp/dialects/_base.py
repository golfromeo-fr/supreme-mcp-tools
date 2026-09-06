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
