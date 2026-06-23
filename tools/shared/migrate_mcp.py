"""
MCP tool wrappers for the migration tool.

Registered on the shared FastMCP instance during tool startup.
Provides migrateMemoryBackend and verifyBackendParity MCP tools.

Phase 4 of the backend abstraction plan.
"""
from __future__ import annotations

import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def register_migrate_tools(mcp):
    """Register migration MCP tools on the given FastMCP instance."""

    @mcp.tool()
    async def migrateMemoryBackend(
        export_backend: str = "auto",
        import_backend: str = "auto",
        file_path: str | None = None,
    ) -> str:
        """
        Migrate memory data from one backend to another.

        Args:
            export_backend: Source combo, e.g. "postgres+qdrant" (default: current config)
            import_backend: Target combo, e.g. "turso+turso" (default: current config)
            file_path: Optional intermediate JSONL file. If omitted, pipes directly.

        Returns:
            Migration summary (rows + vectors moved)
        """
        import argparse
        from shared.migrate_store import cmd_export, cmd_import, _resolve_backend

        # Resolve "auto" to current config
        if export_backend == "auto" or import_backend == "auto":
            from shared.store_factory import resolve_sql_backend, resolve_vector_backend
            sql_cfg = resolve_sql_backend()
            vec_cfg = resolve_vector_backend()
            sql_name = sql_cfg["name"] if sql_cfg else "none"
            vec_name = vec_cfg["name"] if vec_cfg else "none"
            combo = f"{sql_name}+{vec_name}"
            if export_backend == "auto":
                export_backend = combo
            if import_backend == "auto":
                import_backend = combo

        if file_path:
            # Export to file, then import
            cmd_export(argparse.Namespace(
                backend=export_backend, out=file_path, progress_every=None,
            ))
            cmd_import(argparse.Namespace(
                backend=import_backend, **{"in": file_path},
                progress_every=None, backup_before=False,
            ))
        else:
            # Pipe directly via temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as tmp:
                tmp_path = tmp.name
            try:
                cmd_export(argparse.Namespace(
                    backend=export_backend, out=tmp_path, progress_every=None,
                ))
                cmd_import(argparse.Namespace(
                    backend=import_backend, **{"in": tmp_path},
                    progress_every=None, backup_before=False,
                ))
            finally:
                os.unlink(tmp_path)

        return f"Migration complete: {export_backend} -> {import_backend}"

    @mcp.tool()
    async def verifyBackendParity(
        left_backend: str,
        right_backend: str,
    ) -> str:
        """
        Verify two backend combos hold the same data.

        Args:
            left_backend: e.g. "postgres+qdrant"
            right_backend: e.g. "turso+turso"

        Returns:
            Verification result (pass/fail with details)
        """
        from shared.migrate_store import _resolve_backend

        left_sql, left_vec = _resolve_backend(left_backend)
        right_sql, right_vec = _resolve_backend(right_backend)

        issues = []

        if left_sql and right_sql:
            left_ids = set(left_sql.get_all_memory_ids()) if left_sql.is_available else set()
            right_ids = set(right_sql.get_all_memory_ids()) if right_sql.is_available else set()
            if left_ids != right_ids:
                missing = left_ids - right_ids
                extra = right_ids - left_ids
                if missing:
                    issues.append(f"{len(missing)} IDs in left but not right")
                if extra:
                    issues.append(f"{len(extra)} IDs in right but not left")

        if left_vec and right_vec:
            left_colls = set(left_vec.list_collections())
            right_colls = set(right_vec.list_collections())
            if left_colls != right_colls:
                issues.append(f"Collections differ: {left_colls ^ right_colls}")
            for coll in left_colls & right_colls:
                li = left_vec.get_collection(coll)
                ri = right_vec.get_collection(coll)
                if li.points_count != ri.points_count:
                    issues.append(
                        f"Collection '{coll}': {li.points_count} vs {ri.points_count} points"
                    )

        if issues:
            return f"FAILED:\n" + "\n".join(f"  - {i}" for i in issues)
        return "PASSED: backends are in parity"
