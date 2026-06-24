#!/usr/bin/env python3
"""
Migration tool for moving data between backend combos.

Usage:
  python -m tools.shared.migrate_store export --out backup.jsonl
  python -m tools.shared.migrate_store import --in backup.jsonl
  python -m tools.shared.migrate_store pipe --from postgres+qdrant --to turso+turso
  python -m tools.shared.migrate_store verify --left postgres+qdrant --right turso+turso

Phase 4 of the backend abstraction plan.

R2: Filters __collection_metadata__ from vector points on import.
R3: Uses make_sql_store() / make_vector_store() public APIs.
R9: Supports --progress-every for progress reporting.
R11: Includes schema_version in JSONL meta header.
R12: Supports --backup-before for safety.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_PROGRESS_INTERVAL = 1000  # default progress report interval


# ---------------------------------------------------------------------------
# Backend resolution (R3: delegates to make_*_store public APIs)
# ---------------------------------------------------------------------------

def _resolve_backend(spec: str | None) -> tuple:
    """
    Parse a backend combo spec like 'postgres+qdrant' or 'turso+turso'.
    If spec is None, resolves from current config (env vars).
    Returns (sql_store, vector_store) instances.
    """
    if spec is None:
        from shared.store_factory import resolve_sql_backend, resolve_vector_backend
        sql_cfg = resolve_sql_backend()
        vec_cfg = resolve_vector_backend()
        sql_name = sql_cfg["name"] if sql_cfg else "none"
        vec_name = vec_cfg["name"] if vec_cfg else "none"
        spec = f"{sql_name}+{vec_name}"

    parts = spec.split("+", 1)
    sql_name = parts[0] if len(parts) > 0 else "none"
    vec_name = parts[1] if len(parts) > 1 else "none"

    sql_store = None
    vector_store = None

    if sql_name and sql_name != "none":
        from shared.sql_store import make_sql_store
        config = {}
        if sql_name == "turso":
            config["url"] = os.getenv("TURSO_DATABASE_URL", "file::memory:")
            config["auth_token"] = os.getenv("TURSO_AUTH_TOKEN")
        elif sql_name == "postgres":
            from shared.store_factory import pg_dsn_from_env
            config["dsn"] = pg_dsn_from_env()
        sql_store = make_sql_store(sql_name, config)

    if vec_name and vec_name != "none":
        from shared.vector_store import make_vector_store
        config = {}
        if vec_name == "turso":
            config["url"] = os.getenv("TURSO_DATABASE_URL", "file::memory:")
            config["auth_token"] = os.getenv("TURSO_AUTH_TOKEN")
        elif vec_name == "postgres":
            from shared.store_factory import pg_dsn_from_env
            config["dsn"] = pg_dsn_from_env()
        elif vec_name == "qdrant":
            config["host"] = os.getenv("QDRANT_HOST", "qdrant")
            config["port"] = int(os.getenv("QDRANT_PORT", "6333"))
        vector_store = make_vector_store(vec_name, config)

    return sql_store, vector_store


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_export(args):
    """Export from specified backends to a JSONL file."""
    sql_store, vector_store = _resolve_backend(args.backend)

    progress_every = args.progress_every or _PROGRESS_INTERVAL
    count = 0

    with open(args.out, "w") as f:
        # SQL section
        if sql_store and sql_store.is_available:
            f.write(json.dumps({"_meta": {
                "kind": "sql",
                "schema_version": SCHEMA_VERSION,  # R11
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source": args.backend.split("+")[0] if "+" in args.backend else args.backend,
            }}) + "\n")
            for row in sql_store.iter_all():
                f.write(json.dumps({"_sql": row}, default=str) + "\n")
                count += 1
                if count % progress_every == 0:
                    logger.info(f"Exported {count} SQL records...")
            logger.info(f"SQL export complete: {count} records")

        # Vector section (per collection)
        if vector_store:
            vec_count = 0
            for coll_name in vector_store.list_collections():
                info = vector_store.get_collection(coll_name)
                f.write(json.dumps({"_meta": {
                    "kind": "vector",
                    "collection": coll_name,
                    "dim": info.dim,
                    "has_sparse": info.has_sparse,
                    "distance": info.distance,
                    "schema_version": SCHEMA_VERSION,
                }}) + "\n")

                for point in vector_store.iter_all(coll_name, with_vectors=True):
                    # R2: skip __collection_metadata__ — it's in the header
                    if str(point.id) == "__collection_metadata__":
                        continue
                    f.write(json.dumps({
                        "_vec": coll_name,
                        "id": str(point.id),
                        "vector": point.vector if isinstance(point.vector, list) else None,
                        "payload": point.payload,
                    }, default=str) + "\n")
                    vec_count += 1
                    if vec_count % progress_every == 0:
                        logger.info(f"Exported {vec_count} vector records...")
            logger.info(f"Vector export complete: {vec_count} records")

    logger.info(f"Export written to {args.out}")


def cmd_import(args):
    """Import from a JSONL file into specified backends."""
    sql_store, vector_store = _resolve_backend(args.backend)

    # R12: backup before import if requested
    if args.backup_before and (sql_store or vector_store):
        backup_path = f"backup_before_import_{int(time.time())}.jsonl"
        logger.info(f"Creating backup: {backup_path}")
        backup_args = argparse.Namespace(
            backend=args.backend, out=backup_path, progress_every=None,
        )
        cmd_export(backup_args)
        logger.info(f"Backup saved to {backup_path}")

    progress_every = args.progress_every or _PROGRESS_INTERVAL
    collection_metas: dict[str, dict] = {}
    sql_rows = []
    vec_points: dict[str, list] = {}

    with open(getattr(args, "in")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_meta" in record:
                meta = record["_meta"]
                if meta["kind"] == "vector":
                    collection_metas[meta["collection"]] = meta
                continue
            if "_sql" in record:
                sql_rows.append(record["_sql"])
            elif "_vec" in record:
                coll = record["_vec"]
                vec_points.setdefault(coll, []).append(record)

    # Bulk insert SQL
    if sql_store and sql_store.is_available and sql_rows:
        count = 0
        for row in sql_rows:
            sql_store.upsert_memory(
                memory_id=row.get("id", ""),
                text=row.get("text", ""),
                memory_type=row.get("memory_type", "concept"),
                source=row.get("source", "agent_action"),
                tags=row.get("tags", []) if isinstance(row.get("tags"), list) else [],
                path=row.get("path"),
                commit=row.get("commit"),
                agent_id=row.get("agent_id"),
                sensitivity=row.get("sensitivity", "low"),
                retention_policy=row.get("retention_policy", "auto-delete"),
            )
            count += 1
            if count % progress_every == 0:
                logger.info(f"Imported {count} SQL records...")
        logger.info(f"SQL import complete: {count} records")

    # Bulk insert vectors
    if vector_store:
        from shared.store_models import PointStruct
        for coll_name, points in vec_points.items():
            meta = collection_metas.get(coll_name, {})
            vector_store.ensure_collection(
                coll_name,
                dense_dim=meta.get("dim"),
                sparse=meta.get("has_sparse", False),
                distance=meta.get("distance", "Cosine"),
            )
            structs = []
            for p in points:
                vec = p.get("vector")
                if vec and isinstance(vec, str):
                    try:
                        vec = [float(x) for x in vec.strip("[]").split(",")]
                    except Exception:
                        vec = []
                structs.append(PointStruct(
                    id=p["id"],
                    vector=vec or [],
                    payload=p.get("payload"),
                ))
            count = vector_store.bulk_upsert(coll_name, structs)
            logger.info(f"Imported {count} vectors into '{coll_name}'")


def cmd_verify(args):
    """Verify two backends hold the same data."""
    left_sql, left_vec = _resolve_backend(args.left)
    right_sql, right_vec = _resolve_backend(args.right)

    issues = []

    # SQL parity
    if left_sql and right_sql:
        left_ids = set(left_sql.get_all_memory_ids()) if left_sql.is_available else set()
        right_ids = set(right_sql.get_all_memory_ids()) if right_sql.is_available else set()
        if left_ids != right_ids:
            missing = left_ids - right_ids
            extra = right_ids - left_ids
            if missing:
                issues.append(f"SQL: {len(missing)} IDs in left but not right")
            if extra:
                issues.append(f"SQL: {len(extra)} IDs in right but not left")
        else:
            logger.info(f"SQL parity OK: {len(left_ids)} matching IDs")

    # Vector parity
    if left_vec and right_vec:
        left_colls = set(left_vec.list_collections())
        right_colls = set(right_vec.list_collections())
        if left_colls != right_colls:
            issues.append(f"Vector: collections differ: {left_colls ^ right_colls}")
        else:
            logger.info(f"Vector collections match: {left_colls}")
        for coll in left_colls & right_colls:
            li = left_vec.get_collection(coll)
            ri = right_vec.get_collection(coll)
            if li.points_count != ri.points_count:
                issues.append(
                    f"Vector '{coll}': point count {li.points_count} vs {ri.points_count}"
                )
            else:
                logger.info(f"Vector '{coll}': {li.points_count} points match")

    if issues:
        print("VERIFICATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("VERIFICATION PASSED: backends are in parity")


def cmd_pipe(args):
    """Export from source, import into destination (no intermediate file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info(f"Piping {args.from_backend} → {args.to_backend}")
        # Export
        export_args = argparse.Namespace(
            backend=args.from_backend, out=tmp_path, progress_every=args.progress_every,
        )
        cmd_export(export_args)
        # Import
        import_args = argparse.Namespace(
            backend=args.to_backend, **{"in": tmp_path},
            progress_every=args.progress_every, backup_before=args.backup_before,
        )
        cmd_import(import_args)
        logger.info(f"Pipe complete: {args.from_backend} → {args.to_backend}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backend migration tool for memorymcp/ragmcp"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    p_export = sub.add_parser("export", help="Export to JSONL")
    p_export.add_argument("--out", required=True, help="Output JSONL file path")
    p_export.add_argument(
        "--backend", default=None,
        help='Backend combo, e.g. "postgres+qdrant" (default: current config)',
    )
    p_export.add_argument(
        "--progress-every", type=int, default=None,
        help="Print progress every N records (R9)",
    )

    p_import = sub.add_parser("import", help="Import from JSONL")
    p_import.add_argument("--in", dest="in", required=True, help="Input JSONL file path")
    p_import.add_argument(
        "--backend", default=None,
        help='Target backend combo, e.g. "turso+turso" (default: current config)',
    )
    p_import.add_argument("--progress-every", type=int, default=None)
    p_import.add_argument(
        "--backup-before", action="store_true",
        help="Export destination to a backup file before importing (R12)",
    )

    p_pipe = sub.add_parser("pipe", help="Export → import without intermediate file")
    p_pipe.add_argument("--from", dest="from_backend", required=True)
    p_pipe.add_argument("--to", dest="to_backend", required=True)
    p_pipe.add_argument("--progress-every", type=int, default=None)
    p_pipe.add_argument("--backup-before", action="store_true")

    p_verify = sub.add_parser("verify", help="Verify two backends hold the same data")
    p_verify.add_argument("--left", required=True, help='e.g. "postgres+qdrant"')
    p_verify.add_argument("--right", required=True, help='e.g. "turso+turso"')

    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Default backend from config/env
    if hasattr(args, "backend") and args.backend is None:
        from shared.store_factory import resolve_sql_backend, resolve_vector_backend
        sql_cfg = resolve_sql_backend()
        vec_cfg = resolve_vector_backend()
        sql_name = sql_cfg["name"] if sql_cfg else "none"
        vec_name = vec_cfg["name"] if vec_cfg else "none"
        args.backend = f"{sql_name}+{vec_name}"

    if args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "pipe":
        cmd_pipe(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
