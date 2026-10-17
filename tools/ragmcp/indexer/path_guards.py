"""Dependency-free path/name guards for ragmcp (audit 2026-10-01 P2-4, P2-5).

Kept free of FastMCP/store imports so tests can exercise the real logic
without triggering ragmcp_fastmcp's import-time backend initialization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# The collection name reaches FILE paths (the indexer child log) as well as
# SQL identifiers (the store impls sanitize those, the file path does not) —
# reject path separators and dot-segments outright (audit P2-5).
_COLLECTION_NAME_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def validate_collection_name(collection_name: str) -> str | None:
    """Return an error string for an unsafe collection name, else None.

    'ALL' (clear-everything verb) is a valid name here; the tools give it its
    special meaning downstream."""
    if not collection_name or not _COLLECTION_NAME_SAFE.match(collection_name) \
            or ".." in collection_name:
        return (f"Error: Invalid collection name {collection_name!r}. Allowed: "
                "letters, digits, '.', '_', '-' (max 120 chars, no path separators).")
    return None


def allowed_roots_for(config_path: Path) -> list[str]:
    """Directories ragmcp is allowed to index (audit P2-4).

    Read from the tool config's `allowed_roots` at call time; the default
    mirrors convertermcp's gate (the repo itself + /tmp) so existing usage
    keeps working unchanged."""
    default = [str(config_path.parent.parent), "/tmp"]
    try:
        cfg = json.loads(Path(config_path).read_text())
        roots = cfg.get("allowed_roots")
        return roots if isinstance(roots, list) and roots else default
    except Exception:
        return default


def workspace_allowed(workspace_root: str, allowed_roots: list[str]) -> str | None:
    """Return an error string when the resolved workspace is outside
    allowed_roots, else None."""
    try:
        root = Path(workspace_root).expanduser().resolve()
    except Exception:
        return f"Error: Cannot resolve workspace_root {workspace_root!r}."
    for a in allowed_roots:
        try:
            allowed = Path(a).expanduser().resolve()
        except Exception:
            continue
        if root == allowed or root.is_relative_to(allowed):
            return None
    return (f"Error: workspace_root resolves to {str(root)!r}, which is outside "
            f"ragmcp's allowed_roots ({', '.join(allowed_roots)}). "
            "Add the directory to allowed_roots in tools/ragmcp/config.json to index it.")
