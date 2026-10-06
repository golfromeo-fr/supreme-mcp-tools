"""
Embedding lanes — per-model collection naming for ragmcp.

A "lane" is one vector collection per embedding fingerprint
(provider/model/dim). Switching embedding models never mixes dimensions
inside one collection: each preset writes to (and reads from) its own
``<base>-<slug>`` lane, and the pre-existing unsuffixed collection stays
searchable as the "legacy" lane when its fingerprint matches the active
embedder. Sparse-only indexing is model-independent and keeps using the
plain base collection.

Lane naming lives in the CALLER (ragmcp tools / indexer CLI). VectorStore
implementations stay lane-agnostic — a lane is just a collection name.

Pure stdlib so the logic is unit-testable without importing the FastMCP
server module. The fingerprint source of truth is the ``__collection_metadata__``
point the indexer writes at collection creation (provider/model/dimensions);
slug parsing is never needed to read a lane's config.

Design: plans/odysseus-reuse-2026-10-06.md P2 — pattern re-derived from
odysseus's embedding-lane idea (per-model collections + fingerprint),
zero code taken (AGPL).
"""
from __future__ import annotations

import hashlib
import re

# The indexer stores collection provenance under this point id
# (incremental_indexer.main -> payload {"embedding_provider", "embedding_model",
# "embedding_dimensions", "_is_metadata": True}).
METADATA_POINT_ID = "__collection_metadata__"

_SLUG_FORBIDDEN = re.compile(r"[^a-z0-9_-]+")
_SLUG_MAX = 60


def sanitize_slug(text: str) -> str:
    """Lowercase and map every char outside [a-z0-9_-] to '-'."""
    return _SLUG_FORBIDDEN.sub("-", text.lower()).strip("-")


def lane_fingerprint(provider: str, model: str, dim: int | None) -> str:
    """Stable fingerprint string for an embedding config: provider:model:dim."""
    return (
        f"{(provider or 'unknown').lower()}:"
        f"{model or 'unknown'}:"
        f"{dim if dim else 'unknown'}"
    )


def lane_slug(provider: str, model: str, dim: int | None) -> str:
    """
    Provider-prefixed lane slug, sanitized to ^[a-z0-9_-]{1,60}$.

    Model names carry '/', ':', uppercase (BAAI/bge-m3, openrouter/foo:free),
    so everything outside the allowed set maps to '-'. On overflow an 8-char
    hash suffix replaces the tail instead of truncating — truncated prefixes
    collide.
    """
    raw = sanitize_slug(f"{provider or 'unknown'}-{model or 'unknown'}-{dim if dim else 'x'}")
    if len(raw) <= _SLUG_MAX:
        return raw
    suffix = hashlib.sha256(lane_fingerprint(provider, model, dim).encode()).hexdigest()[:8]
    keep = _SLUG_MAX - len(suffix) - 1
    return f"{raw[:keep]}-{suffix}"


def resolve_lane(base: str, provider: str, model: str, dim: int | None) -> str:
    """Full lane collection name for a base collection + embedding config."""
    return f"{base}-{lane_slug(provider, model, dim)}"


def legacy_lane(base: str) -> str:
    """The pre-lane collection name (the plain base)."""
    return base


def is_lane_name(base: str, name: str) -> bool:
    """True if `name` belongs to `base`: the base itself or any <base>-<slug>."""
    return name == base or name.startswith(f"{base}-")


def lane_label(base: str, collection: str) -> str:
    """Human label for a collection of `base`: 'legacy' or the slug part."""
    if collection == base:
        return "legacy"
    return collection[len(base) + 1:]


def list_lanes(vector_store, base: str) -> list[str]:
    """
    All collections belonging to `base`: the base itself (legacy lane, may
    not exist) plus every ``<base>-<slug>`` collection. Legacy first, lanes
    sorted after. Returns [] when the store is unavailable.
    """
    try:
        collections = vector_store.list_collections()
    except Exception:
        return []
    return sorted(
        (c for c in collections if is_lane_name(base, c)),
        key=lambda c: (c != base, c),
    )


def dense_dim_of(collection_info) -> int | None:
    """Dense vector dimension from a CollectionInfo ('dense' named vector first)."""
    named = getattr(collection_info, "named_vectors", None) or {}
    if "dense" in named:
        return named["dense"]
    return getattr(collection_info, "dim", None)


def collection_fingerprint(vector_store, collection: str) -> dict | None:
    """
    Read the indexer's __collection_metadata__ point for a collection.

    Returns {'provider', 'model', 'dimensions'} or None when absent
    (legacy collections created before metadata was written, or the point
    was filtered out on import).
    """
    try:
        points = vector_store.retrieve(collection, [METADATA_POINT_ID], with_payload=True)
    except Exception:
        return None
    if not points:
        return None
    payload = points[0].payload or {}
    if not payload.get("_is_metadata"):
        return None
    return {
        "provider": payload.get("embedding_provider"),
        "model": payload.get("embedding_model"),
        "dimensions": payload.get("embedding_dimensions"),
    }


def fingerprint_matches(fp: dict | None, provider: str, model: str, dim: int | None) -> bool:
    """True when a collection fingerprint equals the given embedder config."""
    if fp is None:
        return False
    return (
        (fp.get("provider") or "").lower() == (provider or "").lower()
        and fp.get("model") == model
        and fp.get("dimensions") == dim
    )


def matches_active_embedder(vector_store, collection: str, active: dict) -> bool:
    """
    True when `collection` was built by the same embedder as `active`
    ({'provider','model','dimensions'}). Falls back to a dimension-only
    match when the collection carries no metadata point — dim equality is
    the operational proxy for 'same embedding space' without provenance.
    """
    fp = collection_fingerprint(vector_store, collection)
    if fp is not None:
        return fingerprint_matches(fp, active.get("provider"), active.get("model"), active.get("dimensions"))
    try:
        dims = dense_dim_of(vector_store.get_collection(collection))
    except Exception:
        return False
    return bool(dims) and dims == active.get("dimensions")


def check_lane_compatibility(vector_store, target: str, provider: str, model: str, dim: int | None) -> str | None:
    """
    Pre-flight for indexing into `target`: None when safe to proceed, else a
    user-facing error message. Aborts on a dimension mismatch BEFORE any
    embedding work — ensure_collection never re-validates an existing
    collection, so without this the failure would surface mid-run at the
    first upsert.
    """
    try:
        info = vector_store.get_collection(target)
    except Exception as e:
        if "Not found" in str(e) or "doesn't exist" in str(e):
            return None  # fresh lane — will be created with the right dim
        return f"Error checking collection '{target}': {e}"
    existing_dims = dense_dim_of(info)
    if not existing_dims:
        return (
            f"Error: collection '{target}' has no dense vectors configured — "
            "it was created sparse-only. Dense indexing needs a new lane "
            "(a different base collection name)."
        )
    if dim is not None and existing_dims != dim:
        fp = collection_fingerprint(vector_store, target)
        provenance = f" (provider={fp['provider']}, model={fp['model']})" if fp else ""
        return (
            f"Error: Embedding dimension mismatch (pre-flight)\n\n"
            f"Collection `{target}` has {existing_dims}d vectors{provenance}, but the "
            f"requested embedding config produces {dim}d "
            f"(provider={provider}, model={model}). Indexing would fail at write time — "
            f"every point must match the collection's fixed vector size.\n\n"
            f"Options:\n"
            f"1. Index this base with the matching embedding preset (each fingerprint "
            f"gets its own lane: `<base>-<provider-model-dim>`).\n"
            f"2. Use a different base collection name for the new model.\n"
            f"3. Deliberately replacing the index: clear_index(collection_name=\"{target}\", "
            f"confirm=true) first, then reindex."
        )
    return None


def resolve_search_candidates(
    vector_store,
    base: str,
    *,
    mode: str,
    active: dict,
    search_all_lanes: bool = False,
) -> dict:
    """
    Which collections to search for `base`, per the P2 cost model:

    - sparse mode: model-independent BM25 → every collection of the base
      (embedding cost is zero).
    - default dense/hybrid: the ACTIVE lane only, plus the legacy base when
      its fingerprint (or dim, without provenance) matches the active
      embedder — one query embedding total.
    - search_all_lanes=True (opt-in): every collection, embedding the query
      once per distinct fingerprint — the caller pays per-model cost.

    Returns {'candidates': [(collection, label, preset_or_None)], 'notes': [...]}.
    Preset dicts carry provider/model/dimensions so the caller can embed the
    query with each candidate's own model. Collections that don't exist are
    skipped with a note; the caller decides what "nothing found" looks like.
    """
    notes: list[str] = []
    candidates: list[tuple[str, str, dict | None]] = []

    def _exists(c: str) -> bool:
        try:
            vector_store.get_collection(c)
            return True
        except Exception:
            return False

    collections = list_lanes(vector_store, base)
    if not collections:
        return {"candidates": [], "notes": [f"no collections exist for base '{base}'"]}

    if mode == "sparse":
        for c in collections:
            candidates.append((c, lane_label(base, c), None))
        return {"candidates": candidates, "notes": notes}

    if search_all_lanes:
        for c in collections:
            if not _exists(c):
                notes.append(f"'{c}' does not exist — skipped")
                continue
            if c == base:
                if matches_active_embedder(vector_store, c, active):
                    candidates.append((c, lane_label(base, c), active))
                else:
                    fp = collection_fingerprint(vector_store, c)
                    notes.append(
                        f"legacy '{c}' skipped: fingerprint "
                        f"{fp or 'unknown'} does not match active embedder "
                        f"{lane_fingerprint(active.get('provider'), active.get('model'), active.get('dimensions'))}"
                    )
                continue
            fp = collection_fingerprint(vector_store, c)
            if fp and fp.get("provider") and fp.get("model") and fp.get("dimensions"):
                candidates.append((c, lane_label(base, c), dict(fp)))
            else:
                notes.append(f"lane '{c}' skipped: no collection metadata (provider/model/dims unknown)")
        return {"candidates": candidates, "notes": notes}

    # Default: active lane + legacy base when it matches the active embedder.
    active_lane = resolve_lane(base, active.get("provider"), active.get("model"), active.get("dimensions"))
    if _exists(active_lane):
        candidates.append((active_lane, lane_label(base, active_lane), active))
    else:
        notes.append(f"active lane '{active_lane}' does not exist yet")
    if _exists(base):
        if matches_active_embedder(vector_store, base, active):
            candidates.append((base, lane_label(base, base), active))
        else:
            fp = collection_fingerprint(vector_store, base)
            notes.append(
                f"legacy '{base}' skipped: fingerprint "
                f"{fp or 'unknown'} does not match active embedder "
                f"{lane_fingerprint(active.get('provider'), active.get('model'), active.get('dimensions'))}"
            )
    return {"candidates": candidates, "notes": notes}
