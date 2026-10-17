"""
Tidy-state sidecar for memorymcp maintenance passes (P4,
plans/odysseus-reuse-2026-10-06.md).

A tiny JSON sidecar (~/.memorymcp/tidy_state.json) records the fingerprint of
the store contents at the last successful non-dry maintenance pass, so
expensive passes (O(n²) duplicate scan, TTL sweeps) short-circuit when
nothing changed. Patterns re-derived from odysseus's audit short-circuit and
over-deletion guard; zero code copied (AGPL).

Guards implemented here + in the callers:
- fingerprint skip: same fp + same pass parameters ⇒ nothing to do
- fail-closed scroll: store shrunk by more than the drop floor since the
  last pass ⇒ suspected degraded read; the destructive pass needs force=true
  (echo of odysseus's destroyed-store bug: a degraded read persisted back)
- over-deletion: a destructive pass deleting more than the drop floor of the
  current count is refused

Sidecar failures NEVER break a pass: load/clear swallow errors (fail open —
a missing or corrupt sidecar just means "run the pass"), save is best-effort.

Invariant (P4): every row-count-changing write path refreshes the state —
either by saving fresh state (the passes themselves) or by clearing it
(delete_memory, bulk imports). A stale sidecar otherwise trips the
fail-closed guard on a legitimate wipe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("tidy_state")

DEFAULT_STATE_PATH = Path.home() / ".memorymcp" / "tidy_state.json"


def state_path() -> Path:
    return Path(os.getenv("MEMORY_TIDY_STATE_PATH", str(DEFAULT_STATE_PATH)))


def current_fingerprint(points) -> str:
    """
    Cheap identity of the store contents: sha256 over the sorted
    'id|text_hash|last_accessed' lines. Accepts anything with .payload (or
    bare payload dicts). Deliberately vector-free — callers pass whatever
    they already have in memory.
    """
    parts = []
    for p in points:
        payload = p.payload if hasattr(p, "payload") else (p or {})
        pid = str(payload.get("id") or getattr(p, "id", ""))
        thash = str(payload.get("text_hash") or "")
        last = str(payload.get("last_accessed") or "")
        parts.append(f"{pid}|{thash}|{last}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


def params_fingerprint(**params) -> str:
    """Fingerprint of the pass parameters — a fp-skip is only valid when the
    parameters (thresholds, TTLs) match the saved pass."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_state() -> dict:
    """Saved state or {} — corrupt/missing sidecar fails open."""
    try:
        return json.loads(state_path().read_text())
    except Exception:
        return {}


def save_state(fp: str, count: int, params_fp: str | None = None) -> None:
    """Best-effort persist after a successful non-dry pass."""
    try:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "fp": fp,
            "count": count,
            "params_fp": params_fp,
            "ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    except Exception as e:
        logger.warning(f"Could not save tidy state: {e}")


def clear_state() -> None:
    """Row-count invariant: call after any other write that changes the
    memory count (delete_memory, bulk imports, direct wipes). The next pass
    then recomputes from scratch instead of tripping the fail-closed guard."""
    try:
        state_path().unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Could not clear tidy state: {e}")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_drop(previous_count: int) -> int:
    """
    Largest count change that is NOT suspicious: max(ABS, FRAC·previous) with
    MEMORY_TIDY_MAX_DROP_ABS (default 50) and MEMORY_TIDY_MAX_DROP_FRAC
    (default 0.05) — "abnormal if >50 rows OR >5% vanished", so healthy
    churn on a small store isn't blocked and a stale sidecar on a big store
    doesn't mask a real wipe.
    """
    abs_d = _env_int("MEMORY_TIDY_MAX_DROP_ABS", 50)
    frac = _env_float("MEMORY_TIDY_MAX_DROP_FRAC", 0.05)
    return max(abs_d, int(previous_count * frac))


def shrink_suspicious(previous_count: int, current_count: int) -> bool:
    """True when the store shrank by more than the drop floor since the last
    saved pass — a degraded/partial read is the likely cause."""
    if previous_count <= 0:
        return False
    return current_count < (previous_count - max_drop(previous_count))
