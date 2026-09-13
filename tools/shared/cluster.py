"""M4 — cluster node registry + override mirroring (shared state plane).

Three documents in the shared backend (state_docs, db mode):

  cluster_nodes  {node_name: {"central_url": ..., "registered_at": ...}}
                 — who is in the cluster; enables runtime mask fan-out.
  env_overrides  {var_name: value}
                 — env vars set via a central API; ADOPTED BY EVERY NODE
                   AT BOOT (os.environ, before tool imports).
  auth_overrides {tool_name: {"api_key": ...}}
                 — tool auth sections set via the central API; adopted at
                   boot into the node's local tools/<name>/config.json
                   (before discovery, so verifiers see the new key).

Writes go through compare-and-swap on the doc's updated_at stamp with a
bounded retry (a lost race re-applies on the winner's copy); reads are
fresh. In json mode every helper is a no-op (single-node behavior
unchanged).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DOC_NODES = "cluster_nodes"
DOC_ENV = "env_overrides"
DOC_AUTH = "auth_overrides"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def is_db_mode() -> bool:
    """True when the shared db state plane is selected. Delegates to
    state_docs so the MCP_STATE_BACKEND gate lives in exactly one place."""
    from tools.shared import state_docs

    return state_docs.is_db_mode()


_CAS_ATTEMPTS = 5
_CAS_RETRY_PAUSE_S = 0.05


def _update_doc(name: str, mutate) -> bool:
    """Read-mutate-write one state doc under optimistic concurrency: the
    write is a compare-and-swap on the doc's updated_at stamp and a lost
    race retries on the winner's fresh copy (bounded — two nodes registering
    at once both survive instead of one being silently dropped). False =
    json mode/unavailable, or the mutate hook declined."""
    from tools.shared import state_docs

    if not state_docs.is_db_mode():
        return False
    for attempt in range(_CAS_ATTEMPTS):
        doc, version = state_docs.load_doc_versioned(name)
        if doc is None:
            doc = {}
        result = mutate(doc)
        if result is False:
            return False
        if state_docs.save_doc_cas(name, doc, version):
            return True
        if attempt + 1 < _CAS_ATTEMPTS:
            time.sleep(_CAS_RETRY_PAUSE_S)
    logger.warning(
        f"[M4] state doc '{name}' write lost after {_CAS_ATTEMPTS} "
        "concurrent-write attempts - caller falls back to local")
    return False


# ---------------------------------------------------------------------------
# node registry
# ---------------------------------------------------------------------------

def register_node(name: str, central_url: str) -> bool:
    """Add/update this node in the cluster registry (idempotent). Boot-time
    registration also CLEARS any unreachable mark — a node that comes back
    rejoins the fan-out set without manual cleanup."""
    def _mutate(doc: dict) -> bool:
        entry = doc.get(name) if isinstance(doc.get(name), dict) else {}
        entry.update({"central_url": central_url,
                      "registered_at": _now()})
        entry.pop("unreachable_since", None)
        doc[name] = entry
        return True
    return _update_doc(DOC_NODES, _mutate)


def sibling_nodes(exclude_name: str | None = None,
                  include_unreachable: bool = False) -> dict[str, str]:
    """{node_name: central_url} for every OTHER registered node. Nodes
    marked unreachable (fan-out failures) are excluded unless asked —
    they rejoin via their next boot registration."""
    from tools.shared import state_docs

    doc = state_docs.load_doc(DOC_NODES) or {}
    out = {}
    for n, e in doc.items():
        if n == exclude_name or not isinstance(e, dict):
            continue
        if not e.get("central_url"):
            continue
        if not include_unreachable and e.get("unreachable_since"):
            continue
        out[n] = e["central_url"]
    return out


def mark_unreachable(name: str) -> bool:
    """Stamp first-failure time on a node (kept in the registry for
    visibility; excluded from fan-out until it re-registers)."""
    def _mutate(doc: dict) -> bool:
        entry = doc.get(name)
        if not isinstance(entry, dict):
            return False  # unknown/gone — nothing to mark
        entry.setdefault("unreachable_since", _now())
        return True
    return _update_doc(DOC_NODES, _mutate)


# ---------------------------------------------------------------------------
# override mirroring (central mutation → boot adoption)
# ---------------------------------------------------------------------------

def mirror_env(var_name: str, value: str | None) -> bool:
    """Mirror an env-var mutation; value=None records a deletion."""
    def _mutate(doc: dict) -> bool:
        if value is None:
            doc.pop(var_name, None)
        else:
            doc[var_name] = value
        return True
    return _update_doc(DOC_ENV, _mutate)


def mirror_auth(tool_name: str, api_key: str) -> bool:
    def _mutate(doc: dict) -> bool:
        doc[tool_name] = {"api_key": api_key}
        return True
    return _update_doc(DOC_AUTH, _mutate)


def adopt_env_overrides(env: dict[str, str] | None = None) -> list[str]:
    """Apply env_overrides to os.environ (before tool imports). Returns
    the applied var names. Values already correct are left alone."""
    from tools.shared import state_docs

    doc = state_docs.load_doc(DOC_ENV)
    target = env if env is not None else os.environ
    applied = []
    for var, value in (doc or {}).items():
        if target.get(var) != value:
            target[var] = value
            applied.append(var)
    if applied:
        logger.warning(f"[M4] adopted {len(applied)} env override(s) from the "
                       f"cluster state: {sorted(applied)}")
    return applied


def adopt_auth_overrides(tools_dir: str | Path | None = None) -> list[str]:
    """Apply auth_overrides into tools/<name>/config.json (before
    discovery, so verifiers resolve the mirrored keys). Returns the
    tools updated. Container note: writes to the container layer — the
    shared doc stays authoritative and is re-applied every boot."""
    from tools.shared import state_docs
    from tools.shared.atomic_io import atomic_write_text

    doc = state_docs.load_doc(DOC_AUTH)
    logger.debug(f"[M4] auth adoption: doc keys={sorted((doc or {}).keys())}")
    if not doc:
        return []
    # cluster.py lives at <tools>/shared/cluster.py → parents[1] IS tools/
    base = Path(tools_dir) if tools_dir else Path(__file__).resolve().parents[1]
    updated = []
    for tool, auth in doc.items():
        cfg_path = base / tool / "config.json"
        logger.debug(
            f"[M4] auth adoption: tool={tool} cfg={cfg_path} "
            f"exists={cfg_path.exists()}")
        if not cfg_path.exists() or not isinstance(auth, dict):
            continue
        try:
            config = json.loads(cfg_path.read_text())
            if config.get("auth", {}).get("api_key") == auth.get("api_key"):
                continue
            config.setdefault("auth", {})["api_key"] = auth["api_key"]
            # flock+tmp+replace: config.json has several concurrent writers
            # (launcher, central API, mcp_ui) — a plain truncate-write can
            # corrupt it mid-crash (see tools/shared/atomic_io.py).
            atomic_write_text(cfg_path, json.dumps(config, indent=2) + "\n")
            updated.append(tool)
            logger.warning(f"[M4] adopted cluster auth override for '{tool}'")
        except Exception as e:
            logger.warning(f"[M4] auth adoption failed for '{tool}': {e}")
    return updated


def adopt_all() -> dict[str, list[str]]:
    """Boot hook: apply every override family. Best-effort."""
    applied: dict[str, list[str]] = {"env": [], "auth": []}
    try:
        applied["env"] = adopt_env_overrides()
    except Exception as e:
        logger.warning(f"[M4] env adoption failed ({type(e).__name__}: {e})")
    try:
        applied["auth"] = adopt_auth_overrides()
    except Exception as e:
        logger.warning(f"[M4] auth adoption failed ({type(e).__name__}: {e})")
    return applied
