#!/usr/bin/env python3
"""
RAG MCP Server - FastMCP Implementation
Provides tools for semantic code search, sparse search, and code indexing using Qdrant.

FEF V3 Integration preserved from original implementation.
"""
import sys
import os
import logging
import re
import time
import asyncio
import functools
import subprocess
import psutil
import json
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

# Check for required dependencies before importing
try:
    import anyio
    from dotenv import load_dotenv
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure the virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Ensure the tool's directory is on sys.path for sibling imports
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Import optional dependencies
try:
    from indexer.sparse_vector_gen import generate_sparse_vector, generate_query_vector, get_global_generator
    SPARSE_VECTORS_AVAILABLE = True
except ImportError:
    SPARSE_VECTORS_AVAILABLE = False
    logging.warning("Sparse vector generator not available. Install sparse_vector_gen.py to enable sparse search.")

# P2 embedding lanes + dead-endpoint latch (stdlib-only modules, core deps)
from indexer.lanes import (
    lane_slug, lane_fingerprint, resolve_lane, list_lanes, lane_label,
    dense_dim_of, collection_fingerprint, matches_active_embedder,
    check_lane_compatibility, resolve_search_candidates,
)
from indexer import embedding_gate

try:
    from copilot_context_injector import CopilotContextInjector, get_injector
    COPILOT_INJECTOR_AVAILABLE = True
except ImportError:
    COPILOT_INJECTOR_AVAILABLE = False
    logging.warning("Copilot context injector not available. Install copilot_context_injector.py for context injection.")

# Local embedding models
LOCAL_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
    logging.info("sentence-transformers available for local embeddings")
except ImportError:
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Configure logging
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "ragmcp.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ragmcp")


# ============================================================================
# E3.5 — per-user collection access (identity from the request's auth key)
# ============================================================================

def _caller_rag_access() -> tuple[str | None, set[str] | None]:
    """(client_id, allowed_collections) for the in-flight caller.

    allowed_collections=None means UNRESTRICTED (admin role, or identity not
    applicable: mono mode / in-memory). Anomaly returns are fail-open for the
    mono DualHeaderVerifier (matching the E3 gate's mono behavior) and fail
    CLOSED for the user-store verifier — an installed user-store with an
    unresolvable identity must not grant the whole collection space
    (audit 2026-10-01 P2-3).

    The discriminator is the verifier's is_user_store marker (both shapes
    expose callable .tokens(), so the attribute shape cannot tell them
    apart; and MCP_AUTH_MODE in the env cannot either — the tool modules
    load the root .env at import, so a test or tool process can run with
    multi in the env while the verifier under test is the mono one)."""
    verifier = getattr(mcp, "auth", None)
    get_tokens = getattr(verifier, "tokens", None)
    user_store = bool(getattr(verifier, "is_user_store", False))
    if get_tokens is None:
        return None, None
    try:
        from fastmcp.server.dependencies import get_http_request
        request = get_http_request()
        auth = (request.headers.get("authorization", "") or "")
        token = auth[7:] if auth.lower().startswith("bearer ") else \
            request.headers.get("x-api-key")
    except Exception:
        return ("unknown", set()) if user_store else (None, None)
    if not token:
        return ("unknown", set()) if user_store else (None, None)
    entry = get_tokens().get(token)
    if entry is None:
        return ("unknown", set()) if user_store else (None, None)
    # Mono-map convention (same as databasemcp._current_caller_grants and
    # the flush gate): the legacy single-entry verifier carries NO role —
    # defaulting it to "user" denied EVERY collection to the mono operator
    # (found at a mono-mode deployment 2026-10-26: every search answered
    # "Collection ... is not granted to your account"). Under a user-store
    # verifier a role-less entry stays fail-CLOSED (scoped user) — mono
    # fail-open, multi fail-closed.
    role = entry.get("role", "user" if user_store else "admin")
    if role == "admin":
        return entry.get("client_id"), None
    return entry.get("client_id"), set(entry.get("rag_collections") or [])


def _collection_allowed(name: str, allowed: set[str] | None) -> bool:
    return allowed is None or name in allowed


# Load configuration from root .env
root_env = SCRIPT_DIR.parent.parent / ".env"
if root_env.exists():
    load_dotenv(root_env, encoding="utf-8-sig")
    logger.info(f"Loaded configuration from {root_env}")

# Read embedding configuration
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'azure').lower()
LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'bge-m3')

# ============================================================================
# Port Configuration (from ports.json only)
# ============================================================================

TOOL_NAME = "ragmcp"

try:
    from launcher.launcher_config import load_ports_config
    ports_config = load_ports_config()
    MCP_PORT = int(os.environ.get(
        "MCP_PORT",
        ports_config["assignments"]["mcp"][TOOL_NAME]
    ))
    MGMT_PORT = int(os.environ.get(
        "MCP_MGMT_PORT",
        ports_config["assignments"]["mgmt"][TOOL_NAME]
    ))
except Exception as e:
    print(f"ERROR: Failed to load ports.json: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Local Embedding Model Definitions
# ============================================================================

LOCAL_EMBEDDING_MODELS = {
    'bge-m3': {
        'model_name': 'BAAI/bge-m3',
        'dimensions': 1024,
        'description': 'BGE-M3 - 1024d multilingual with dense+sparse hybrid',
        'device': 'cpu'
    },
    'base': {
        'model_name': 'BAAI/bge-base-en-v1.5',
        'dimensions': 768,
        'description': 'BGE Base - Fast English embeddings for quick testing',
        'device': 'cpu'
    }
}

# Embedding model presets for simplified user experience
EMBEDDING_MODEL_PRESETS = {
    "auto": {"provider": "local", "model": "bge-m3", "dimensions": 1024},
    "fast": {"provider": "local", "model": "base", "dimensions": 768},
    "high-quality": {"provider": "azure", "model": "text-embedding-3-large", "dimensions": 3072},
    # OPTIONAL availability-dependent provider (2026-10-18): free NVIDIA
    # embeddings via OpenRouter. Nobody guarantees the model's availability —
    # the embedding_gate latch treats outages as expected, and the P2 lane
    # machinery gives this fingerprint its OWN collection, so trying it never
    # touches existing indexes.
    "nvidia-free": {
        "provider": "openrouter",
        "model": os.getenv("OPENROUTER_EMBEDDING_MODEL",
                           "nvidia/nemotron-nemotron-nano-9b-v2"),
        "dimensions": 2048,
    },
}

# Global model cache
_local_embedding_model = None
_local_embedding_model_name = None


def resolve_local_model_alias(model_name: str) -> str | None:
    """Map a stored model name to a LOCAL_EMBEDDING_MODELS key.

    Collections record the model that indexed them, which may be either a short
    alias ('base') or the full HuggingFace id the indexer resolved it to
    ('BAAI/bge-base-en-v1.5'). Accept both, so a query embeds with the model
    that actually built the index instead of falling back to bge-m3.

    Returns None when the name is unknown.
    """
    if not model_name:
        return None
    if model_name in LOCAL_EMBEDDING_MODELS:
        return model_name
    for alias, cfg in LOCAL_EMBEDDING_MODELS.items():
        if cfg.get('model_name') == model_name:
            return alias
    return None


def get_local_embedding_model(model_name: str = None):
    """
    Get or create a local embedding model instance.

    Args:
        model_name: Model identifier (bge-m3, base)
                  Defaults to LOCAL_EMBEDDING_MODEL env var

    Returns:
        SentenceTransformer instance or None if not available
    """
    global _local_embedding_model, _local_embedding_model_name

    if not LOCAL_EMBEDDINGS_AVAILABLE:
        return None

    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL

    alias = resolve_local_model_alias(model_name)
    model_config = LOCAL_EMBEDDING_MODELS.get(alias) if alias else None
    if not model_config:
        logger.warning(f"Unknown local embedding model: {model_name}, falling back to bge-m3")
        alias = 'bge-m3'
        model_config = LOCAL_EMBEDDING_MODELS[alias]

    # Key the cache on the alias so 'base' and 'BAAI/bge-base-en-v1.5' share
    # one loaded model instead of thrashing it.
    if _local_embedding_model is not None and _local_embedding_model_name == alias:
        logger.debug(f"Using cached local embedding model: {alias} ({model_name})")
        return _local_embedding_model

    try:
        logger.info(f"Loading local embedding model: {model_config['model_name']}")
        _local_embedding_model = SentenceTransformer(model_config['model_name']).to(model_config['device'])
        _local_embedding_model_name = alias
        logger.info(f"Local embedding model loaded: {model_config['description']} ({model_config['dimensions']}d)")
        return _local_embedding_model
    except Exception as e:
        logger.error(f"Failed to load local embedding model {model_config['model_name']}: {e}")
        return None


def generate_local_embeddings(texts: list, model_name: str = None):
    """
    Generate embeddings for a list of texts using local models.

    Args:
        texts: List of text strings to embed
        model_name: Model identifier (bge-m3, base)

    Returns:
        numpy array of embeddings or None if failed
    """
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        logger.error("Local embeddings not available - sentence-transformers not installed")
        return None

    model = get_local_embedding_model(model_name)
    if model is None:
        logger.error("Failed to get local embedding model")
        return None

    try:
        logger.debug(f"Generating local embeddings for {len(texts)} texts using model: {model_name or LOCAL_EMBEDDING_MODEL}")
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating local embeddings: {e}")
        return None

# Log startup information
logger.info("="*80)
logger.info("RAG MCP Server Starting (FastMCP)")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Embedding provider: {EMBEDDING_PROVIDER}")
logger.info(f"Local embedding model: {LOCAL_EMBEDDING_MODEL}")
logger.info("="*80)

# ============================================================================
# FEF V3 Integration
# ============================================================================

_repo_root = str((Path(__file__).parent / ".." / "..").resolve())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    from launcher.tool_extensions import Extension, ExtensionType, ExtensionRegistry
    FEF_V3_AVAILABLE = True
    logger.info("FEF V3 modules loaded successfully")
except ImportError as e:
    FEF_V3_AVAILABLE = False
    logger.warning(f"FEF V3 not available: {e}")

# RAGMCP-specific metrics
ragmcp_metrics = {
    "semantic_searches": 0,
    "sparse_searches": 0,
    "index_operations": 0,
    "embedding_calls": 0,
    "total_search_time_ms": 0.0,
    "min_search_time_ms": None,
    "max_search_time_ms": 0.0,
    "search_errors": 0,
}


def with_metrics(tool_name: str):
    """Decorator to add metrics recording to a tool function."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                _record_metrics(tool_name, elapsed, True)
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                _record_metrics(tool_name, elapsed, False)
                raise
        return wrapper
    return decorator


def _record_metrics(tool_name: str, elapsed_ms: float, success: bool = True) -> None:
    """Record tool call metrics to local dict and FEF manager."""
    if success:
        if tool_name == "search_code":
            ragmcp_metrics["semantic_searches"] += 1
            ragmcp_metrics["total_search_time_ms"] += elapsed_ms
        elif tool_name == "search_code_sparse":
            ragmcp_metrics["sparse_searches"] += 1
            ragmcp_metrics["total_search_time_ms"] += elapsed_ms
        elif tool_name in ("start_indexing", "clear_index"):
            ragmcp_metrics["index_operations"] += 1
        elif tool_name in ("get_copilot_context", "list_collections", "check_indexing_progress"):
            ragmcp_metrics["total_search_time_ms"] += elapsed_ms
        if ragmcp_metrics["min_search_time_ms"] is None or elapsed_ms < ragmcp_metrics["min_search_time_ms"]:
            ragmcp_metrics["min_search_time_ms"] = elapsed_ms
        if elapsed_ms > ragmcp_metrics["max_search_time_ms"]:
            ragmcp_metrics["max_search_time_ms"] = elapsed_ms
    else:
        ragmcp_metrics["search_errors"] += 1
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name=tool_name,
            success=success, duration_ms=elapsed_ms
        )

# Collection configuration (reads from env vars for hot-reload)

# Audit 2026-10-01 P2-4/P2-5: the guards live in the dependency-free
# path_guards module (importable by tests without backend init).
from indexer.path_guards import allowed_roots_for, validate_collection_name  # noqa: E402


def _rag_allowed_roots() -> list[str]:
    return allowed_roots_for(SCRIPT_DIR / "config.json")


def _workspace_allowed(workspace_root: str) -> str | None:
    from indexer.path_guards import workspace_allowed
    return workspace_allowed(workspace_root, _rag_allowed_roots())


def get_collection_config() -> dict:
    """Get collection config from env vars (hot-reload)."""
    return {
        "default_collection": os.environ.get("RAGMCP_DEFAULT_COLLECTION", "code_index"),
        "similarity_threshold": float(os.environ.get("RAGMCP_SIMILARITY_THRESHOLD", "0.7")),
        "max_results": int(os.environ.get("RAGMCP_MAX_RESULTS", "10")),
    }


def get_vector_db_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get vector database statistics."""
    collections = []
    if vector_store:
        try:
            collections = vector_store.list_collections()
        except Exception:
            pass

    return {
        "connected": vector_store is not None,
        "collections": collections,
        "default_collection": get_collection_config()["default_collection"]
    }


def get_embedding_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get embedding statistics."""
    return {
        "provider": EMBEDDING_PROVIDER,
        "local_model": LOCAL_EMBEDDING_MODEL,
        "embedding_calls": ragmcp_metrics["embedding_calls"],
        "local_available": LOCAL_EMBEDDINGS_AVAILABLE
    }


def get_collection_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get collection statistics."""
    collection_name = params.get("collection", get_collection_config()["default_collection"])
    if not vector_store:
        return {"error": "Qdrant client not initialized"}

    try:
        collection_info = vector_store.get_collection(collection_name)
        return {
            "collection": collection_name,
            "points_count": collection_info.points_count,
            "vectors_size": collection_info.dim
        }
    except Exception as e:
        # Return user-friendly message instead of raw error
        err_str = str(e)
        if "doesn't exist" in err_str or "Not found" in err_str:
            return {"error": f"Collection '{collection_name}' not found", "exists": False}
        return {"error": err_str}


def list_collections_handler(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: List all indexed collections with stats."""
    if not vector_store:
        return {"error": "Qdrant not connected", "collections": []}

    try:
        collections = vector_store.list_collections()
        result = {}
        for coll_name in collections:
            info = vector_store.get_collection(coll_name)
            named_vectors = info.named_vectors

            # Handle both single vector and hybrid (dict) configurations
            if named_vectors:
                dims_parts = []
                for name, dim in named_vectors.items():
                    dims_parts.append(f"{name}={dim}d")
                # Check has_sparse for hybrid
                if info.has_sparse:
                    dims_parts.append("sparse")
                dims = ", ".join(dims_parts) if dims_parts else "unknown"
            else:
                dims = f"{info.dim}d" if info.dim else "unknown"

            result[coll_name] = f"{info.points_count:,} chunks @ {dims}"

        result["total"] = len(collections)
        return result
    except Exception as e:
        return {"error": str(e), "collections": []}


def check_indexing_progress_handler(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get current indexing progress."""
    pid = params.get('pid')

    # Load from PID file
    pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
    if not pid_file.exists():
        return {"status": "no_active_indexing", "message": "No indexing in progress"}

    try:
        with Path(pid_file).open('r') as f:
            pid_info = json.load(f)
            pid = pid_info.get("pid")
            log_file = Path(pid_info.get("log_file", str(SCRIPT_DIR / "logs" / "indexing.log")))
            workspace_root = pid_info.get("workspace_root", "Unknown")
            collection_name = pid_info.get("collection_name", "Unknown")
            started_at = pid_info.get("started_at", 0)

        # Check if process is still running
        try:
            process = psutil.Process(pid)
            is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, AttributeError):
            is_running = False

        status = "running" if is_running else "completed_stopped"

        # Parse log for progress info and completion status
        progress_info = ""
        completed = False
        if log_file.exists():
            with Path(log_file).open('r') as f:
                lines = f.readlines()

            # Find last progress line with ETA
            for line in reversed(lines):
                if "[50/2738]" in line or "[100/" in line or "[/" in line:
                    progress_info = line.strip()
                    break
                if "✅" in line and "INDEXING COMPLETE" in line:
                    progress_info = "Indexing complete"
                    completed = True
                    break

        # Determine if indexing actually completed
        if not is_running and not completed:
            status = "incomplete"

        # Calculate runtime
        runtime = "unknown"
        if started_at > 0:
            elapsed = int(time.time() - started_at)
            minutes = elapsed // 60
            seconds = elapsed % 60
            runtime = f"{minutes}m {seconds}s"

        return {
            "status": status,
            "pid": pid,
            "workspace": workspace_root,
            "collection": collection_name,
            "runtime": runtime,
            "progress": progress_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def start_indexing_handler(params: dict[str, Any]) -> dict[str, Any]:
    """Action: Start indexing a workspace."""
    workspace_root = params.get('workspace_root', '')
    collection_name = params.get('collection_name', 'code-index')
    embedding_model = params.get('embedding_model', 'auto')
    directories = params.get('directories')

    if not workspace_root:
        return {"success": False, "error": "workspace_root is required"}

    _err = validate_collection_name(collection_name)
    if _err:
        return {"success": False, "error": _err}
    _err = _workspace_allowed(workspace_root)
    if _err:
        return {"success": False, "error": _err}

    if not vector_store:
        return {"success": False, "error": "Qdrant not initialized"}

    try:
        preset = EMBEDDING_MODEL_PRESETS.get(embedding_model, EMBEDDING_MODEL_PRESETS["auto"])

        indexer_dir = os.getenv('RAGMCP_INDEXER_DIR', '')
        if indexer_dir:
            indexer_script = Path(indexer_dir) / "incremental_indexer.py"
            indexer_cwd = Path(indexer_dir).parent
        else:
            indexer_script = SCRIPT_DIR / "indexer" / "incremental_indexer.py"
            indexer_cwd = SCRIPT_DIR

        cmd = ["python3", str(indexer_script), workspace_root, "--collection", collection_name]

        # The indexer's --dirs default is the Oracle-era layout (pc/pkg/ddl/...
        # relative to the workspace) — without an explicit list it scans
        # nothing on a modern repo and silently completes with 0 files,
        # leaving a metadata-point-only lane (found live 2026-10-17: lane
        # created, search then failed on unresolvable dimensions). Default to
        # the whole workspace; an explicit directories list is passed through.
        if directories:
            cmd += ["--dirs", *directories]
        else:
            cmd += ["--dirs", str(workspace_root)]

        env = os.environ.copy()
        env['EMBEDDING_PROVIDER'] = preset['provider']
        if preset['provider'] == 'local':
            env['LOCAL_EMBEDDING_MODEL'] = preset['model']
            env['USE_SPARSE_VECTORS'] = 'true'

        logs_dir = SCRIPT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"indexing_{collection_name}.log"

        with Path(log_file).open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=indexer_cwd, env=env, start_new_session=True)

        new_pid = process.pid

        pid_file = SCRIPT_DIR / "logs" / "indexing.pid"

        # Ensure subprocess is reaped properly to prevent zombies
        import threading
        def _reap_on_exit(proc, pid_path):
            """Background thread to reap subprocess when it exits."""
            try:
                proc.wait()
            except Exception:
                pass
            try:
                if pid_path.exists():
                    with pid_path.open('r') as f:
                        saved_pid = json.load(f).get('pid')
                    if saved_pid == new_pid:
                        pid_path.unlink(missing_ok=True)
            except Exception:
                pass

        threading.Thread(target=_reap_on_exit, args=(process, pid_file), daemon=True).start()

        with Path(pid_file).open('w') as f:
            json.dump({
                "pid": new_pid,
                "workspace_root": workspace_root,
                "collection_name": collection_name,
                "log_file": str(log_file),
                "started_at": time.time()
            }, f)

        return {
            "success": True,
            "pid": new_pid,
            "message": f"Indexing started for {workspace_root} -> {collection_name}",
            "log_file": str(log_file)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def reindex(params: dict[str, Any]) -> dict[str, Any]:
    """Action: Trigger reindexing."""
    collection = params.get("collection", get_collection_config()["default_collection"])
    ragmcp_metrics["index_operations"] += 1
    logger.info(f"[ragmcp] Reindex triggered for collection: {collection}")

    return {
        "success": True,
        "message": f"Reindexing initiated for collection: {collection}",
        "collection": collection
    }


def validate_search_request(collection_name: str, query_vector: list, using_vector: str = None) -> dict:
    """
    Validate that query vector dimension matches collection vector config.

    Returns:
        dict with keys:
            - valid: bool
            - error: str (if not valid)
            - warning: str (if valid but degraded)
            - recommended_vector: str (if different vector recommended)
    """
    try:
        info = vector_store.get_collection(collection_name)
        named_vectors = info.named_vectors

        if named_vectors:
            available = dict(named_vectors)
        elif info.dim:
            available = {"default": info.dim}
        else:
            available = {}

        query_dim = len(query_vector)

        # Check if using_vector matches
        if using_vector and using_vector in available:
            if available[using_vector] != query_dim:
                return {
                    "valid": False,
                    "error": f"Query dimension ({query_dim}) vs '{using_vector}' vector ({available[using_vector]}). "
                             f"Use mode='dense' or 'sparse' to select compatible vector."
                }
            return {"valid": True}

        # Find matching vector
        for name, dim in available.items():
            if dim == query_dim:
                if using_vector and using_vector != name:
                    return {
                        "valid": True,
                        "warning": f"Using '{name}' vector (matches {query_dim}d). "
                                   f"Original '{using_vector}' had mismatched dimension.",
                        "recommended_vector": name
                    }
                return {"valid": True}

        # Dimension mismatch - provide helpful error
        return {
            "valid": False,
            "error": f"Query dimension ({query_dim}) does not match collection vectors ({available}). "
                     f"Collection was indexed with different embedding model than search query. "
                     f"Options: 1) Reindex with matching model, 2) Use different collection."
        }
    except Exception as e:
        return {"valid": True, "warning": f"Validation skipped: {e}"}


def _get_collection_dimensions(collection_info) -> int | None:
    """Get the dense vector dimensions from a CollectionInfo object."""
    return dense_dim_of(collection_info)


def setup_fef_v3():
    """Set up FEF V3 extensions for ragmcp."""
    if not FEF_V3_AVAILABLE:
        logger.warning("FEF V3 not available, skipping extension setup")
        return None, None, None

    custom_extensions = [
        Extension(
            name="vector_db_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "connected": {"type": "boolean"},
                        "collections": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            handler=get_vector_db_stats,
            metadata={"description": "Vector database connection and collection info", "category": "metrics"}
        ),
        Extension(
            name="embedding_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "embedding_calls": {"type": "integer"}
                    }
                }
            },
            handler=get_embedding_stats,
            metadata={"description": "Embedding provider statistics", "category": "metrics"}
        ),
        Extension(
            name="collection_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "points_count": {"type": "integer"}
                    }
                }
            },
            handler=get_collection_stats,
            metadata={"description": "Collection statistics", "category": "metrics"}
        ),
        Extension(
            name="reindex",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"}
                    }
                },
                "output": {"type": "object", "properties": {"success": {"type": "boolean"}}}
            },
            handler=reindex,
            metadata={"description": "Trigger reindexing of a collection", "category": "maintenance"}
        ),
        Extension(
            name="list_collections",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object"}
            },
            handler=list_collections_handler,
            metadata={"description": "List all indexed collections with stats", "category": "collections"}
        ),
        Extension(
            name="check_indexing_progress",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {"pid": {"type": "integer", "description": "Optional PID to check"}}},
                "output": {"type": "object"}
            },
            handler=check_indexing_progress_handler,
            metadata={"description": "Show current indexing progress with ETA", "category": "collections"}
        ),
        Extension(
            name="start_indexing",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {
                        "workspace_root": {"type": "string", "description": "Path to workspace to index"},
                        "collection_name": {"type": "string", "description": "Target collection name"},
                        "embedding_model": {"type": "string", "enum": ["auto", "fast", "high-quality", "nvidia-free"], "description": "Embedding model preset"}
                    },
                    "required": ["workspace_root"]
                },
                "output": {"type": "object"}
            },
            handler=start_indexing_handler,
            metadata={"description": "Start indexing a workspace into a collection", "category": "collections"}
        ),
    ]

    return setup_tool_extensions(
        tool_name="ragmcp",
        mgmt_port=MGMT_PORT,
        custom_extensions=custom_extensions
    )


# ============================================================================
# Qdrant Client Initialization
# ============================================================================

logger.info("Initializing vector store for semantic code search...")
vector_store = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.vector_store import get_vector_store
    vector_store = get_vector_store()
    if vector_store is not None:
        logger.info(f"VectorStore initialized: {type(vector_store).__name__}")
    else:
        logger.info("No vector backend configured (Qdrant not available)")
except Exception as e:
    logger.warning(f"Could not initialize VectorStore: {e}")

# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server, DEFAULT_HOST
from tools.shared.migrate_mcp import register_migrate_tools

mcp = create_fastmcp_server(TOOL_NAME)

# Register backend migration tools (migrateMemoryBackend, verifyBackendParity)
register_migrate_tools(mcp)


# ============================================================================
# Tool Implementations
# ============================================================================

@with_metrics("search_code")
@mcp.tool()
async def search_code(
    query: str,
    limit: int = 5,
    file_type: str | None = None,
    function_name: str | None = None,
    collection_name: str = "folder.to.index-database-code"
) -> str:
    """
    Semantic search across indexed code (Pro*C, PL/SQL, Java, etc.) using natural language.
    Returns relevant code chunks with function names and file locations.

    [DEPRECATED] Use 'search' tool with mode='dense' instead.
    """
    logger.warning("search_code is deprecated. Use search(mode='dense') instead.")
    return await search(query=query, limit=limit, collection_name=collection_name,
                       mode="dense", file_type=file_type, function_name=function_name)


@with_metrics("search_code_sparse")
@mcp.tool()
async def search_code_sparse(
    query: str,
    limit: int = 5,
    file_type: str | None = None,
    function_name: str | None = None,
    collection_name: str = "folder.to.index-database-code"
) -> str:
    """
    Lexical (BM25-style) code search using sparse vectors.
    Excellent for finding exact identifiers, table names, function names. Works offline without API costs.
    Use this for precise code lookups (e.g., 'STOMVT table', 'get_movement_type function').

    [DEPRECATED] Use 'search' tool with mode='sparse' instead.
    """
    logger.warning("search_code_sparse is deprecated. Use search(mode='sparse') instead.")
    return await search(query=query, limit=limit, collection_name=collection_name,
                        mode="sparse", file_type=file_type, function_name=function_name)


def _no_sparse_index_error(collection_name: str) -> str:
    """Loud failure for sparse/hybrid queries on collections with no sparse index.

    Dense-only collections (e.g. migrated before the Turso FTS5 sidecar
    existed) used to answer sparse queries with silent zero hits.
    """
    return (f"Error: Collection '{collection_name}' has no sparse index on this "
            f"backend. Rebuild it with: python -m tools.shared.migrate_store "
            f"repair-sparse --collection {collection_name}")


@with_metrics("search")
@mcp.tool()
async def search(
    query: str,
    limit: int = 5,
    collection_name: str = "folder.to.index-database-code",
    mode: str = "auto",  # auto | dense | sparse | hybrid
    file_type: str | None = None,
    function_name: str | None = None,
    copilot_format: str | None = None,  # "comment" | "sidebar"
    language: str = "c",
    max_lines: int = 50,
    embedding_model: str | None = None,  # P2 lanes: preset for the query embedder + active lane (auto | fast | high-quality)
    search_all_lanes: bool = False,      # P2 lanes: opt-in multi-lane search (one query embedding per lane model)
) -> str:
    """
    Unified code search with automatic detection of collection capabilities.

    - mode='auto': Detect collection and use best available search method
    - mode='dense': Semantic search using embeddings (requires matching dimension)
    - mode='sparse': BM25 lexical search for exact identifiers
    - mode='hybrid': Combined dense + sparse (requires hybrid collection)

    P2 embedding lanes: when `<collection_name>-<provider-model-dim>` lanes
    exist, search targets the ACTIVE lane (matching embedding_model, or the
    server env config) plus any legacy collection built by the same embedder.
    search_all_lanes=true queries every lane (embeds the query once per lane
    model — remote lanes cost one API call each). Sparse mode is
    model-independent and sweeps every lane.

    copilot_format adds GitHub Copilot formatting:
    - 'comment': Inline comment block
    - 'sidebar': Sidebar context block
    """
    logger.debug(f"Processing unified search: query={query}, mode={mode}, copilot_format={copilot_format}")
    _client_id, _allowed = _caller_rag_access()
    if not _collection_allowed(collection_name, _allowed):
        return f"Error: Collection '{collection_name}' is not granted to your account."

    if not vector_store:
        return "Error: Qdrant client not initialized."

    if not query:
        raise ValueError("Missing argument: query")

    # P2 lanes: when lane collections exist for this base, search across
    # them; with no lanes the single-collection flow below is unchanged.
    lane_collections = [
        c for c in list_lanes(vector_store, collection_name) if c != collection_name
    ]
    if lane_collections:
        return await _search_lanes(
            query, limit, collection_name, lane_collections, mode, file_type,
            function_name, copilot_format, language, max_lines,
            embedding_model, search_all_lanes,
        )

    # Detect collection capabilities
    try:
        info = vector_store.get_collection(collection_name)
        has_dense = info.has_dense
        has_sparse = info.has_sparse
    except Exception as e:
        return f"Error: Collection '{collection_name}' not found: {e}"

    # Auto-select mode based on collection
    if mode == "auto":
        if has_sparse and has_dense:
            mode = "hybrid"
        elif has_sparse:
            mode = "sparse"
        elif has_dense:
            mode = "dense"
        else:
            return f"Error: Collection '{collection_name}' has no vectors configured."

    if mode in ("sparse", "hybrid") and not has_sparse:
        return _no_sparse_index_error(collection_name)

    # Handle copilot format - route to copilot handler if requested
    if copilot_format:
        return await _search_copilot(query, limit, collection_name, file_type, function_name,
                                     copilot_format, language, max_lines)

    # Route to appropriate search
    if mode == "dense":
        return await _search_dense(query, limit, collection_name, file_type, function_name)
    elif mode == "sparse":
        return await _search_sparse(query, limit, collection_name, file_type, function_name)
    elif mode == "hybrid":
        return await _search_hybrid(query, limit, collection_name, file_type, function_name)
    else:
        return f"Error: Unknown mode '{mode}'. Use: auto, dense, sparse, hybrid"


async def _embed_query(query: str, preset: dict | None = None) -> list[float]:
    """Embed one query string.

    preset ({'provider','model',...}) selects the embedder explicitly (P2
    lanes: each lane is searched with its own model); None uses the server
    env config (pre-lane behavior). Local models embed in-process.
    """
    provider = (preset or {}).get('provider') or os.getenv('EMBEDDING_PROVIDER', 'azure')

    if provider == 'azure':
        import httpx
        azure_api_url = os.getenv('AZURE_EMBEDDING_API_URL',
                                  'https://put.your.API.gateway.ai/v1/embeddings')
        azure_model = (preset or {}).get('model') or os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
        api_key = os.getenv('AI_API_KEY', '')

        if not api_key:
            raise ValueError("AI_API_KEY not set. Cannot generate search embedding.")

        embedding_gate.check(azure_api_url)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    azure_api_url,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    json={"input": [query], "model": azure_model}
                )
                response.raise_for_status()
                data = response.json()
                query_vector = data['data'][0]['embedding']
            embedding_gate.record_success()
            return query_vector
        except Exception:
            embedding_gate.record_failure(azure_api_url)
            raise

    if provider == 'openrouter':
        # OPTIONAL provider (free NVIDIA embeddings via OpenRouter) — same
        # OpenAI-compatible body as the azure path, availability-gated by
        # embedding_gate; errors name the local presets as the offline route.
        import httpx
        url = os.getenv('OPENROUTER_BASE', 'https://openrouter.ai/api/v1/embeddings')
        model = (preset or {}).get('model') or os.getenv(
            'OPENROUTER_EMBEDDING_MODEL', 'nvidia/nemotron-nemotron-nano-9b-v2')
        api_key = os.getenv('OPENROUTER_KEY', '')

        if not api_key:
            raise ValueError(
                "OPENROUTER_KEY not set — the 'nvidia-free' preset needs it. "
                "Availability is not guaranteed; use the 'auto' (local bge-m3) "
                "or 'fast' (local bge-base) presets for offline work.")

        embedding_gate.check(url)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {api_key}"},
                    json={"input": [query], "model": model}
                )
                response.raise_for_status()
                query_vector = response.json()['data'][0]['embedding']
            embedding_gate.record_success()
            return query_vector
        except Exception:
            embedding_gate.record_failure(url)
            raise
    else:
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            raise ValueError("Local embeddings module not available. Install sentence-transformers.")

        # Query-model resolution order (lanes ∘ recorded-metadata): an active
        # lane pins the model (P2); otherwise prefer the model recorded in the
        # collection's __collection_metadata__ point (written by the indexer)
        # so queries match the index even when the server's global embedding
        # model differs (e.g. bge-m3 server, bge-base index). Azure model
        # names are ignored here — this branch only embeds locally.
        query_model = (preset or {}).get('model') or LOCAL_EMBEDDING_MODEL
        if not (preset or {}).get('model'):
            try:
                meta_points = vector_store.retrieve(collection_name, ids=["__collection_metadata__"])
                if meta_points:
                    stored_model = (meta_points[0].payload or {}).get('embedding_model')
                    if stored_model and not stored_model.startswith('text-embedding'):
                        query_model = stored_model
            except Exception as meta_err:
                logger.debug(f"Could not read collection metadata for model selection: {meta_err}")

        query_embeddings = generate_local_embeddings([query], model_name=query_model)
        if query_embeddings is None or len(query_embeddings) == 0:
            raise ValueError("Failed to generate local embedding for query.")
        return query_embeddings[0].tolist()


async def _dense_search_with_vector(query_vector: list, limit: int, collection_name: str,
                                    file_type: str | None, function_name: str | None) -> list:
    """Dense search with a pre-computed query vector."""
    from shared.store_models import Filter, FieldCondition, MatchValue

    conditions = []
    if file_type:
        conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))
    if function_name:
        conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))
    search_filter = Filter(must=conditions) if conditions else None

    collection_info = vector_store.get_collection(collection_name)
    named_vectors = collection_info.named_vectors

    if named_vectors:
        vector_dim = len(query_vector)
        vector_name = None

        for vname, dim in named_vectors.items():
            if dim == vector_dim:
                vector_name = vname
                break

        if not vector_name:
            vector_name = next(iter(named_vectors.keys()))

        results = vector_store.query_dense(
            collection_name, query_vector,
            limit=limit, filter=search_filter, using=vector_name,
        )
    else:
        results = vector_store.query_dense(
            collection_name, query_vector,
            limit=limit, filter=search_filter,
        )

    return [{"id": str(p.id), "score": float(p.score), "payload": p.payload}
            for p in results]


async def _do_dense_search(query: str, limit: int, collection_name: str,
                           file_type: str | None, function_name: str | None) -> list:
    """Execute dense search and return raw scoreless points list.

    Returns list of dicts: {id, score, payload}.
    Raises on error (caller handles).
    """
    query_vector = await _embed_query(query)

    validation = validate_search_request(collection_name, query_vector)
    if not validation.get('valid', True) and 'error' in validation:
        raise ValueError(validation['error'])
    if validation.get('warning'):
        logger.warning(validation['warning'])

    return await _dense_search_with_vector(
        query_vector, limit, collection_name, file_type, function_name)


async def _sparse_search_with_vector(query_text: str, sparse_vec: dict,
                                     limit: int, collection_name: str,
                                     file_type: str | None, function_name: str | None) -> list:
    """Sparse (BM25) search with a pre-built term-hash vector (model-independent)."""
    from shared.store_models import Filter, FieldCondition, MatchValue, SparseVector as NeutralSparseVector

    conditions = []
    if file_type:
        conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))
    if function_name:
        conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))
    search_filter = Filter(must=conditions) if conditions else None

    neutral_vec = NeutralSparseVector(
        indices=list(sparse_vec.keys()),
        values=list(sparse_vec.values()),
    )
    # Pass the raw query text so lexical backends (Turso FTS5 / PG tsvector) can
    # run a real MATCH. The integer sparse_vec is Qdrant-native only.
    results = vector_store.query_sparse(
        collection_name, neutral_vec,
        limit=limit, filter=search_filter,
        query_text=query_text,
    )

    return [{"id": str(p.id), "score": float(p.score), "payload": p.payload}
            for p in results]


async def _do_sparse_search(query: str, limit: int, collection_name: str,
                            file_type: str | None, function_name: str | None) -> list:
    """Execute sparse search and return structured results list.

    Returns list of dicts: {id, score, payload}.
    Raises on error (caller handles).
    """
    from shared.store_models import SparseVector as NeutralSparseVector

    if not SPARSE_VECTORS_AVAILABLE:
        raise ValueError("Sparse vector search not available. Missing sparse_vector_gen.py module.")

    query_metadata = {'language': file_type if file_type else 'unknown'}
    query_sparse_vec = generate_sparse_vector(query, query_metadata)

    if not query_sparse_vec:
        raise ValueError("Failed to generate sparse vector for query.")

    return await _sparse_search_with_vector(
        query, query_sparse_vec, limit, collection_name, file_type, function_name)


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list contains dicts with 'id', 'score', 'payload'.
        k: RRF constant (default 60).

    Returns:
        Single deduplicated list sorted by fused RRF score (desc).
    """
    rrf_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for result_list in ranked_lists:
        for rank, item in enumerate(result_list, start=1):
            doc_id = item["id"]
            if doc_id not in payloads:
                payloads[doc_id] = item["payload"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = [
        {"id": doc_id, "score": score, "payload": payloads[doc_id]}
        for doc_id, score in rrf_scores.items()
    ]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def _format_structured_results(results: list[dict]) -> str:
    """Format structured search results (list of dicts) into display string."""
    if not results:
        return "No results found."

    formatted = []
    formatted.append(f"Found {len(results)} relevant code chunks:\n")
    formatted.append("=" * 80 + "\n")

    for i, item in enumerate(results, 1):
        payload = item["payload"]
        score = item["score"]

        formatted.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
        formatted.append(f"File: {payload.get('filePath', 'Unknown')}\n")
        formatted.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
        formatted.append(f"Type: {payload.get('fileType', 'Unknown')}\n")
        if payload.get('_lane'):
            formatted.append(f"Lane: {payload['_lane']}\n")

        if payload.get('functionName'):
            formatted.append(f"Function: {payload['functionName']}\n")
        if payload.get('chunkType'):
            formatted.append(f"Chunk Type: {payload['chunkType']}\n")

        formatted.append("\nCode:\n```\n")
        code_chunk = payload.get('codeChunk', '')
        lines = code_chunk.split('\n')
        if len(lines) > 50:
            formatted.append('\n'.join(lines[:50]))
            formatted.append(f"\n... ({len(lines) - 50} more lines)")
        else:
            formatted.append(code_chunk)
        formatted.append("\n```\n")
        formatted.append("-" * 80 + "\n")

    return ''.join(formatted)


def _resolve_active_preset(embedding_model: str | None) -> dict:
    """Preset for the 'active embedder': explicit param wins, else env config."""
    if embedding_model:
        preset = EMBEDDING_MODEL_PRESETS.get(embedding_model)
        if preset is None:
            raise ValueError(
                f"Unknown embedding_model '{embedding_model}'. Use: {', '.join(EMBEDDING_MODEL_PRESETS)}"
            )
        return {**preset, "label": embedding_model}
    provider = os.getenv('EMBEDDING_PROVIDER', 'azure')
    if provider == 'local':
        model = LOCAL_EMBEDDING_MODEL
        dims = LOCAL_EMBEDDING_MODELS.get(model, LOCAL_EMBEDDING_MODELS['bge-m3'])['dimensions']
    else:
        model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
        dims = int(os.getenv('AZURE_EMBEDDING_DIMENSIONS', '3072'))
    return {"provider": provider, "model": model, "dimensions": dims, "label": f"{provider}:{model}"}


async def _search_lanes(query: str, limit: int, base: str, lane_collections: list[str],
                        mode: str, file_type: str | None, function_name: str | None,
                        copilot_format: str | None, language: str, max_lines: int,
                        embedding_model: str | None, search_all_lanes: bool) -> str:
    """Multi-lane search (P2): per-candidate mode detection, one query
    embedding per distinct fingerprint, RRF fusion across candidates."""
    try:
        active = _resolve_active_preset(embedding_model)
    except ValueError as e:
        return f"Error: {e}"

    resolved = resolve_search_candidates(
        vector_store, base,
        mode=mode, active=active, search_all_lanes=search_all_lanes,
    )
    candidates, notes = resolved["candidates"], resolved["notes"]

    if not candidates:
        msg = f"No searchable collections for `{base}`."
        if notes:
            msg += "\n\n" + "\n".join(f"- {n}" for n in notes)
        return msg

    # copilot formatting keeps single-collection semantics: first candidate.
    if copilot_format:
        coll = candidates[0][0]
        return await _search_copilot(query, limit, coll, file_type, function_name,
                                     copilot_format, language, max_lines)

    dense_lists: list[list[dict]] = []
    sparse_lists: list[list[dict]] = []
    embed_cache: dict[str, list[float]] = {}
    sparse_vec: dict | None = None

    for coll, label, preset in candidates:
        try:
            info = vector_store.get_collection(coll)
        except Exception as e:
            notes.append(f"'{coll}' unreadable: {e}")
            continue

        cmode = mode
        if cmode == "auto":
            cmode = "hybrid" if (info.has_sparse and info.has_dense) else ("sparse" if info.has_sparse else "dense")
        if cmode == "sparse" and not info.has_sparse:
            notes.append(f"'{coll}' ({label}): no sparse index — skipped")
            continue
        if cmode in ("dense", "hybrid") and not info.has_dense:
            if info.has_sparse:
                cmode = "sparse"
            else:
                notes.append(f"'{coll}' ({label}): no vectors — skipped")
                continue

        try:
            if cmode in ("dense", "hybrid"):
                fp = lane_fingerprint(preset['provider'], preset['model'], preset['dimensions'])
                if fp not in embed_cache:
                    embed_cache[fp] = await _embed_query(query, preset)
                qv = embed_cache[fp]
                validation = validate_search_request(coll, qv)
                if validation.get('valid') is False:
                    notes.append(f"'{coll}' ({label}) dense skipped: {validation.get('error')}")
                else:
                    if validation.get('warning'):
                        notes.append(f"'{coll}' ({label}): {validation['warning']}")
                    hits = await _dense_search_with_vector(qv, limit, coll, file_type, function_name)
                    for h in hits:
                        h["payload"]["_lane"] = label
                    dense_lists.append(hits)
            if cmode in ("sparse", "hybrid"):
                if sparse_vec is None:
                    if not SPARSE_VECTORS_AVAILABLE:
                        notes.append(f"'{coll}' ({label}): sparse generator unavailable — skipped")
                    else:
                        sparse_vec = generate_sparse_vector(query, {'language': file_type or 'unknown'})
                if sparse_vec:
                    hits = await _sparse_search_with_vector(
                        query, sparse_vec, limit, coll, file_type, function_name)
                    for h in hits:
                        h["payload"]["_lane"] = label
                    sparse_lists.append(hits)
        except embedding_gate.EmbeddingEndpointDown as e:
            notes.append(f"'{coll}' ({label}): {e}")
        except Exception as e:
            notes.append(f"'{coll}' ({label}): {e}")

    fused = _reciprocal_rank_fusion(dense_lists + sparse_lists)[:limit]

    if not fused:
        msg = (f"No results found for `{base}` across {len(candidates)} lane candidate(s).")
        if notes:
            msg += "\n\n" + "\n".join(f"- {n}" for n in notes)
        return msg

    parts = [
        f"Lane search results for `{base}` "
        f"(RRF across {len(dense_lists)} dense + {len(sparse_lists)} sparse lists, "
        f"active embedder: {active['label']}):"
    ]
    parts.append(_format_structured_results(fused))
    if notes:
        parts.append("Notes:\n" + "\n".join(f"- {n}" for n in notes))
    return "\n".join(parts)


async def _search_dense(query: str, limit: int, collection_name: str,
                        file_type: str | None, function_name: str | None) -> str:
    """Internal: dense semantic search."""
    try:
        results = await _do_dense_search(query, limit, collection_name, file_type, function_name)
    except Exception as e:
        return f"Error in dense search: {str(e)}"

    return _format_structured_results(results)


async def _search_sparse(query: str, limit: int, collection_name: str,
                         file_type: str | None, function_name: str | None) -> str:
    """Internal: sparse BM25 lexical search."""
    try:
        results = await _do_sparse_search(query, limit, collection_name, file_type, function_name)
    except Exception as e:
        return f"Error in sparse search: {str(e)}"

    return _format_structured_results(results)


async def _search_hybrid(query: str, limit: int, collection_name: str,
                         file_type: str | None, function_name: str | None) -> str:
    """Internal: combined dense + sparse search using reciprocal rank fusion."""

    K = 60
    try:
        dense_task = asyncio.create_task(
            _do_dense_search(query, limit, collection_name, file_type, function_name))
        sparse_task = asyncio.create_task(
            _do_sparse_search(query, limit, collection_name, file_type, function_name))
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task)

        fused = _reciprocal_rank_fusion([dense_results, sparse_results], k=K)
        fused = fused[:limit]

        formatted = _format_structured_results(fused)

        return (
            f"Hybrid search results (reciprocal rank fusion, k={K}):\n"
            f"Dense: {len(dense_results)} results | Sparse: {len(sparse_results)} results | "
            f"Merged (deduplicated): {len(fused)} results\n\n"
            f"=== Fused Results ===\n{formatted}"
        )
    except Exception as e:
        return f"Error in hybrid search: {str(e)}"


async def _search_copilot(query: str, limit: int, collection_name: str,
                          file_type: str | None, function_name: str | None,
                          copilot_format: str, language: str, max_lines: int) -> str:
    """Internal: search with copilot formatting."""
    from shared.store_models import SparseVector as NeutralSparseVector

    if not COPILOT_INJECTOR_AVAILABLE:
        return "Error: Copilot context injector not available. Missing copilot_context_injector.py module."

    if not SPARSE_VECTORS_AVAILABLE:
        return "Error: Sparse vector search required for copilot context. Missing sparse_vector_gen.py module."

    try:
        if not vector_store.get_collection(collection_name).has_sparse:
            return _no_sparse_index_error(collection_name)
    except Exception as e:
        return f"Error: Collection '{collection_name}' not found: {e}"

    try:
        injector = get_injector(max_context_lines=max_lines)

        # Extract keywords
        keywords = injector.extract_keywords_from_context(query)
        if not keywords:
            return injector._format_no_context(language)

        search_query = ' '.join(keywords[:5])

        # Sparse search
        query_metadata = {'language': 'unknown'}
        query_sparse_vec = generate_query_vector(search_query, query_metadata)

        if not query_sparse_vec:
            return injector._format_no_context(language)

        sparse_vec = NeutralSparseVector(
            indices=list(query_sparse_vec.keys()),
            values=list(query_sparse_vec.values()),
        )
        # Pass the derived keyword query so lexical backends run a real MATCH.
        results = vector_store.query_sparse(
            collection_name, sparse_vec,
            limit=limit,
            query_text=search_query,
        )

        chunks = [hit.payload for hit in results]

        if not chunks:
            return injector._format_no_context(language)

        if copilot_format == "sidebar":
            return injector.format_sidebar_context(chunks)
        else:
            return injector.format_context_comment(chunks, language=language)

    except Exception as e:
        return f"Error in copilot search: {str(e)}"


def _format_search_results(search_results) -> str:
    """Format search results consistently."""
    if not search_results:
        return "No results found."

    formatted = []
    formatted.append(f"Found {len(search_results)} relevant code chunks:\n")
    formatted.append("=" * 80 + "\n")

    for i, hit in enumerate(search_results, 1):
        payload = hit.payload
        score = hit.score

        formatted.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
        formatted.append(f"File: {payload.get('filePath', 'Unknown')}\n")
        formatted.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
        formatted.append(f"Type: {payload.get('fileType', 'Unknown')}\n")
        if payload.get('_lane'):
            formatted.append(f"Lane: {payload['_lane']}\n")

        if payload.get('functionName'):
            formatted.append(f"Function: {payload['functionName']}\n")
        if payload.get('chunkType'):
            formatted.append(f"Chunk Type: {payload['chunkType']}\n")

        formatted.append("\nCode:\n```\n")
        code_chunk = payload.get('codeChunk', '')
        lines = code_chunk.split('\n')
        if len(lines) > 50:
            formatted.append('\n'.join(lines[:50]))
            formatted.append(f"\n... ({len(lines) - 50} more lines)")
        else:
            formatted.append(code_chunk)
        formatted.append("\n```\n")
        formatted.append("-" * 80 + "\n")

    return ''.join(formatted)


@with_metrics("get_copilot_context")
@mcp.tool()
async def get_copilot_context(
    current_context: str,
    format: str = "comment",
    language: str = "c",
    limit: int = 3,
    max_lines: int = 50,
    collection_name: str = "folder.to.index-database-code"
) -> str:
    """
    Get formatted code context for GitHub Copilot injection.
    Retrieves relevant code using sparse vectors and formats it as inline comments or markdown.
    Perfect for making Copilot project-aware.

    [DEPRECATED] Use 'search' tool with copilot_format='comment' or 'sidebar' instead.
    """
    logger.warning("get_copilot_context is deprecated. Use search(copilot_format='comment' or 'sidebar') instead.")
    copilot_format = "sidebar" if format == "sidebar" else "comment"
    return await search(query=current_context, limit=limit, collection_name=collection_name,
                        copilot_format=copilot_format, language=language, max_lines=max_lines)


@with_metrics("index_code")
@mcp.tool()
async def index_code(
    workspace_root: str,
    collection_name: str = "folder.to.index-database-code",
    directories: list[str] | None = None,
    embedding_model: str = "auto",  # auto | fast | high-quality
    force: bool = False,
    log_level: str = "info"  # debug | info | warning | error
) -> str:
    """
    Index code files into Qdrant for semantic search.

    Simple interface: just specify workspace and collection name.

    embedding_model presets:
    - 'auto': BGE-M3 (1024d) - best local quality, multilingual, hybrid
    - 'fast': BGE Base (768d) - quick testing, local
    - 'high-quality': text-embedding-3-large (3072d) - best, Azure API cost

    For advanced options (mode, custom embedding provider), use start_indexing.
    """
    _client_id, _allowed = _caller_rag_access()
    if not _collection_allowed(collection_name, _allowed):
        return f"Error: Collection '{collection_name}' is not granted to your account."
    _err = validate_collection_name(collection_name)
    if _err:
        return _err
    _err = _workspace_allowed(workspace_root)
    if _err:
        return _err
    logger.debug(f"Processing index_code tool: workspace_root={workspace_root}, embedding_model={embedding_model}")

    if not vector_store:
        return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT environment variables."

    # Resolve preset
    preset = EMBEDDING_MODEL_PRESETS.get(embedding_model, EMBEDDING_MODEL_PRESETS["auto"])
    logger.info(f"Using embedding preset '{embedding_model}': provider={preset['provider']}, model={preset['model']}, dims={preset['dimensions']}")

    # P2 lanes: each embedding fingerprint indexes into its own
    # `<base>-<provider-model-dim>` collection, so switching models never
    # collides with an existing collection's fixed vector size. Sparse-only
    # mode is model-independent → plain base collection, no lane.
    sparse_only = os.getenv('SPARSE_ONLY_MODE', 'false').lower() in ('true', '1', 'yes')
    if sparse_only:
        target_collection = collection_name
    else:
        target_collection = resolve_lane(collection_name, preset['provider'], preset['model'], preset['dimensions'])

    # Auto-create collection if it doesn't exist (with correct dimensions)
    collection_exists = False
    try:
        vector_store.get_collection(target_collection)
        collection_exists = True
        logger.info(f"Collection '{target_collection}' already exists")
    except Exception as e:
        if "Not found" in str(e) or "doesn't exist" in str(e):
            logger.info(f"Collection '{target_collection}' does not exist, will be auto-created by indexer")
            collection_exists = False
        else:
            return f"Error: Error checking collection: {str(e)}"

    # Pre-flight dim check (P2): abort BEFORE spawning the indexer instead of
    # failing mid-run at the first upsert (ensure_collection never re-validates
    # an existing collection's vector size).
    if collection_exists:
        mismatch = check_lane_compatibility(
            vector_store, target_collection,
            preset['provider'], preset['model'], preset['dimensions'],
        )
        if mismatch:
            return mismatch

    # If force=True and collection exists, warn but don't delete automatically
    if force and collection_exists:
        return f"""Warning: Collection Already Exists

Collection `{target_collection}` already exists. The `force` parameter is no longer used to delete existing collections for safety.

Options:
1. Continue indexing (recommended): Remove `force=true` and rerun. The indexer will skip already-indexed files.
2. Clear and reindex: First use `clear_index(collection_name="{target_collection}", confirm=true)` to delete the collection, then start indexing again.

Why this changed: To prevent accidental data loss, the indexer now requires explicit confirmation before deleting existing indexes."""

    # Build command - using incremental indexer
    indexer_dir = os.getenv('RAGMCP_INDEXER_DIR', '')
    if indexer_dir:
        indexer_script = Path(indexer_dir) / "incremental_indexer.py"
        indexer_cwd = Path(indexer_dir).parent
    else:
        indexer_script = SCRIPT_DIR / "indexer" / "incremental_indexer.py"
        indexer_cwd = SCRIPT_DIR
    if not indexer_script.exists():
        return f"Error: Indexer script not found at {indexer_script}"

    cmd = [
        "python3",
        str(indexer_script),
        workspace_root,
        "--collection", target_collection
    ]

    if force:
        cmd.append("--force")

    cmd.extend(["--log-level", log_level])

    if directories:
        cmd.extend(["--dirs"] + directories)
    else:
        auto_dirs = [d.name for d in Path(workspace_root).iterdir() if d.is_dir() and not d.name.startswith('.')]
        if auto_dirs:
            cmd.extend(["--dirs"] + auto_dirs)
        else:
            # Flat workspace (files at the root, no subdirs): scan the root
            # itself — otherwise no --dirs lands and the indexer's Oracle-era
            # default (pc/pkg/ddl/...) silently indexes nothing (found live
            # 2026-10-17: lane created with only its metadata point, then
            # search failed on unresolvable dimensions).
            cmd.extend(["--dirs", str(workspace_root)])

    # Create logs/ directory if it doesn't exist
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path in logs/ subfolder
    log_file = logs_dir / f"indexing_{target_collection}.log"

    # Prepare environment with embedding configuration from preset
    env = os.environ.copy()
    env['EMBEDDING_PROVIDER'] = preset['provider']
    if preset['provider'] == 'local':
        env['LOCAL_EMBEDDING_MODEL'] = preset['model']
        env['USE_SPARSE_VECTORS'] = 'true'

    # Start process in background
    logger.info(f"Starting indexing process: {' '.join(cmd)}")

    with Path(log_file).open('w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=indexer_cwd,
            env=env,
            start_new_session=True
        )

    pid = process.pid

    # Ensure subprocess is reaped properly to prevent zombies
    # Use a non-blocking approach - we just need to track the PID
    # The actual wait happens when stop_indexing kills the process or it naturally exits
    import threading
    def _reap_on_exit(proc, pid_path):
        """Background thread to reap subprocess when it exits."""
        try:
            proc.wait()
        except Exception:
            pass
        # Clean up PID file when process exits normally
        try:
            if pid_path.exists():
                with pid_path.open('r') as f:
                    saved_pid = json.load(f).get('pid')
                if saved_pid == pid:
                    pid_path.unlink(missing_ok=True)
        except Exception:
            pass

    pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
    threading.Thread(target=_reap_on_exit, args=(process, pid_file), daemon=True).start()

    # Save PID to file in logs/ subfolder for later reference
    pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
    with Path(pid_file).open('w') as f:
        json.dump({
            "pid": pid,
            "workspace_root": workspace_root,
            "collection_name": target_collection,
            "base_collection": collection_name,
            "embedding_preset": embedding_model,
            "log_file": str(log_file),
            "started_at": time.time(),
            "embedding_model": embedding_model,
            "embedding_dimensions": preset['dimensions']
        }, f)

    result = f"""Indexing started successfully!

Process Information:
- PID: {pid}
- Workspace: {workspace_root}
- Collection: {target_collection}
- Embedding model: {embedding_model} ({preset['dimensions']}d)
- Force reindex: {force}
- Log file: {log_file}

Next Steps:
1. Monitor progress: Use check_indexing_progress tool
2. View live logs: tail -f {log_file}
3. Check collection: Query Qdrant at qdrant:6333

The indexing process is running in the background. Use check_indexing_progress to monitor status.
"""

    logger.info(f"Indexing started with PID {pid}")
    return result


@with_metrics("start_indexing")
@mcp.tool()
async def start_indexing(
    workspace_root: str = "/path/to/your/workspace",
    collection_name: str = "folder.to.index-database-code",
    mode: str | None = None,
    force: bool = False,
    directories: list[str] | None = None,
    embedding_provider: str | None = None,
    local_embedding_model: str | None = None
) -> str:
    """
    Start background indexing of code files into Qdrant.
    Returns process ID (PID) for monitoring. Indexes Pro*C, PL/SQL, Java, and other files with smart function-level chunking.
    Supports sparse (BM25, $0), dense (embeddings, API cost), or hybrid modes.

    [DEPRECATED] Use 'index_code' tool with embedding_model preset instead.
    """
    logger.warning("start_indexing is deprecated. Use index_code with embedding_model preset instead.")
    return await index_code(workspace_root=workspace_root, collection_name=collection_name,
                            directories=directories, force=force,
                            embedding_model=local_embedding_model)  # simplified mapping


@with_metrics("check_indexing_progress")
@mcp.tool()
async def check_indexing_progress(pid: int | None = None) -> str:
    """
    Check the progress of background indexing process.
    Returns status, files processed, chunks indexed, errors, and recent log entries.
    """
    logger.debug(f"Processing check_indexing_progress tool: pid={pid}")

    try:
        # Load PID info from file if not provided (in logs/ subfolder)
        pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
        if pid_file.exists():
            with Path(pid_file).open('r') as f:
                pid_info = json.load(f)
                if not pid:
                    pid = pid_info.get("pid")
                log_file = Path(pid_info.get("log_file", SCRIPT_DIR / "logs" / "indexing.log"))
                collection_name = pid_info.get("collection_name", "folder.to.index-application-code")
                workspace_root = pid_info.get("workspace_root", "Unknown")
                started_at = pid_info.get("started_at", 0)
        else:
            log_file = SCRIPT_DIR / "logs" / "indexing.log"
            collection_name = "folder.to.index-application-code"
            workspace_root = "Unknown"
            started_at = 0

        # Check if process is running
        is_running = False
        if pid:
            try:
                process = psutil.Process(pid)
                is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                is_running = False

        # Read log file
        if not log_file.exists():
            return "No indexing log found. Start indexing first with start_indexing tool."

        with Path(log_file).open('r') as f:
            log_lines = f.readlines()

        # Parse progress from logs
        files_processed = sum(1 for line in log_lines if "Processing [" in line)
        chunks_indexed = sum(1 for line in log_lines if "Indexed batch" in line)
        errors = sum(1 for line in log_lines if "ERROR" in line)

        # Get total files if available
        total_files = "Unknown"
        for line in reversed(log_lines):
            if "Found " in line and " files to index" in line:
                try:
                    total_files = line.split("Found ")[1].split(" files")[0]
                    break
                except:
                    pass

        # Get last 15 lines of log
        recent_logs = ''.join(log_lines[-15:]) if log_lines else "No logs yet"

        # Get collection stats
        collection_stats = "Unknown"
        if vector_store:
            try:
                collection_info = vector_store.get_collection(collection_name)
                points_count = collection_info.points_count

                # Build vector dimensions string from named_vectors
                if collection_info.named_vectors:
                    vector_parts = []
                    for name, dim in collection_info.named_vectors.items():
                        vector_parts.append(f"{name}={dim}d")
                    if collection_info.has_sparse:
                        vector_parts.append("sparse")
                    vector_dims = ", ".join(vector_parts) if vector_parts else "unknown"
                elif collection_info.dim:
                    vector_dims = f"{collection_info.dim}d"
                else:
                    vector_dims = "unknown"

                collection_stats = f"{points_count:,} chunks indexed ({vector_dims})"
            except Exception as e:
                collection_stats = f"Error: {str(e)}"

        # Calculate runtime
        runtime = "Unknown"
        if started_at > 0:
            elapsed = int(time.time() - started_at)
            minutes = elapsed // 60
            seconds = elapsed % 60
            runtime = f"{minutes}m {seconds}s"

        # Build status report
        status_icon = "RUNNING" if is_running else "COMPLETED/STOPPED"

        result = f"""Indexing Progress Report

Status: {status_icon}
Process ID: {pid if pid else 'Unknown'}
Workspace: {workspace_root}
Collection: {collection_name}
Runtime: {runtime}

Progress:
- Files processed: {files_processed} / {total_files}
- Chunks indexed: {chunks_indexed} batches
- Errors: {errors}
- Collection size: {collection_stats}

Recent Log Entries (last 15 lines):
```
{recent_logs}```

Log File: {log_file}

{'Process is still running. Check back later for updates.' if is_running else 'Process has completed. Review logs for final status.'}
"""

        logger.info(f"Progress check: {files_processed} files, {chunks_indexed} batches, running={is_running}")
        return result

    except Exception as e:
        error_msg = f"Error checking indexing progress: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


@with_metrics("list_collections")
@mcp.tool()
async def list_collections() -> str:
    """List all Qdrant collections with their stats (number of chunks, vector dimensions)."""
    try:
        if not vector_store:
            return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."

        collections = vector_store.list_collections()

        if not collections:
            return """No Collections Found

The Qdrant vector database is empty. No code has been indexed yet.

To start indexing:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="your-database-code")
```
"""

        # Build formatted output
        result = ["Qdrant Collections\n"]
        result.append("=" * 80 + "\n\n")

        client_id, allowed = _caller_rag_access()
        collections = [c for c in collections
                       if _collection_allowed(c, allowed)]
        for collection in collections:
            try:
                collection_info = vector_store.get_collection(collection)
                points_count = collection_info.points_count

                # Build vector dimensions from named_vectors
                if collection_info.named_vectors:
                    vector_parts = []
                    for name, dim in collection_info.named_vectors.items():
                        vector_parts.append(f"{name}={dim}d")
                    if collection_info.has_sparse:
                        vector_parts.append("sparse")
                    vector_size_str = ", ".join(vector_parts) if vector_parts else "unknown"
                elif collection_info.dim:
                    vector_size_str = f"{collection_info.dim}d"
                else:
                    vector_size_str = "unknown"

                result.append(f"**{collection}**\n")
                result.append(f"  - Chunks indexed: {points_count:,}\n")
                result.append(f"  - Vector dimensions: {vector_size_str}\n")
                result.append("\n")
            except Exception as e:
                result.append(f"**{collection}**\n")
                result.append(f"  - Error: {str(e)}\n\n")

        result.append("-" * 80 + "\n")
        result.append(f"\nTotal collections: {len(collections)}\n")
        result.append("\nNote: The standard embedding dimension is 3072 for text-embedding-3-large model.")

        return ''.join(result)

    except Exception as e:
        error_msg = f"Error listing collections: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"



@with_metrics("stop_indexing")
@mcp.tool()
async def stop_indexing(
    force: bool = False
) -> str:
    """Stop the currently running background indexing process.
    
    Gracefully terminates the indexer by sending SIGTERM. If the process
    does not stop within 5 seconds, sends SIGKILL (force kill).
    Also kills any orphaned indexer child processes.
    Use force=True to skip the graceful SIGTERM and kill immediately.
    """
    import signal
    
    try:
        # Find all running indexer processes (not just the one in PID file)
        stopped_pids = []
        errors = []
        
        # 1. Try to stop the tracked process from PID file
        pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
        tracked_pid = None
        collection_name = "unknown"
        workspace_root = "unknown"
        
        if pid_file.exists():
            with Path(pid_file).open('r') as f:
                pid_info = json.load(f)
            tracked_pid = pid_info.get("pid")
            collection_name = pid_info.get("collection_name", "unknown")
            workspace_root = pid_info.get("workspace_root", "unknown")
        
        # 2. Build kill list from tracked PID + its children only
        indexer_pids = []
        if tracked_pid:
            try:
                proc = psutil.Process(tracked_pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    indexer_pids.append(tracked_pid)
                    for child in proc.children(recursive=True):
                        indexer_pids.append(child.pid)
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                indexer_pids.append(tracked_pid)
        
        if not indexer_pids:
            # Clean up stale PID file
            if pid_file.exists():
                pid_file.unlink(missing_ok=True)
            return "No running indexer processes found. Nothing to stop."
        
        # 3. Kill all found indexer processes
        for pid in indexer_pids:
            try:
                proc = psutil.Process(pid)
                
                if force:
                    proc.kill()
                    stopped_pids.append(f"{pid} (force killed)")
                else:
                    # Graceful: SIGTERM first
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                        stopped_pids.append(f"{pid} (stopped gracefully)")
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=3)
                        stopped_pids.append(f"{pid} (force killed after timeout)")
            except psutil.NoSuchProcess:
                stopped_pids.append(f"{pid} (already gone)")
            except Exception as e:
                errors.append(f"{pid}: {str(e)}")
        
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        
        result_parts = [
            f"Stopped {len(stopped_pids)} indexer process(es):",
            "\n".join(f"  - PID {p}" for p in stopped_pids),
        ]
        if tracked_pid:
            result_parts.append(f"Collection: {collection_name}, Workspace: {workspace_root}")
        if errors:
            result_parts.append(f"Errors: {' | '.join(errors)}")
        
        return "\n".join(result_parts)
            
    except Exception as e:
        error_msg = f"Error stopping indexing process: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


@with_metrics("clear_index")
@mcp.tool()
async def clear_index(
    collection_name: str = "folder.to.index-application-code",
    confirm: bool = False,
    include_lanes: bool = False,  # P2 lanes: also delete every <collection>-<slug> lane
) -> str:
    """Clear all indexed code from Qdrant vector database. Only use this when you are CERTAIN an index needs to be deleted."""
    _client_id, _allowed = _caller_rag_access()
    if not _collection_allowed(collection_name, _allowed):
        return f"Error: Collection '{collection_name}' is not granted to your account."
    _err = validate_collection_name(collection_name)
    if _err:
        return _err
    try:
        if not vector_store:
            return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."

        # Check if user wants to delete ALL collections
        delete_all = collection_name.upper() == "ALL"

        # P2 lanes: collections belonging to this base (base itself + lanes)
        lane_collections = list_lanes(vector_store, collection_name)
        lane_only = [c for c in lane_collections if c != collection_name]

        # Safety check - require explicit confirmation
        if not confirm:
            if delete_all:
                # Get all collections
                collections = vector_store.list_collections()
                collection_list = "\n".join([f"  - {c}" for c in collections])

                return f"""CONFIRMATION REQUIRED - DELETE ALL COLLECTIONS

You are about to DELETE EVERYTHING from Qdrant, including:
{collection_list}

This action is IRREVERSIBLE and will:
- Delete ALL indexed code chunks from ALL collections
- Remove ALL embeddings (including Roo Code indexing)
- Completely wipe the vector database

To confirm, call this tool again with `confirm: true`

Example:
```
clear_index(collection_name="ALL", confirm=true)
```
"""
            else:
                lane_note = ""
                if lane_only and not include_lanes:
                    lane_list = "\n".join([f"  - {c}" for c in lane_only])
                    lane_note = f"""

Note: `{collection_name}` has {len(lane_only)} embedding lane(s) that will NOT be deleted:
{lane_list}
Add `include_lanes: true` to delete them too.
"""
                return f"""CONFIRMATION REQUIRED

You are about to DELETE ALL DATA from collection: `{collection_name}`
{lane_note}
This action is IRREVERSIBLE and will:
- Delete all indexed code chunks
- Remove all embeddings
- Clear the vector database

To confirm, call this tool again with `confirm: true`

Example:
```
clear_index(collection_name="{collection_name}", confirm=true)
```
"""

        # Delete ALL collections
        if delete_all:
            collections = vector_store.list_collections()
            total_deleted = 0
            deleted_names = []

            for collection in collections:
                try:
                    collection_info = vector_store.get_collection(collection)
                    points_count = collection_info.points_count
                    vector_store.delete_collection(collection)
                    total_deleted += points_count
                    deleted_names.append(f"  - {collection} ({points_count:,} chunks)")
                    logger.info(f"Deleted collection '{collection}' with {points_count} chunks")
                except Exception as e:
                    deleted_names.append(f"  - {collection} (error: {str(e)})")

            deleted_list = "\n".join(deleted_names)
            result = f"""All Collections Cleared Successfully

Deleted {len(collections)} collections:
{deleted_list}

Total chunks deleted: {total_deleted:,}
Status: Qdrant completely wiped clean

You can now start fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="your-application-code")
```
"""
            logger.info(f"Cleared ALL collections - total {total_deleted} chunks deleted")
            return result

        # Delete single collection
        try:
            collection_info = vector_store.get_collection(collection_name)
            points_count = collection_info.points_count

            # Delete the collection
            vector_store.delete_collection(collection_name)

            # P2 lanes: optionally sweep the base's lane collections too
            lane_results = []
            if include_lanes:
                for lane in lane_only:
                    try:
                        lane_info = vector_store.get_collection(lane)
                        vector_store.delete_collection(lane)
                        lane_results.append(f"  - {lane} ({lane_info.points_count:,} chunks)")
                    except Exception as lane_e:
                        lane_results.append(f"  - {lane} (error: {lane_e})")

            lane_section = ""
            if lane_results:
                lane_section = f"\nLane collections deleted:\n" + "\n".join(lane_results)

            result = f"""Index Cleared Successfully

Collection: `{collection_name}`
Deleted: {points_count:,} code chunks{lane_section}
Status: Collection removed from Qdrant

The vector database has been wiped clean. You can now start a fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace")
```
"""
            logger.info(f"Cleared collection '{collection_name}' - deleted {points_count} chunks")
            return result

        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                return f"""Collection Not Found

Collection `{collection_name}` does not exist in Qdrant.

Available actions:
- Start indexing: `start_indexing(workspace_root="/path/to/your/workspace")`
- Check progress: `check_indexing_progress()`
"""
            else:
                raise e

    except Exception as e:
        error_msg = f"Error clearing index: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================

fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", MGMT_PORT))

    custom_extensions = [
        Extension(
            name="vector_db_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "connected": {"type": "boolean"},
                        "collections": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            handler=get_vector_db_stats,
            metadata={"description": "Vector database connection and collection info", "category": "metrics"}
        ),
        Extension(
            name="embedding_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "embedding_calls": {"type": "integer"}
                    }
                }
            },
            handler=get_embedding_stats,
            metadata={"description": "Embedding provider statistics", "category": "metrics"}
        ),
        Extension(
            name="collection_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "points_count": {"type": "integer"}
                    }
                }
            },
            handler=get_collection_stats,
            metadata={"description": "Collection statistics", "category": "metrics"}
        ),
        Extension(
            name="reindex",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"}
                    }
                },
                "output": {"type": "object", "properties": {"success": {"type": "boolean"}}}
            },
            handler=reindex,
            metadata={"description": "Trigger reindexing of a collection", "category": "maintenance"}
        ),
        Extension(
            name="list_collections",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object"}
            },
            handler=list_collections_handler,
            metadata={"description": "List all indexed collections with stats", "category": "collections"}
        ),
        Extension(
            name="check_indexing_progress",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {"pid": {"type": "integer", "description": "Optional PID to check"}}},
                "output": {"type": "object"}
            },
            handler=check_indexing_progress_handler,
            metadata={"description": "Show current indexing progress with ETA", "category": "collections"}
        ),
        Extension(
            name="start_indexing",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {
                    "type": "object",
                    "properties": {
                        "workspace_root": {"type": "string", "description": "Path to workspace to index"},
                        "collection_name": {"type": "string", "description": "Target collection name"},
                        "embedding_model": {"type": "string", "enum": ["auto", "fast", "high-quality", "nvidia-free"], "description": "Embedding model preset"}
                    },
                    "required": ["workspace_root"]
                },
                "output": {"type": "object"}
            },
            handler=start_indexing_handler,
            metadata={"description": "Start indexing a workspace into a collection", "category": "collections"}
        ),
    ]

    if registry is not None:
        # Use launcher's registry
        fef_registry = registry
        fef_manager = ToolExtensionManager(TOOL_NAME)
        register_common_extensions(TOOL_NAME, fef_registry, fef_manager)
        for ext in custom_extensions:
            fef_registry.register(TOOL_NAME, ext)
        fef_http_server = None
        logger.info(f"[{TOOL_NAME}] FEF V3 registered with launcher's registry")
    else:
        # Standalone mode
        fef_manager, fef_registry, fef_http_server = setup_tool_extensions(
            tool_name=TOOL_NAME,
            mgmt_port=mgmt_port,
            custom_extensions=custom_extensions
        )
        logger.info(f"[{TOOL_NAME}] FEF V3 standalone mode on port {mgmt_port}")

    fef_setup_done = True


# ============================================================================
# Lifespan
# ============================================================================



@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup/shutdown."""

    logger.info(f"{TOOL_NAME} FastMCP server starting on port {MCP_PORT}...")

    # Setup FEF V3 if not done by launcher
    if not fef_setup_done:
        setup_extensions(registry=None)

    # Start FEF V3 management server if standalone
    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.start()
            logger.info("FEF V3 management server started")
        except Exception as e:
            logger.warning(f"Failed to start FEF V3 management server: {e}")

    yield

    logger.info(f"{TOOL_NAME} FastMCP server shutting down...")
    if fef_http_server:
        try:
            await fef_http_server.stop()
        except Exception:
            pass


# ============================================================================
# App Export
# Transports: streamable-http (default, /mcp) or sse (legacy compat, /sse + /messages)
# ============================================================================

from tools.shared.server_factory import get_transport_app
from tools.shared.function_masks import apply_function_masks

apply_function_masks(mcp, TOOL_NAME)

app = get_transport_app(mcp)


# ============================================================================
# Exports for Launcher
# ============================================================================

__all__ = ["app", "setup_extensions", "mcp"]


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {TOOL_NAME} FastMCP server (multi-transport: /mcp, /mcp-stateless, /sse)")
    logger.info(f"  MCP port: {MCP_PORT}")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://localhost:{MGMT_PORT}")

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )