#!/usr/bin/env python3
"""
RAG MCP Server - FastMCP Implementation
Provides tools for semantic code search, sparse search, and code indexing using Qdrant.

FEF V3 Integration preserved from original implementation.
"""
import sys
import os
import logging
import time
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
    from indexer.sparse_vector_gen import generate_sparse_vector, get_global_generator
    SPARSE_VECTORS_AVAILABLE = True
except ImportError:
    SPARSE_VECTORS_AVAILABLE = False
    logging.warning("Sparse vector generator not available. Install sparse_vector_gen.py to enable sparse search.")

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

# Load configuration from root .env
root_env = SCRIPT_DIR.parent.parent / ".env"
if root_env.exists():
    load_dotenv(root_env)
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
}

# Global model cache
_local_embedding_model = None
_local_embedding_model_name = None


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

    if _local_embedding_model is not None and _local_embedding_model_name == model_name:
        logger.debug(f"Using cached local embedding model: {model_name}")
        return _local_embedding_model

    model_config = LOCAL_EMBEDDING_MODELS.get(model_name)
    if not model_config:
        logger.warning(f"Unknown local embedding model: {model_name}, falling back to bge-m3")
        model_config = LOCAL_EMBEDDING_MODELS['bge-m3']

    try:
        logger.info(f"Loading local embedding model: {model_config['model_name']}")
        _local_embedding_model = SentenceTransformer(model_config['model_name']).to(model_config['device'])
        _local_embedding_model_name = model_name
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

sys.path.insert(0, (Path(__file__).parent / ".." / "..").resolve())

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
    "min_search_time_ms": float("inf"),
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
        if elapsed_ms < ragmcp_metrics["min_search_time_ms"]:
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
    if qdrant_client:
        try:
            cols = qdrant_client.get_collections().collections
            collections = [c.name for c in cols]
        except Exception:
            pass

    return {
        "connected": qdrant_client is not None,
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
    if not qdrant_client:
        return {"error": "Qdrant client not initialized"}

    try:
        collection_info = qdrant_client.get_collection(collection_name)
        return {
            "collection": collection_name,
            "points_count": collection_info.points_count,
            "vectors_size": collection_info.config.params.vectors.size
        }
    except Exception as e:
        # Return user-friendly message instead of raw error
        err_str = str(e)
        if "doesn't exist" in err_str or "Not found" in err_str:
            return {"error": f"Collection '{collection_name}' not found", "exists": False}
        return {"error": err_str}


def list_collections_handler(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: List all indexed collections with stats."""
    if not qdrant_client:
        return {"error": "Qdrant not connected", "collections": []}

    try:
        collections = qdrant_client.get_collections().collections
        result = {}
        for coll in collections:
            info = qdrant_client.get_collection(coll.name)
            vectors_config = info.config.params.vectors

            # Handle both single vector and hybrid (dict) configurations
            if isinstance(vectors_config, dict):
                dims_parts = []
                for name, cfg in vectors_config.items():
                    if hasattr(cfg, 'size'):
                        dims_parts.append(f"{name}={cfg.size}d")
                    else:
                        dims_parts.append(f"{name}=sparse")
                dims = ", ".join(dims_parts) if dims_parts else "unknown"
            else:
                dims = f"{vectors_config.size}d" if vectors_config and hasattr(vectors_config, 'size') else "unknown"

            result[coll.name] = f"{info.points_count:,} chunks @ {dims}"

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

    if not workspace_root:
        return {"success": False, "error": "workspace_root is required"}

    if not qdrant_client:
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

        env = os.environ.copy()
        env['EMBEDDING_PROVIDER'] = preset['provider']
        if preset['provider'] == 'local':
            env['LOCAL_EMBEDDING_MODEL'] = preset['model']

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
        info = qdrant_client.get_collection(collection_name)
        vectors = info.config.params.vectors

        if isinstance(vectors, dict):
            available = {name: cfg.size for name, cfg in vectors.items() if hasattr(cfg, 'size')}
        else:
            available = {"default": vectors.size} if vectors else {}

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
    """Get the dense vector dimensions from a collection info object."""
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        if 'dense' in vectors:
            return vectors['dense'].size
        return None
    return vectors.size if vectors else None


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
                        "embedding_model": {"type": "string", "enum": ["auto", "fast", "high-quality"], "description": "Embedding model preset"}
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

logger.info("Initializing Qdrant client for semantic code search...")
try:
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Qdrant client connected to {qdrant_host}:{qdrant_port}")
except Exception as e:
    logger.warning(f"Could not initialize Qdrant client: {e}")
    qdrant_client = None

# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server, DEFAULT_HOST

mcp = create_fastmcp_server(TOOL_NAME)


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
    logger.debug(f"Processing search_code tool: query={query}, limit={limit}")

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Code search is unavailable."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not query:
        logger.error("Missing required parameter: query")
        raise ValueError("Missing argument: query")

    try:
        logger.info(f"Searching code: '{query}' (limit={limit}, collection={collection_name})")

        # Detect collection's vector configuration
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' config: {collection_info.config}")
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")

        # Generate embedding for search query
        embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'azure')

        if embedding_provider == 'azure':
            import httpx
            azure_api_url = os.getenv('AZURE_EMBEDDING_API_URL',
                                     'https://put.your.API.gateway.ai/v1/embeddings')
            azure_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
            api_key = os.getenv('AI_API_KEY', '')

            if not api_key:
                return "Error: AI_API_KEY not set. Cannot generate search embedding."

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    azure_api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json={
                        "input": [query],
                        "model": azure_model
                    }
                )
                response.raise_for_status()
                data = response.json()
                query_vector = data['data'][0]['embedding']
        else:
            if not LOCAL_EMBEDDINGS_AVAILABLE:
                return "Error: Local embeddings module not available. Install sentence-transformers."

            try:
                model_info = LOCAL_EMBEDDING_MODELS.get(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODELS['bge-m3'])
                logger.info(f"Using local embeddings for search query (model: {LOCAL_EMBEDDING_MODEL} - {model_info['description']})")
                query_embeddings = generate_local_embeddings([query], model_name=LOCAL_EMBEDDING_MODEL)

                if query_embeddings is None or len(query_embeddings) == 0:
                    return "Error: Failed to generate local embedding for query."

                query_vector = query_embeddings[0].tolist()
                logger.info(f"Generated local embedding: dimension={len(query_vector)}")

            except Exception as e:
                return f"Error generating local embedding: {str(e)}"

        # Build filter if file_type or function_name specified
        search_filter = None
        conditions = []

        if file_type:
            conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

        if function_name:
            conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

        if conditions:
            search_filter = Filter(must=conditions)

        # Perform search
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            if isinstance(vectors_config, dict):
                vector_names = list(vectors_config.keys())
                logger.info(f"Collection has named vectors: {vector_names}")

                vector_dim = len(query_vector)
                vector_name = None

                for vname in vector_names:
                    vconfig = vectors_config[vname]
                    if hasattr(vconfig, 'size') and vconfig.size == vector_dim:
                        vector_name = vname
                        logger.info(f"Using vector '{vector_name}' (matches dimension {vector_dim})")
                        break

                if not vector_name:
                    vector_name = vector_names[0]
                    logger.warning(f"No matching vector dimension, using first: '{vector_name}'")

                query_response = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
            else:
                logger.info(f"Collection has single unnamed vector")
                query_response = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
        except Exception as e:
            logger.error(f"Error detecting vector config: {e}, trying without 'using' parameter")
            query_response = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

        search_results = query_response.points

        # Format results
        if not search_results:
            return "No results found."

        formatted_results = []
        formatted_results.append(f"Found {len(search_results)} relevant code chunks:\n")
        formatted_results.append("=" * 80 + "\n")

        for i, hit in enumerate(search_results, 1):
            payload = hit.payload
            score = hit.score

            formatted_results.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
            formatted_results.append(f"File: {payload.get('filePath', 'Unknown')}\n")
            formatted_results.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
            formatted_results.append(f"Type: {payload.get('fileType', 'Unknown')}\n")

            if payload.get('functionName'):
                formatted_results.append(f"Function: {payload['functionName']}\n")
            if payload.get('chunkType'):
                formatted_results.append(f"Chunk Type: {payload['chunkType']}\n")

            formatted_results.append("\nCode:\n```\n")
            code_chunk = payload.get('codeChunk', '')
            lines = code_chunk.split('\n')
            if len(lines) > 50:
                formatted_results.append('\n'.join(lines[:50]))
                formatted_results.append(f"\n... ({len(lines) - 50} more lines)")
            else:
                formatted_results.append(code_chunk)
            formatted_results.append("\n```\n")
            formatted_results.append("-" * 80 + "\n")

        result_text = ''.join(formatted_results)
        logger.info(f"Search completed: {len(search_results)} results")
        return result_text

    except Exception as e:
        error_msg = f"Error searching code: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


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
    logger.debug(f"Processing search_code_sparse tool: query={query}, limit={limit}")

    if not SPARSE_VECTORS_AVAILABLE:
        error_msg = "Sparse vector search not available. Missing sparse_vector_gen.py module."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Code search is unavailable."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not query:
        logger.error("Missing required parameter: query")
        raise ValueError("Missing argument: query")

    try:
        logger.info(f"Sparse searching code: '{query}' (limit={limit}, collection={collection_name})")

        # Generate sparse vector for the query
        query_metadata = {
            'language': file_type if file_type else 'unknown'
        }
        query_sparse_vec = generate_sparse_vector(query, query_metadata)

        if not query_sparse_vec:
            return "Error: Failed to generate sparse vector for query."

        # Build filter if file_type or function_name specified
        search_filter = None
        conditions = []

        if file_type:
            conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

        if function_name:
            conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

        if conditions:
            search_filter = Filter(must=conditions)

        # Perform sparse search
        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=list(query_sparse_vec.keys()),
                values=list(query_sparse_vec.values())
            ),
            using="sparse",
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        search_results = query_response.points

        # Format results
        if not search_results:
            return "No results found."

        formatted_results = []
        formatted_results.append(f"Found {len(search_results)} relevant code chunks (sparse/lexical search):\n")
        formatted_results.append("=" * 80 + "\n")

        for i, hit in enumerate(search_results, 1):
            payload = hit.payload
            score = hit.score

            formatted_results.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
            formatted_results.append(f"File: {payload.get('filePath', 'Unknown')}\n")
            formatted_results.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
            formatted_results.append(f"Type: {payload.get('fileType', 'Unknown')}\n")

            if payload.get('functionName'):
                formatted_results.append(f"Function: {payload['functionName']}\n")
            if payload.get('chunkType'):
                formatted_results.append(f"Chunk Type: {payload['chunkType']}\n")

            formatted_results.append("\nCode:\n```\n")
            code_chunk = payload.get('codeChunk', '')
            lines = code_chunk.split('\n')
            if len(lines) > 50:
                formatted_results.append('\n'.join(lines[:50]))
                formatted_results.append(f"\n... ({len(lines) - 50} more lines)")
            else:
                formatted_results.append(code_chunk)
            formatted_results.append("\n```\n")
            formatted_results.append("-" * 80 + "\n")

        result_text = ''.join(formatted_results)
        logger.info(f"Sparse search completed: {len(search_results)} results")
        return result_text

    except Exception as e:
        error_msg = f"Error in sparse code search: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return f"Error: {error_msg}"


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
    max_lines: int = 50
) -> str:
    """
    Unified code search with automatic detection of collection capabilities.

    - mode='auto': Detect collection and use best available search method
    - mode='dense': Semantic search using embeddings (requires matching dimension)
    - mode='sparse': BM25 lexical search for exact identifiers
    - mode='hybrid': Combined dense + sparse (requires hybrid collection)

    copilot_format adds GitHub Copilot formatting:
    - 'comment': Inline comment block
    - 'sidebar': Sidebar context block
    """
    logger.debug(f"Processing unified search: query={query}, mode={mode}, copilot_format={copilot_format}")

    if not qdrant_client:
        return "Error: Qdrant client not initialized."

    if not query:
        raise ValueError("Missing argument: query")

    # Detect collection capabilities
    try:
        info = qdrant_client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        has_sparse = bool(info.config.params.sparse_vectors)

        if isinstance(vectors_config, dict):
            has_dense = "dense" in vectors_config
            has_sparse = has_sparse or "sparse" in vectors_config
        else:
            has_dense = vectors_config is not None
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


async def _search_dense(query: str, limit: int, collection_name: str,
                        file_type: str | None, function_name: str | None) -> str:
    """Internal: dense semantic search."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Generate embedding
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'azure')

    if embedding_provider == 'azure':
        import httpx
        azure_api_url = os.getenv('AZURE_EMBEDDING_API_URL',
                                  'https://put.your.API.gateway.ai/v1/embeddings')
        azure_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
        api_key = os.getenv('AI_API_KEY', '')

        if not api_key:
            return "Error: AI_API_KEY not set. Cannot generate search embedding."

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                azure_api_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"input": [query], "model": azure_model}
            )
            response.raise_for_status()
            data = response.json()
            query_vector = data['data'][0]['embedding']
    else:
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            return "Error: Local embeddings module not available. Install sentence-transformers."

        try:
            model_info = LOCAL_EMBEDDING_MODELS.get(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODELS['bge-m3'])
            query_embeddings = generate_local_embeddings([query], model_name=LOCAL_EMBEDDING_MODEL)
            if query_embeddings is None or len(query_embeddings) == 0:
                return "Error: Failed to generate local embedding for query."
            query_vector = query_embeddings[0].tolist()
        except Exception as e:
            return f"Error generating local embedding: {str(e)}"

    # Validate dimension matches collection
    validation = validate_search_request(collection_name, query_vector)
    if not validation.get('valid', True) and 'error' in validation:
        return f"Error: {validation['error']}"
    if validation.get('warning'):
        logger.warning(validation['warning'])

    # Build filter
    conditions = []
    if file_type:
        conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))
    if function_name:
        conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))
    search_filter = Filter(must=conditions) if conditions else None

    # Perform search
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        vectors_config = collection_info.config.params.vectors

        if isinstance(vectors_config, dict):
            vector_names = list(vectors_config.keys())
            vector_dim = len(query_vector)
            vector_name = None

            for vname in vector_names:
                vconfig = vectors_config[vname]
                if hasattr(vconfig, 'size') and vconfig.size == vector_dim:
                    vector_name = vname
                    break

            if not vector_name:
                vector_name = vector_names[0]

            query_response = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
        else:
            query_response = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
    except Exception as e:
        return f"Error in dense search: {str(e)}"

    return _format_search_results(query_response.points)


async def _search_sparse(query: str, limit: int, collection_name: str,
                         file_type: str | None, function_name: str | None) -> str:
    """Internal: sparse BM25 lexical search."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if not SPARSE_VECTORS_AVAILABLE:
        return "Error: Sparse vector search not available. Missing sparse_vector_gen.py module."

    query_metadata = {'language': file_type if file_type else 'unknown'}
    query_sparse_vec = generate_sparse_vector(query, query_metadata)

    if not query_sparse_vec:
        return "Error: Failed to generate sparse vector for query."

    conditions = []
    if file_type:
        conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))
    if function_name:
        conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))
    search_filter = Filter(must=conditions) if conditions else None

    try:
        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=SparseVector(indices=list(query_sparse_vec.keys()), values=list(query_sparse_vec.values())),
            using="sparse",
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
    except Exception as e:
        return f"Error in sparse search: {str(e)}"

    return _format_search_results(query_response.points)


async def _search_hybrid(query: str, limit: int, collection_name: str,
                         file_type: str | None, function_name: str | None) -> str:
    """Internal: combined dense + sparse search."""

    # Get both search results and combine
    try:
        dense_results = (await _search_dense(query, limit, collection_name, file_type, function_name))
        sparse_results = (await _search_sparse(query, limit, collection_name, file_type, function_name))

        # Return hybrid summary (both searches performed)
        return f"Hybrid search results:\n\n=== Dense Search ===\n{dense_results}\n\n=== Sparse Search ===\n{sparse_results}"
    except Exception as e:
        return f"Error in hybrid search: {str(e)}"


async def _search_copilot(query: str, limit: int, collection_name: str,
                          file_type: str | None, function_name: str | None,
                          copilot_format: str, language: str, max_lines: int) -> str:
    """Internal: search with copilot formatting."""
    if not COPILOT_INJECTOR_AVAILABLE:
        return "Error: Copilot context injector not available. Missing copilot_context_injector.py module."

    if not SPARSE_VECTORS_AVAILABLE:
        return "Error: Sparse vector search required for copilot context. Missing sparse_vector_gen.py module."

    try:
        injector = get_injector(max_context_lines=max_lines)

        # Extract keywords
        keywords = injector.extract_keywords_from_context(query)
        if not keywords:
            return injector._format_no_context(language)

        search_query = ' '.join(keywords[:5])

        # Sparse search
        query_metadata = {'language': 'unknown'}
        query_sparse_vec = generate_sparse_vector(search_query, query_metadata)

        if not query_sparse_vec:
            return injector._format_no_context(language)

        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=SparseVector(indices=list(query_sparse_vec.keys()), values=list(query_sparse_vec.values())),
            using="sparse",
            limit=limit,
            with_payload=True
        )

        chunks = [hit.payload for hit in query_response.points]

        if not chunks:
            return injector._format_no_context(language)

        if copilot_format == "sidebar":
            return injector.format_sidebar_context(chunks, language)
        else:
            return injector.format_context_comment(chunks, language)

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

    if not COPILOT_INJECTOR_AVAILABLE:
        error_msg = "Copilot context injector not available. Missing copilot_context_injector.py module."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not SPARSE_VECTORS_AVAILABLE:
        error_msg = "Sparse vector search required for context injection. Missing sparse_vector_gen.py module."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Context injection is unavailable."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if not current_context:
        logger.error("Missing required parameter: current_context")
        raise ValueError("Missing argument: current_context")

    try:
        logger.info(f"Getting Copilot context for: '{current_context[:50]}...' (format={format}, limit={limit})")

        # Get context injector instance
        injector = get_injector(max_context_lines=max_lines)

        # Extract keywords from current context
        keywords = injector.extract_keywords_from_context(current_context)

        if not keywords:
            logger.warning("No keywords extracted from context")
            return injector._format_no_context(language)

        # Build search query from keywords (top 5 most relevant)
        search_query = ' '.join(keywords[:5])
        logger.debug(f"Extracted keywords for search: {keywords[:5]}")

        # Generate sparse vector for the query
        query_metadata = {'language': 'unknown'}
        query_sparse_vec = generate_sparse_vector(search_query, query_metadata)

        if not query_sparse_vec:
            logger.warning("Failed to generate sparse vector")
            return injector._format_no_context(language)

        # Perform sparse search to get structured results
        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=list(query_sparse_vec.keys()),
                values=list(query_sparse_vec.values())
            ),
            using="sparse",
            limit=limit,
            with_payload=True
        )
        search_results = query_response.points

        if not search_results:
            logger.warning("No code chunks found")
            return injector._format_no_context(language)

        # Extract chunk payloads
        chunks = [hit.payload for hit in search_results]

        logger.info(f"Found {len(chunks)} relevant chunks")

        # Format based on requested format
        if format == "comment":
            formatted_context = injector.format_context_comment(chunks, max_lines, language)
        elif format == "sidebar":
            formatted_context = injector.format_sidebar_context(chunks)
        else:
            formatted_context = injector.format_context_comment(chunks, max_lines, language)

        logger.info(f"Context formatted successfully ({len(formatted_context)} chars)")
        return formatted_context

    except Exception as e:
        error_msg = f"Error generating Copilot context: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {error_msg}"


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
    logger.debug(f"Processing index_code tool: workspace_root={workspace_root}, embedding_model={embedding_model}")

    if not qdrant_client:
        return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT environment variables."

    # Resolve preset
    preset = EMBEDDING_MODEL_PRESETS.get(embedding_model, EMBEDDING_MODEL_PRESETS["auto"])
    logger.info(f"Using embedding preset '{embedding_model}': provider={preset['provider']}, model={preset['model']}, dims={preset['dimensions']}")

    # Auto-create collection if it doesn't exist (with correct dimensions)
    collection_exists = False
    try:
        qdrant_client.get_collection(collection_name)
        collection_exists = True
        logger.info(f"Collection '{collection_name}' already exists")
    except Exception as e:
        if "Not found" in str(e) or "doesn't exist" in str(e):
            logger.info(f"Collection '{collection_name}' does not exist, will be auto-created by indexer")
            collection_exists = False
        else:
            return f"Error: Error checking collection: {str(e)}"

    # Validate embedding dimensions match existing collection
    if collection_exists:
        try:
            existing_info = qdrant_client.get_collection(collection_name)
            existing_dims = _get_collection_dimensions(existing_info)
            preset_dims = preset['dimensions']
            if existing_dims and preset_dims != existing_dims:
                logger.warning(f"Collection '{collection_name}' has {existing_dims}d vectors but preset '{embedding_model}' uses {preset_dims}d. "
                               f"Will use existing collection dimensions for incremental indexing.")
        except Exception as dim_e:
            logger.warning(f"Could not validate collection dimensions: {dim_e}")

    # If force=True and collection exists, warn but don't delete automatically
    if force and collection_exists:
        return f"""Warning: Collection Already Exists

Collection `{collection_name}` already exists. The `force` parameter is no longer used to delete existing collections for safety.

Options:
1. Continue indexing (recommended): Remove `force=true` and rerun. The indexer will skip already-indexed files.
2. Clear and reindex: First use `clear_index(collection_name="{collection_name}", confirm=true)` to delete the collection, then start indexing again.

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
        "--collection", collection_name
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

    # Create logs/ directory if it doesn't exist
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path in logs/ subfolder
    log_file = logs_dir / f"indexing_{collection_name}.log"

    # Prepare environment with embedding configuration from preset
    env = os.environ.copy()
    env['EMBEDDING_PROVIDER'] = preset['provider']
    if preset['provider'] == 'local':
        env['LOCAL_EMBEDDING_MODEL'] = preset['model']

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
            "collection_name": collection_name,
            "log_file": str(log_file),
            "started_at": time.time(),
            "embedding_model": embedding_model,
            "embedding_dimensions": preset['dimensions']
        }, f)

    result = f"""Indexing started successfully!

Process Information:
- PID: {pid}
- Workspace: {workspace_root}
- Collection: {collection_name}
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

        # Get Qdrant collection stats
        collection_stats = "Unknown"
        if qdrant_client:
            try:
                collection_info = qdrant_client.get_collection(collection_name)
                points_count = collection_info.points_count

                # Get vector dimensions
                vectors_config = collection_info.config.params.vectors
                if isinstance(vectors_config, dict):
                    vector_parts = []
                    for name, params in vectors_config.items():
                        if hasattr(params, 'size'):
                            vector_parts.append(f"{name}={params.size}d ({params.distance})")
                        else:
                            vector_parts.append(f"{name}=sparse")
                    vector_dims = ", ".join(vector_parts) if vector_parts else "unknown"
                else:
                    vector_dims = f"{vectors_config.size}d ({vectors_config.distance})" if vectors_config else "unknown"

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
        if not qdrant_client:
            return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."

        collections = qdrant_client.get_collections().collections

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

        for collection in collections:
            try:
                collection_info = qdrant_client.get_collection(collection.name)
                points_count = collection_info.points_count

                # Handle both single vector and hybrid named vectors (dict)
                vectors_config = collection_info.config.params.vectors
                if isinstance(vectors_config, dict):
                    # Hybrid collection with named vectors like {"dense": VectorParams, "sparse": SparseVectorParams}
                    vector_parts = []
                    for name, params in vectors_config.items():
                        if hasattr(params, 'size'):
                            vector_parts.append(f"{name}={params.size}d ({params.distance})")
                        else:
                            vector_parts.append(f"{name}=sparse")
                    vector_size_str = ", ".join(vector_parts)
                else:
                    vector_size_str = f"{vectors_config.size}d ({vectors_config.distance})"

                result.append(f"**{collection.name}**\n")
                result.append(f"  - Chunks indexed: {points_count:,}\n")
                result.append(f"  - Vector dimensions: {vector_size_str}\n")
                result.append("\n")
            except Exception as e:
                result.append(f"**{collection.name}**\n")
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
        
        # 2. Find ALL indexer processes (handles orphans too)
        indexer_pids = []
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline)
                if 'incremental_indexer' in cmdline_str:
                    indexer_pids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # If no processes found via psutil, try tracked PID
        if not indexer_pids and tracked_pid:
            try:
                proc = psutil.Process(tracked_pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    indexer_pids.append(tracked_pid)
            except psutil.NoSuchProcess:
                pass
        
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
    confirm: bool = False
) -> str:
    """Clear all indexed code from Qdrant vector database. Only use this when you are CERTAIN an index needs to be deleted."""
    try:
        if not qdrant_client:
            return "Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."

        # Check if user wants to delete ALL collections
        delete_all = collection_name.upper() == "ALL"

        # Safety check - require explicit confirmation
        if not confirm:
            if delete_all:
                # Get all collections
                collections = qdrant_client.get_collections().collections
                collection_list = "\n".join([f"  - {c.name}" for c in collections])

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
                return f"""CONFIRMATION REQUIRED

You are about to DELETE ALL DATA from collection: `{collection_name}`

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
            collections = qdrant_client.get_collections().collections
            total_deleted = 0
            deleted_names = []

            for collection in collections:
                try:
                    collection_info = qdrant_client.get_collection(collection.name)
                    points_count = collection_info.points_count
                    qdrant_client.delete_collection(collection.name)
                    total_deleted += points_count
                    deleted_names.append(f"  - {collection.name} ({points_count:,} chunks)")
                    logger.info(f"Deleted collection '{collection.name}' with {points_count} chunks")
                except Exception as e:
                    deleted_names.append(f"  - {collection.name} (error: {str(e)})")

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
            collection_info = qdrant_client.get_collection(collection_name)
            points_count = collection_info.points_count

            # Delete the collection
            qdrant_client.delete_collection(collection_name)

            result = f"""Index Cleared Successfully

Collection: `{collection_name}`
Deleted: {points_count:,} code chunks
Status: Collection removed from Qdrant

The vector database has been wiped clean. You can now start a fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="{collection_name}")
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
                        "embedding_model": {"type": "string", "enum": ["auto", "fast", "high-quality"], "description": "Embedding model preset"}
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
# Transport is selectable via MCP_TRANSPORT env var:
#   - "streamable-http" (default) → /mcp endpoint
#   - "sse"                    → /sse + /messages endpoints
# ============================================================================

from tools.shared.server_factory import get_transport_app

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

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http").lower()
    logger.info(f"Starting {TOOL_NAME} FastMCP server (transport: {transport})")
    logger.info(f"  MCP port: {MCP_PORT}")
    if transport == "sse":
        logger.info(f"  SSE endpoint: http://localhost:{MCP_PORT}/sse")
        logger.info(f"  Messages: http://localhost:{MCP_PORT}/messages")
    else:
        logger.info(f"  Streamable HTTP: http://localhost:{MCP_PORT}/mcp")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://localhost:{MGMT_PORT}")

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )