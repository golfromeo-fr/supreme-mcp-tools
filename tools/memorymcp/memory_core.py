#!/usr/bin/env python3
"""
Memory MCP Core - Shared configuration, utilities, and initialization.

This module contains:
- Global imports and dependency checking
- Logging configuration
- .env loading from root
- Backend initialization (via SqlStore + VectorStore ABCs)
- Local embedding model management
- Port configuration
- FastMCP instance creation
- Core utility functions (get_now_iso, get_memory_id, scroll_all, etc.)

This is imported by all other memorymcp modules.

Phase 5 cleanup: removed direct pg_store / qdrant_client references.
Use get_sql_store() and get_vector_store() from the shared ABCs.
"""

import sys
import os
import re
import logging
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ============================================================================
# Dependency Checking
# ============================================================================

try:
    import anyio
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Path Setup
# ============================================================================

# Ensure tool directory is on sys.path
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Add parent (tools/) to path for shared imports
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ============================================================================
# Import Shared Utilities
# ============================================================================

from shared.memory_models import (
    MemoryType, RetentionPolicy, Sensitivity,
    MemoryItem, MemoryHit,
)
from shared.relevance_scorer import (
    ScoringWeights, score_relevance, compute_recency_decay, compute_usage_boost,
)
from shared.pii_redactor import redact_sensitive_text, check_sensitivity, get_redactor
from shared.sql_store import get_sql_store
from shared.vector_store import get_vector_store

# ============================================================================
# Logging Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "memorymcp.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memorymcp")

# Load .env from root
root_env = SCRIPT_DIR.parent.parent / ".env"
if root_env.exists():
    from dotenv import load_dotenv
    load_dotenv(root_env)
    logger.info(f"Loaded configuration from {root_env}")

# ============================================================================
# Tool Configuration
# ============================================================================

TOOL_NAME = "memorymcp"
COLLECTION_NAME = os.getenv("MEMORY_COLLECTION", "memory-store")

# ============================================================================
# Backend Initialization (Phase 2/3 abstraction)
# ============================================================================

# Vector store: resolved from env/config via the factory
vector_store = get_vector_store()
if vector_store is not None:
    logger.info(f"VectorStore initialized: {type(vector_store).__name__}")
    # Ensure the default collection exists
    try:
        vector_store.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists")
    except Exception:
        try:
            embedding_dim = int(os.getenv('EMBEDDING_DIM', '1024'))
            vector_store.ensure_collection(
                COLLECTION_NAME, dense_dim=embedding_dim, sparse=False,
            )
            logger.info(f"Created collection '{COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"Could not create collection: {e}")
else:
    logger.info("No vector backend configured")

# SQL store: resolved from env/config via the factory
sql_store = get_sql_store()
if sql_store is not None and sql_store.is_available:
    logger.info(f"SqlStore initialized: {type(sql_store).__name__}")
else:
    logger.info("No SQL backend configured (vector-only mode)")

# ============================================================================
# Local Embeddings
# ============================================================================

LOCAL_EMBEDDINGS_AVAILABLE = False
_local_embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Install for local embeddings.")

def get_local_embedding_model():
    """Get or create the local embedding model singleton."""
    global _local_embedding_model
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        return None
    if _local_embedding_model is None:
        model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-m3')
        _local_embedding_model = SentenceTransformer(model_name)
    return _local_embedding_model

def generate_embedding(text: str) -> list[float] | None:
    """Generate embedding vector for text using local model."""
    model = get_local_embedding_model()
    if model is None:
        return None
    embeddings = model.encode([text])
    return embeddings[0].tolist()

# ============================================================================
# Port Configuration
# ============================================================================

try:
    from launcher.launcher_config import load_ports_config
    ports_config = load_ports_config()
    MCP_PORT = int(os.environ.get("MCP_PORT", ports_config["assignments"]["mcp"][TOOL_NAME]))
    MGMT_PORT = int(os.environ.get("MCP_MGMT_PORT", ports_config["assignments"]["mgmt"][TOOL_NAME]))
except Exception as e:
    logger.warning(f"Could not load ports config: {e}")
    MCP_PORT = int(os.environ.get("MCP_PORT", "8005"))
    MGMT_PORT = int(os.environ.get("MCP_MGMT_PORT", "8105"))

# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server
from tools.shared.migrate_mcp import register_migrate_tools

mcp = create_fastmcp_server(TOOL_NAME)

# Register backend migration tools (migrateMemoryBackend, verifyBackendParity)
register_migrate_tools(mcp)


# ============================================================================
# Core Utility Functions
# ============================================================================

def get_now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def get_memory_id() -> str:
    """Generate a new UUID for a memory item."""
    return str(uuid.uuid4())

def scroll_all(collection_name: str, **kwargs) -> list:
    """Scroll through all points in a collection, handling pagination."""
    from shared.store_models import Filter
    payload_filter = None
    if "filter" in kwargs:
        payload_filter = kwargs.pop("filter")
    all_points = []
    offset = None
    while True:
        results, next_offset = vector_store.scroll(
            collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            filter=payload_filter,
        )
        all_points.extend(results)
        if not next_offset or not results:
            break
        offset = next_offset
    return all_points

def parse_memory_type(type_str: str | None) -> MemoryType:
    """Parse a string into a MemoryType enum, defaulting to CONCEPT."""
    if not type_str:
        return MemoryType.CONCEPT
    try:
        return MemoryType(type_str.lower())
    except ValueError:
        return MemoryType.CONCEPT

def memory_item_to_payload(item: MemoryItem) -> dict:
    """Convert a MemoryItem to Qdrant payload format."""
    return {
        "text": item.text,
        "memory_type": item.type.value if isinstance(item.type, MemoryType) else item.type,
        "source": item.source,
        "path": item.path,
        "commit": item.commit,
        "file_range": item.file_range,
        "agent_id": item.agent_id,
        "created_at": item.timestamp or get_now_iso(),
        "last_accessed": item.timestamp or get_now_iso(),
        "usage_count": 0,
        "retention_policy": item.retention_policy.value if isinstance(item.retention_policy, RetentionPolicy) else item.retention_policy,
        "raw_object_key": item.raw_object_key,
        "provenance": {},
        "sensitivity": "low",
        "tags": item.tags,
        "text_preview": item.text[:200] if len(item.text) > 200 else item.text,
    }

def payload_to_memory_hit(payload: dict, score: float = 0.0) -> MemoryHit:
    """Convert a Qdrant payload to a MemoryHit with scoring."""
    return MemoryHit(
        id=payload.get("id", ""),
        text=payload.get("text", ""),
        type=parse_memory_type(payload.get("memory_type")),
        source=payload.get("source", ""),
        tags=payload.get("tags", []),
        score=score,
        recency_score=compute_recency_decay(payload.get("last_accessed")),
        usage_boost=compute_usage_boost(payload.get("usage_count", 0)),
        created_at=payload.get("created_at"),
        last_accessed=payload.get("last_accessed"),
        usage_count=payload.get("usage_count", 0),
        provenance=payload.get("provenance"),
    )