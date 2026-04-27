#!/usr/bin/env python3
"""
Memory MCP Server - FastMCP Implementation
A memory system for agentic coding with semantic search, provenance, and lifecycle management.

Features:
- Store/retrieve memories with automatic embedding
- Recency and usage-based relevance scoring
- Provenance tracking for agent-generated content
- PII redaction before storage
- TTL-based memory lifecycle management

💡 Tip: After discovering useful patterns, solving tricky bugs, or learning something novel
about the codebase, use upsertMemory to remember it. Future-you will thank you!
"""

import sys
import os
import re
import logging
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Check dependencies
try:
    import anyio
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    sys.exit(1)

# Ensure tool directory is on sys.path
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Add parent (tools/) to path for shared imports
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import shared utilities
from shared.memory_models import (
    MemoryType, RetentionPolicy, Sensitivity,
    MemoryItem, MemoryHit,
)
from shared.relevance_scorer import (
    ScoringWeights, score_relevance, compute_recency_decay, compute_usage_boost,
)
from shared.pii_redactor import redact_sensitive_text, check_sensitivity, get_redactor
from shared import pg_store

# Configure logging
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
# Configuration
# ============================================================================

TOOL_NAME = "memorymcp"
COLLECTION_NAME = os.getenv("MEMORY_COLLECTION", "memory-store")


# Qdrant connection
qdrant_client = None
try:
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Qdrant client connected to {qdrant_host}:{qdrant_port}")

    # Ensure collection exists
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists")
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'")
        from qdrant_client.models import VectorParams, Distance
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

except Exception as e:
    logger.warning(f"Could not initialize Qdrant client: {e}")

# PostgreSQL store (optional)
pg_store.init_pg()
if pg_store.is_available():
    logger.info("PostgreSQL store available for metadata and dedup")
else:
    logger.info("PostgreSQL not available, using Qdrant-only mode")

# Local embeddings
LOCAL_EMBEDDINGS_AVAILABLE = False
_local_embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Install for local embeddings.")

def get_local_embedding_model():
    global _local_embedding_model
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        return None
    if _local_embedding_model is None:
        model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-m3')
        _local_embedding_model = SentenceTransformer(model_name)
    return _local_embedding_model

def generate_embedding(text: str) -> list[float] | None:
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
# FastMCP Instance
# ============================================================================

mcp = FastMCP(
    TOOL_NAME,
    sse_path="/sse",
    streamable_http_path="/mcp",
)

# ============================================================================
# Utility Functions
# ============================================================================

def get_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_memory_id() -> str:
    return str(uuid.uuid4())


def scroll_all(collection_name: str, **kwargs) -> list:
    """Scroll through all points in a collection, handling pagination."""
    all_points = []
    offset = None
    while True:
        results, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            **kwargs,
        )
        all_points.extend(results)
        if not next_offset or not results:
            break
        offset = next_offset
    return all_points

def parse_memory_type(type_str: str | None) -> MemoryType:
    if not type_str:
        return MemoryType.CONCEPT
    try:
        return MemoryType(type_str.lower())
    except ValueError:
        return MemoryType.CONCEPT

def memory_item_to_payload(item: MemoryItem) -> dict:
    """Convert MemoryItem to Qdrant payload."""
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
    """Convert Qdrant payload to MemoryHit."""
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

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def upsertMemory(
    text: str,
    memory_type: str = "concept",
    source: str = "agent_action",
    tags: list[str] | None = None,
    path: str | None = None,
    commit: str | None = None,
    agent_id: str | None = None,
    retention_policy: str = "auto-delete",
    embedding: list[float] | None = None,
    memory_id: str | None = None,
) -> str:
    """
    Store or update a memory.

    💡 Tip: After discovering useful patterns, solving tricky bugs, or learning something
    novel about the codebase, use this to remember it. Future-you will thank you!

    Args:
        text: The memory content to store (required)
        memory_type: Type of memory - code_pattern, architectural_decision, trick, plan, lesson, concept, idea, decision
        source: Where this memory came from - URL, file path, or "agent_action"
        tags: List of tags for filtering, e.g., ["plan", "trick", "pattern"]
        path: File path if applicable
        commit: Git commit hash if applicable
        agent_id: Agent identifier if applicable
        retention_policy: "permanent", "temp", or "auto-delete" (default: auto-delete)
        embedding: Pre-computed embedding vector (optional, will auto-generate if not provided)
        memory_id: Existing memory ID to update (optional, creates new if omitted)

    Returns:
        Memory ID of stored or updated memory
    """
    logger.info(f"upsertMemory: {"update" if memory_id else "create"}, type={memory_type}, text_len={len(text)}")

    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    # Check sensitivity and redact if needed
    sensitivity = check_sensitivity(text)
    if sensitivity != "low":
        text, _ = redact_sensitive_text(text)
        logger.info(f"Redacted PII from memory, sensitivity={sensitivity}")

    # Generate embedding if not provided
    if embedding is None:
        embedding = generate_embedding(text)
        if embedding is None:
            return "Error: Could not generate embedding. Install sentence-transformers or provide embedding."

    # Use existing ID or generate a new one
    is_update = memory_id is not None
    if not is_update:
        memory_id = get_memory_id()
    now = get_now_iso()

    item = MemoryItem(
        id=memory_id,
        text=text,
        type=parse_memory_type(memory_type),
        source=source,
        tags=tags or [],
        path=path,
        commit=commit,
        agent_id=agent_id,
        timestamp=now,
        retention_policy=RetentionPolicy(retention_policy) if retention_policy else RetentionPolicy.AUTO_DELETE,
    )

    payload = memory_item_to_payload(item)
    payload["sensitivity"] = sensitivity

    # Upsert to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": memory_id,
                "vector": embedding,
                "payload": payload,
            }]
        )
        logger.info(f"{"Updated" if is_update else "Stored"} memory {memory_id}")

        # Also store in PostgreSQL for metadata/dedup (if available)
        if pg_store.is_available():
            actual_id = pg_store.upsert_memory(
                memory_id=memory_id,
                text=text,
                memory_type=memory_type,
                source=source,
                tags=tags or [],
                path=path,
                commit=commit,
                agent_id=agent_id,
                sensitivity=sensitivity,
                retention_policy=retention_policy,
            )
            if actual_id != memory_id:
                logger.info(f"PG dedup: remapping {memory_id} -> {actual_id}")
                return actual_id

        return memory_id

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def queryMemory(
    query: str,
    k: int = 10,
    memory_type: str | None = None,
    tags: list[str] | None = None,
    agent_id: str | None = None,
    recency_weight: float = 0.5,
) -> str:
    """
    Query memories with recency and usage weighting.

    💡 Tip: Before starting a new task, query memory for relevant past learnings.
    When stuck, search for similar problems others have solved.

    Args:
        query: Search query text
        k: Number of results to return (default: 10)
        memory_type: Filter by memory type (optional)
        tags: Filter by tags (optional)
        agent_id: Filter by agent ID (optional)
        recency_weight: Weight for recency vs semantic similarity (0-1, higher = more recency)

    Returns:
        Formatted list of memory hits with scores
    """
    logger.info(f"queryMemory: query={query[:50]}..., k={k}")

    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    # Generate query embedding
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return "Error: Could not generate query embedding"

    # Build filter
    conditions = []
    if memory_type:
        conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
    if agent_id:
        conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
    if tags:
        for tag in tags:
            conditions.append(FieldCondition(key="tags", match=MatchValue(value=tag)))

    search_filter = Filter(must=conditions) if conditions else None

    # Perform search
    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=k,
            query_filter=search_filter,
            with_payload=True,
        )

        hits = []
        weights = ScoringWeights()
        # Override alpha/beta based on recency_weight
        weights.alpha = 1.0 - recency_weight
        weights.beta = recency_weight

        for result in results.points:
            payload = result.payload
            payload["id"] = str(result.id)

            # Compute combined relevance score (pass Qdrant similarity as semantic_score)
            relevance = score_relevance(payload, query_embedding, weights, semantic_score=result.score)

            hit = payload_to_memory_hit(payload, result.score)
            hit.score = relevance  # Override with combined score
            hits.append(hit)

            # Update last_accessed and usage_count
            current_usage = payload.get("usage_count", 0)
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "last_accessed": get_now_iso(),
                    "usage_count": current_usage + 1,
                },
                points=[str(result.id)],
            )

        # Format output
        if not hits:
            return "No memories found matching query."

        output = f"Found {len(hits)} memories:\n\n"
        for i, hit in enumerate(hits, 1):
            output += f"{i}. [{hit.type.value}] (score: {hit.score:.3f})\n"
            output += f"   {hit.text[:150]}{'...' if len(hit.text) > 150 else ''}\n"
            if hit.tags:
                output += f"   Tags: {', '.join(hit.tags)}\n"
            output += f"   Accessed {hit.usage_count}x, last at {hit.last_accessed}\n\n"

        return output

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def getMemory(memory_id: str) -> str:
    """
    Get a specific memory by ID.

    💡 Tip: Use when you have a memory ID from a previous query and want the full content.

    Args:
        memory_id: UUID of the memory

    Returns:
        Full memory details
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        pg_mem = pg_store.get_memory(memory_id) if pg_store.is_available() else None

        results = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
        )

        if not results:
            return f"Memory not found: {memory_id}"

        payload = results[0].payload
        payload["id"] = str(results[0].id)

        # Update usage
        new_usage = (payload.get("usage_count", 0)) + 1
        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "last_accessed": get_now_iso(),
                "usage_count": new_usage,
            },
            points=[memory_id],
        )

        # Update payload with incremented values so display is accurate
        payload["usage_count"] = new_usage
        payload["last_accessed"] = get_now_iso()
        hit = payload_to_memory_hit(payload, 1.0)
        output = f"""Memory Details:
ID: {hit.id}
Type: {hit.type.value}
Source: {hit.source}
Tags: {', '.join(hit.tags) if hit.tags else 'none'}
Created: {hit.created_at}
Last Accessed: {hit.last_accessed}
Usage Count: {hit.usage_count}
Retention: {payload.get('retention_policy', 'unknown')}

Content:
{hit.text}
"""

        if payload.get('provenance'):
            output += f"\nProvenance: {json.dumps(payload['provenance'], indent=2)}"

        # Show edges (graph links)
        edges = payload.get("edges", [])
        forward = [e for e in edges if not e.get("relation", "").startswith("back:")]
        if forward:
            output += "\n\nLinks:"
            for e in forward:
                to_id = e.get("to", "")[:8]
                rel = e.get("relation", "related_to")
                lbl = e.get("label", "")
                output += f"\n  → {to_id}... [{rel}]"
                if lbl:
                    output += f" — {lbl}"

        return output

    except Exception as e:
        logger.error(f"Get memory failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def deleteMemory(memory_id: str) -> str:
    """
    Delete a memory by ID.

    💡 Tip: Use to remove outdated or incorrect memories.

    Args:
        memory_id: UUID of the memory to delete

    Returns:
        Success or error message
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[memory_id],
        )
        if pg_store.is_available():
            pg_store.delete_memory(memory_id)
        logger.info(f"Deleted memory {memory_id}")
        return f"Deleted memory: {memory_id}"

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def listMemoryTypes() -> str:
    """
    List available memory types.

    💡 Tip: Use to see what types of memories you can store. Each type serves
    a different purpose - patterns for code idioms, tricks for clever solutions,
    plans for project direction, lessons for things to avoid.

    Returns:
        List of memory types with descriptions
    """
    types_info = {
        "code_pattern": "Useful coding idiom or pattern discovered in the codebase",
        "architectural_decision": "Why a particular approach or design was chosen",
        "trick": "Clever workaround or unexpected solution to a problem",
        "plan": "Project direction, roadmap, or planned changes",
        "lesson": "Something that went wrong and should be avoided in the future",
        "concept": "General understanding or knowledge gained",
        "idea": "Novel concept or suggestion",
        "decision": "Significant choice made with its context and reasoning",
    }

    output = "Available Memory Types:\n\n"
    for type_name, description in types_info.items():
        output += f"• {type_name}: {description}\n"

    return output


@mcp.tool()
async def decayOrExpire(
    ttl_days: int = 30,
    min_usage_count: int = 0,
    dry_run: bool = True,
) -> str:
    """
    Clean up expired memories based on TTL and usage.

    💡 Tip: Call periodically to clean up temporary memories and enforce
    retention policies. Use dry_run=true first to see what would be deleted.

    Args:
        ttl_days: Delete memories not accessed in this many days (default: 30)
        min_usage_count: Delete memories with fewer accesses than this
        dry_run: If True, only report what would be deleted (default: True)

    Returns:
        Summary of cleanup operation
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=ttl_days)

        # Get all points
        all_points = scroll_all(COLLECTION_NAME)

        to_delete = []
        for point in all_points:
            payload = point.payload
            last_accessed = payload.get("last_accessed")
            usage_count = payload.get("usage_count", 0)
            retention = payload.get("retention_policy", "auto-delete")

            # Skip permanent memories
            if retention == "permanent":
                continue

            should_delete = False

            # Check TTL - expired if last accessed (or created) before cutoff
            ts = last_accessed or payload.get("created_at")
            if ts:
                last_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if last_time < cutoff:
                    should_delete = True

            # Check usage - too low if under threshold (only if TTL didn't already flag it)
            if not should_delete and usage_count < min_usage_count:
                should_delete = True

            if should_delete:
                to_delete.append(str(point.id))

        if dry_run:
            return f"Would delete {len(to_delete)} memories (ttl={ttl_days}d, min_usage={min_usage_count})\nIDs: {to_delete[:10]}{'...' if len(to_delete) > 10 else ''}"
        else:
            if to_delete:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=to_delete,
                )
            pg_deleted = pg_store.decay_memories(ttl_days, min_usage_count) if pg_store.is_available() else 0
            return f"Deleted {len(to_delete)} expired from Qdrant, {pg_deleted} from PostgreSQL"

    except Exception as e:
        logger.error(f"Decay/expire failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def attachProvenance(
    memory_id: str,
    source: str,
    model_version: str | None = None,
    confidence: float | None = None,
    notes: str | None = None,
) -> str:
    """
    Add provenance metadata to a memory.

    💡 Tip: When storing agent-generated content, attach provenance so future
    queries can trace the source and confidence of the memory.

    Args:
        memory_id: UUID of the memory
        source: Source of the memory (e.g., "git diff", "user input", "llm_generated")
        model_version: Version of embedding model used
        confidence: Confidence score (0-1)
        notes: Additional notes

    Returns:
        Success or error message
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        provenance = {
            "source": source,
            "model_version": model_version,
            "confidence": confidence,
            "timestamp": get_now_iso(),
            "notes": notes,
        }

        # Get current payload
        results = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
        )

        if not results:
            return f"Memory not found: {memory_id}"

        current_provenance = results[0].payload.get("provenance", {})
        if isinstance(current_provenance, dict):
            # Append to existing provenance list
            if "history" not in current_provenance:
                current_provenance["history"] = []
            current_provenance["history"].append(provenance)
        else:
            current_provenance = {"history": [provenance]}

        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"provenance": current_provenance},
            points=[memory_id],
        )

        return f"Added provenance to memory {memory_id}"

    except Exception as e:
        logger.error(f"Attach provenance failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def redactSensitive(
    text: str,
    mask_char: str = "█",
) -> str:
    """
    Detect and redact PII from text.

    💡 Tip: Before storing memories that might contain secrets, keys, or personal
    info, use this to check and redact sensitive content.

    Args:
        text: Text to check and redact
        mask_char: Character for masking (default: █)

    Returns:
        Redacted text with sensitivity level
    """
    redactor = get_redactor()
    redacted, matches = redactor.redact(text, mask_char)
    sensitivity = redactor.get_sensitivity_level(text)

    output = f"Sensitivity: {sensitivity}\n"
    output += f"PII matches found: {len(matches)}\n\n"
    output += f"Redacted text:\n{redacted}"

    return output


@mcp.tool()
async def getMemoryMetrics() -> str:
    """
    Get memory system metrics and statistics.

    💡 Tip: Check this to monitor memory system health, storage usage,
    and retrieval patterns.

    Returns:
        Formatted metrics report
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        # Get collection info
        info = qdrant_client.get_collection(COLLECTION_NAME)
        total = info.points_count

        # Scroll for payload stats
        all_points = scroll_all(COLLECTION_NAME)

        by_type = {}
        by_agent = {}
        total_usage = 0

        for point in all_points:
            payload = point.payload
            mtype = payload.get("memory_type", "unknown")
            agent = payload.get("agent_id", "unknown")
            usage = payload.get("usage_count", 0)

            by_type[mtype] = by_type.get(mtype, 0) + 1
            by_agent[agent] = by_agent.get(agent, 0) + 1
            total_usage += usage

        pg_stats = pg_store.get_metrics() if pg_store.is_available() else {}

        output = f"""Memory System Metrics
====================

Total Memories: {total}

By Type:
"""
        for mtype, count in sorted(by_type.items()):
            output += f"  {mtype}: {count}\n"

        output += f"\nBy Agent:\n"
        for agent, count in sorted(by_agent.items(), key=lambda x: -x[1])[:5]:
            output += f"  {agent}: {count}\n"

        output += f"\nTotal Retrieval Count: {total_usage}\n"
        output += f"Avg Usage per Memory: {total_usage / total if total > 0 else 0:.2f}\n"

        return output

    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def onAgentAction(
    action_type: str,
    context: str,
    path: str | None = None,
    tags: list[str] | None = None,
    agent_id: str | None = None,
) -> str:
    """
    Capture agent action context for memory storage.

    💡 Tip: Call this when you discover important patterns in code, make architectural
    decisions, or learn something worth remembering. Include the relevant context!

    Action types:
    - file_open: When you read an important file
    - file_edit: When you modify code
    - test_run: When tests execute with notable results
    - commit: When you commit changes (include diff context)
    - question: When you ask for help (include the problem context)
    - discovery: When you learn something interesting about the codebase

    Args:
        action_type: Type of action (file_open, file_edit, test_run, commit, question, discovery)
        context: Freeform context to remember
        path: File path if applicable
        tags: Additional tags to add
        agent_id: Agent identifier

    Returns:
        Memory ID of stored action context
    """
    # Map action types to memory types
    type_mapping = {
        "file_open": "concept",
        "file_edit": "code_pattern",
        "test_run": "lesson",
        "commit": "decision",
        "question": "concept",
        "discovery": "trick",
    }

    memory_type = type_mapping.get(action_type, "concept")

    # Build tags
    final_tags = [action_type, f"agent:{agent_id or 'unknown'}"]
    if tags:
        final_tags.extend(tags)

    return await upsertMemory(
        text=context,
        memory_type=memory_type,
        source=f"agent_action:{action_type}",
        tags=final_tags,
        path=path,
        agent_id=agent_id,
        retention_policy="auto-delete",
    )


@mcp.tool()
async def getMemorySystemPrompt() -> str:
    """
    Returns the memory system prompt for LLM injection.

    💡 Tip: Call this at session start to load memory usage guidelines into your context.
    This helps you understand when to store and retrieve memories effectively.

    Returns:
        System prompt for memory behavior
    """
    prompt_path = SCRIPT_DIR / "memory_system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        return """## Memory System Guidelines

You have access to a memory system for storing and retrieving important learnings.

### When to STORE:
- Code patterns: Useful idioms or solutions discovered
- Architectural decisions: Why a particular approach was chosen
- Coding tricks: Clever workarounds or unexpected solutions
- Plans: Project direction or roadmap information
- Lessons: Things that went wrong and should be avoided

### When to RETRIEVE:
- Before starting a new task, check relevant memories
- When stuck, search for similar problems others solved
- After errors, check if similar issues were previously encountered

### Quick Store:
upsertMemory(text="...", type="...", tags=[...])
"""


@mcp.tool()
async def reindexMemory(
    new_model: str = "BAAI/bge-m3",
    batch_size: int = 100,
) -> str:
    """
    Re-index all memories with a new embedding model.

    💡 Tip: Use when switching embedding models to re-encode all memories
    with the new model. This ensures consistent similarity calculations.

    Args:
        new_model: HuggingFace model name for new embeddings
        batch_size: Number of memories to process at a time

    Returns:
        Summary of reindexing operation
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading new model: {new_model}")
        model = SentenceTransformer(new_model)

        # Get all points
        all_points = scroll_all(COLLECTION_NAME)

        total = len(all_points)
        logger.info(f"Reindexing {total} memories with {new_model}")

        # Process in batches
        for i in range(0, total, batch_size):
            batch = all_points[i:i + batch_size]
            points_to_update = []

            for point in batch:
                text = point.payload.get("text", "")
                if text:
                    new_embedding = model.encode([text])[0].tolist()
                    # Merge metadata into existing payload to avoid data loss
                    updated_payload = dict(point.payload)
                    updated_payload["embedding_model"] = new_model
                    updated_payload["reindexed_at"] = get_now_iso()
                    points_to_update.append({
                        "id": str(point.id),
                        "vector": new_embedding,
                        "payload": updated_payload,
                    })

            if points_to_update:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_update,
                )

            logger.info(f"Reindexed batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        return f"Reindexed {total} memories with model {new_model}"

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def auditTrail(
    memory_id: str,
    limit: int = 10,
) -> str:
    """
    Get audit trail for a memory (retrieval history).

    💡 Tip: Check this when investigating why a memory was retrieved or
    to debug memory behavior and access patterns.

    Args:
        memory_id: UUID of the memory
        limit: Maximum number of history entries

    Returns:
        Audit trail information
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        results = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
        )

        if not results:
            return f"Memory not found: {memory_id}"

        payload = results[0].payload

        output = f"""Audit Trail for Memory: {memory_id}
========================================

Created: {payload.get('created_at', 'unknown')}
Last Accessed: {payload.get('last_accessed', 'never')}
Usage Count: {payload.get('usage_count', 0)}
Retention Policy: {payload.get('retention_policy', 'unknown')}
Sensitivity: {payload.get('sensitivity', 'unknown')}
"""

        provenance = payload.get("provenance", {})
        if provenance and isinstance(provenance, dict):
            history = provenance.get("history", [])
            if history:
                output += f"\nProvenance History ({len(history)} entries):\n"
                for i, entry in enumerate(history[:limit], 1):
                    output += f"\n{i}. {entry.get('timestamp', 'unknown')}\n"
                    output += f"   Source: {entry.get('source', 'unknown')}\n"
                    if entry.get('model_version'):
                        output += f"   Model: {entry.get('model_version')}\n"
                    if entry.get('confidence'):
                        output += f"   Confidence: {entry.get('confidence')}\n"
                    if entry.get('notes'):
                        output += f"   Notes: {entry.get('notes')}\n"

        return output

    except Exception as e:
        logger.error(f"Audit trail failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def mergeDuplicates(
    threshold: float = 0.95,
    dry_run: bool = True,
) -> str:
    """
    Find and merge duplicate memories based on similarity threshold.

    💡 Tip: Use when memory count grows large and you want to consolidate
    redundant entries. Set threshold to control how similar memories must be
    to be merged (0-1, higher = stricter matching).

    Args:
        threshold: Similarity threshold for duplicates (0-1, default: 0.95)
        dry_run: If True, only report what would be merged

    Returns:
        Summary of merge operation
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        from collections import defaultdict

        # Get all memories
        all_points = scroll_all(COLLECTION_NAME)

        # Group by type and compute similarity
        groups = defaultdict(list)
        for point in all_points:
            mtype = point.payload.get("memory_type", "unknown")
            groups[mtype].append(point)

        merged_count = 0
        to_delete = []
        already_marked = set()

        for mtype, points in groups.items():
            for i, p1 in enumerate(points):
                if str(p1.id) in already_marked:
                    continue
                for p2 in points[i+1:]:
                    if str(p2.id) in already_marked:
                        continue
                    # Compare text similarity (simple word overlap for now)
                    text1 = set(p1.payload.get("text", "").lower().split())
                    text2 = set(p2.payload.get("text", "").lower().split())

                    if not text1 or not text2:
                        continue

                    overlap = len(text1 & text2) / max(len(text1 | text2), 1)

                    if overlap >= threshold:
                        # Keep the one with more usage
                        usage1 = p1.payload.get("usage_count", 0)
                        usage2 = p2.payload.get("usage_count", 0)

                        loser = str(p1.id if usage1 < usage2 else p2.id)
                        to_delete.append(loser)
                        already_marked.add(loser)
                        merged_count += 1

        if dry_run:
            return f"Would merge {merged_count} duplicate pairs\nWould delete {len(to_delete)} memories"
        else:
            if to_delete:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=list(set(to_delete)),
                )
            return f"Merged {merged_count} duplicate pairs, deleted {len(to_delete)} memories"

    except Exception as e:
        logger.error(f"Merge duplicates failed: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# FEF V3 Extensions (optional)
# ============================================================================

fef_setup_done = False

try:
    from launcher.tool_extensions import Extension, ExtensionType
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    FEF_V3_AVAILABLE = True
except ImportError:
    FEF_V3_AVAILABLE = False


def setup_fef_v3(registry=None):
    """Set up FEF V3 extensions for memorymcp."""
    global fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        logger.warning("FEF V3 not available, skipping extension setup")
        fef_setup_done = True
        return None, None, None

    def get_memory_stats(params: dict) -> dict:
        """Data source: Get memory system statistics."""
        try:
            info = qdrant_client.get_collection(COLLECTION_NAME)
            total = info.points_count
            return {
                "collection": COLLECTION_NAME,
                "total_memories": total,
                "status": "operational"
            }
        except Exception as e:
            return {
                "collection": COLLECTION_NAME,
                "total_memories": 0,
                "status": f"error: {e}"
            }

    custom_extensions = [
        Extension(
            name="memory_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object"}
            },
            handler=get_memory_stats,
            metadata={"description": "Memory system statistics", "category": "metrics"}
        ),
        Extension(
            name="list_memory_types",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {"type": "object"}
            },
            handler=lambda params: {"types": [t.value for t in MemoryType]},
            metadata={"description": "List available memory types", "category": "info"}
        ),
    ]

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", MGMT_PORT))

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
    return fef_manager, fef_registry, fef_http_server


# Call setup_fef_v3 early so extensions are registered
# (but handle the case where registry isn't available yet)
try:
    setup_fef_v3()
except Exception as e:
    logger.debug(f"Early FEF setup deferred: {e}")


# ============================================================================
# Streamable HTTP App (for launcher)
# ============================================================================

app = mcp.streamable_http_app()


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {TOOL_NAME} FastMCP server")
    logger.info(f"  MCP port: {MCP_PORT}")
    logger.info(f"  SSE endpoint: http://0.0.0.0:{MCP_PORT}/sse")
    logger.info(f"  Streamable HTTP: http://0.0.0.0:{MCP_PORT}/mcp")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://0.0.0.0:{MGMT_PORT}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )


# ============================================================================
# Graph Tools — memory edges, concept chains, export
# ============================================================================

@mcp.tool()
async def createMemoryEdge(
    from_id: str,
    to_id: str,
    relation: str = "related_to",
    label: str | None = None,
) -> str:
    """
    Create a directed edge (link) between two memories.

    💡 Tip: Use to build knowledge graphs — connect related memories, show
    dependencies, or chain steps in a process. Edges are stored in each
    memory's metadata under the "edges" key.

    Common relations: related_to, depends_on, follows, contradicts, refines, example_of

    Args:
        from_id: Source memory UUID
        to_id: Target memory UUID
        relation: Edge type (default: "related_to")
        label: Optional human-readable label for the edge

    Returns:
        Confirmation message
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        # Verify both memories exist
        results = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[from_id, to_id],
            with_payload=True,
        )
        if len(results) < 2:
            found = {str(r.id) for r in results}
            missing = [x for x in [from_id, to_id] if x not in found]
            return f"Error: Memory not found: {missing[0]}"

        edge = {"to": to_id, "relation": relation, "label": label}

        # Append edge to source memory's edges list
        src = next(r for r in results if str(r.id) == from_id)
        edges = src.payload.get("edges", [])
        edges.append(edge)

        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"edges": edges},
            points=[from_id],
        )

        # Also store reverse reference
        rev_edge = {"to": from_id, "relation": f"back:{relation}", "label": label}
        dst = next(r for r in results if str(r.id) == to_id)
        dst_edges = dst.payload.get("edges", [])
        dst_edges.append(rev_edge)

        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"edges": dst_edges},
            points=[to_id],
        )

        return f"Created edge: {from_id[:8]} --[{relation}]--> {to_id[:8]}"

    except Exception as e:
        logger.error(f"createMemoryEdge failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def getMemoryGraph(
    memory_id: str,
    depth: int = 2,
    format: str = "mermaid",
) -> str:
    """
    Get the graph of memories connected to a starting memory, expanding out to N hops.

    💡 Tip: Use to explore the knowledge neighborhood around a concept.
    Returns a graph you can render in any Mermaid-compatible viewer.

    Args:
        memory_id: Starting memory UUID
        depth: How many hops to expand (default: 2, max: 4)
        format: "mermaid" for diagram, "ascii" for text list

    Returns:
        Graph visualization of connected memories
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        depth = min(depth, 4)
        visited = set()
        nodes = {}
        edges = []
        queue = [(memory_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            results = qdrant_client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[current_id],
                with_payload=True,
            )
            if not results:
                continue

            payload = results[0].payload
            text = payload.get("text", "")[:50].replace('"', "'")
            mtype = payload.get("memory_type", "unknown")
            nodes[current_id] = {"text": text, "type": mtype}

            for edge in payload.get("edges", []):
                to_id = edge.get("to")
                relation = edge.get("relation", "related_to")
                label = edge.get("label", "")
                edges.append((current_id, to_id, relation, label))
                if to_id not in visited:
                    queue.append((to_id, current_depth + 1))

        if not nodes:
            return "No connected memories found."

        if format == "mermaid":
            lines = ["graph TD"]
            for nid, info in nodes.items():
                safe = nid[:8]
                lines.append(f'    {safe}["{safe} [{info['type']}]\\n{info['text']}"]')
            for src, dst, rel, lbl in edges:
                s, d = src[:8], dst[:8]
                edge_label = lbl or rel
                lines.append(f'    {s} -->|"{edge_label}"| {d}')
            return "\n".join(lines)
        else:
            lines = [f"Memory Graph (depth={depth}, {len(nodes)} nodes, {len(edges)} edges)", ""]
            for nid, info in nodes.items():
                lines.append(f"  [{info['type']}] {nid[:8]}: {info['text']}")
            lines.append("")
            for src, dst, rel, lbl in edges:
                lines.append(f"  {src[:8]} --[{rel}]--> {dst[:8]}")
            return "\n".join(lines)

    except Exception as e:
        logger.error(f"getMemoryGraph failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def exportGraphAsMarkdown(
    root_id: str | None = None,
    memory_type: str | None = None,
    tag: str | None = None,
) -> str:
    """
    Export memories and their edges as a Markdown document with embedded Mermaid diagrams.

    💡 Tip: Use to generate a readable document from your knowledge graph.
    Great for LLM context injection, documentation, or sharing knowledge.
    If no root_id, exports all memories (optionally filtered by type or tag).

    Args:
        root_id: Starting memory UUID (optional, exports all if omitted)
        memory_type: Filter to only this memory type (optional)
        tag: Filter to only memories with this tag (optional)

    Returns:
        Markdown document with memory content and Mermaid graph
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        all_points = scroll_all(COLLECTION_NAME)

        # Filter
        points = []
        for point in all_points:
            p = point.payload
            if memory_type and p.get("memory_type") != memory_type:
                continue
            if tag and tag not in p.get("tags", []):
                continue
            points.append(point)

        if root_id:
            # Only include memories reachable from root_id
            visited = set()
            queue = [root_id]
            id_set = set()
            while queue:
                cid = queue.pop(0)
                if cid in visited:
                    continue
                visited.add(cid)
                pt = next((p for p in points if str(p.id) == cid), None)
                if not pt:
                    continue
                id_set.add(cid)
                for edge in pt.payload.get("edges", []):
                    queue.append(edge.get("to"))
            points = [p for p in points if str(p.id) in id_set]

        # Build markdown
        lines = ["# Memory Graph Export", ""]
        lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
        lines.append(f"*Memories: {len(points)}*")
        lines.append("")

        # Memory list
        lines.append("## Memories")
        lines.append("")
        for point in points:
            p = point.payload
            mid = str(point.id)[:8]
            mtype = p.get("memory_type", "unknown")
            text = p.get("text", "")
            tags = p.get("tags", [])
            lines.append(f"### [{mtype}] {mid}...")
            lines.append(f"**Tags**: {', '.join(tags) if tags else 'none'}")
            lines.append(f"**Source**: {p.get('source', 'unknown')}")
            lines.append(f"**Usage**: {p.get('usage_count', 0)}x")
            lines.append(f"**Created**: {p.get('created_at', 'unknown')}")
            lines.append("")
            lines.append(text)
            lines.append("")

            # Show edges
            edges = p.get("edges", [])
            forward = [e for e in edges if not e.get("relation", "").startswith("back:")]
            if forward:
                lines.append("**Links**:")
                for e in forward:
                    to_id = e.get("to", "")[:8]
                    rel = e.get("relation", "related_to")
                    lbl = e.get("label", "")
                    lines.append(f"- → {to_id}... [{rel}]" + (f" — {lbl}" if lbl else ""))
                lines.append("")

        # Mermaid diagram
        lines.append("## Graph Diagram")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph TD")
        for point in points:
            mid = str(point.id)[:8]
            mtype = point.payload.get("memory_type", "unknown")
            preview = point.payload.get("text", "")[:30].replace('"', "'")
            lines.append(f'    {mid}["{mid} [{mtype}]\\n{preview}"]')
        for point in points:
            mid = str(point.id)
            for edge in point.payload.get("edges", []):
                if edge.get("relation", "").startswith("back:"):
                    continue
                to_id = edge.get("to", "")
                if any(str(p.id) == to_id for p in points):
                    rel = edge.get("relation", "related_to")
                    s, d = mid[:8], to_id[:8]
                    lines.append(f'    {s} -->|"{rel}"| {d}')
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"exportGraphAsMarkdown failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def memoryTypeChart(
    format: str = "ascii",
) -> str:
    """
    Show distribution of memories by type as a bar chart.

    💡 Tip: Use to see what kinds of knowledge you store most and spot gaps.

    Args:
        format: "ascii" for terminal bar chart, "mermaid" for pie chart

    Returns:
        Type distribution chart
    """
    if not qdrant_client:
        return "Error: Qdrant client not initialized"

    try:
        all_points = scroll_all(COLLECTION_NAME)
        by_type: dict[str, int] = {}
        for point in all_points:
            mtype = point.payload.get("memory_type", "unknown")
            by_type[mtype] = by_type.get(mtype, 0) + 1

        if not by_type:
            return "No memories found."

        total = sum(by_type.values())

        if format == "mermaid":
            lines = ["pie title Memory Types"]
            for mtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                lines.append(f'    "{mtype}" : {count}')
            return "\n".join(lines)

        max_count = max(by_type.values())
        max_bar = 30
        lines = ["Memory Types", "=" * 50, ""]
        for mtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            bar_len = int((count / max(max_count, 1)) * max_bar)
            bar = "█" * bar_len
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {mtype:<25} {bar} {count} ({pct:.0f}%)")
        lines.append(f"\n  Total: {total} memories")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"memoryTypeChart failed: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def textToGraph(
    text: str,
    title: str | None = None,
    output: str = "text",
) -> str:
    """
    Convert structured text (Markdown, skill files, docs) into a knowledge graph.

    💡 Tip: LLMs reason better over graphs than flat text. Feed this output back
    into an LLM prompt to improve comprehension of complex documents, skills, or
    procedures. Parses headings, lists, numbered steps, prose, code blocks, and
    cross-references into nodes and edges with full content preserved.

    Best for: SKILL.md files, README sections, procedure docs, architecture notes,
    any text with hierarchical or sequential structure.

    Args:
        text: The text content to convert (Markdown, plain text, etc.)
        title: Optional title for the graph root node
        output: Output format:
            - "adjacency" (default, best for LLM): plain-text adjacency list with inline content
            - "text": structured natural language descriptions
            - "dot": compact Graphviz DOT format
            - "json": full structured data with content fields
            - "mermaid": diagram for human visualisation
            - "both": mermaid + json combined

    Returns:
        Knowledge graph in the requested format. For LLM consumption, use "adjacency" (most token-efficient)
        or "text" (best for reasoning tasks).
    """
    try:

        # Parse YAML frontmatter
        lines_in = text.split("\n")
        frontmatter = {}
        if lines_in and lines_in[0].strip() == "---":
            frontmatter_lines = []
            for i, line in enumerate(lines_in[1:], 1):
                if line.strip() == "---":
                    lines_in = lines_in[i + 1:]
                    break
                frontmatter_lines.append(line)
            # Parse simple key: value pairs
            for line in frontmatter_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip()

        title = title or "Document"

        # ── Parse structure ──────────────────────────────────
        nodes = []  # {id, label, content, level, type, line_number}
        edges = []  # {from, to, relation}
        node_counter = 0

        def make_node(label: str, content: str, level: int, ntype: str, line_number: int = 0) -> str:
            nonlocal node_counter
            node_counter += 1
            nid = f"n{node_counter}"
            safe_label = label.replace('"', "'").strip()[:200]
            nodes.append({
                "id": nid,
                "label": safe_label,
                "content": content,
                "level": level,
                "type": ntype,
                "line_number": line_number
            })
            return nid

        # Root node
        root_id = make_node(title, title, 0, "root", 0)

        # Track hierarchy: stack of (node_id, heading_level)
        stack = [(root_id, 0)]
        prev_step_id = None
        in_code_block = False
        code_lang = ""
        code_lines = []
        code_start_line = 0
        prose_buffer = []

        def flush_prose(parent_level):
            nonlocal prose_buffer
            filtered = [l for l in prose_buffer if l.strip() not in ("---", "***", "___")]
            prose_buffer = []
            content = "\n".join(filtered).strip()
            if not content:
                return
            parent_id = stack[-1][0]
            nid = make_node(
                content[:100],
                content,
                parent_level + 1,
                "paragraph",
                0
            )
            edges.append({"from": parent_id, "to": nid, "relation": "has_content"})

        for line_number, raw_line in enumerate(lines_in, 1):
            line = raw_line.rstrip()

            # Handle code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of code block
                    code_content = "\n".join(code_lines)
                    parent_id = stack[-1][0]
                    nid = make_node(
                        f"{code_lang} code" if code_lang else "Code block",
                        code_content,
                        stack[-1][1] + 1,
                        "code",
                        code_start_line
                    )
                    edges.append({"from": parent_id, "to": nid, "relation": "has_code"})
                    code_lines = []
                else:
                    # Start of code block
                    code_lang = line.strip()[3:].strip()
                    code_start_line = line_number
                in_code_block = not in_code_block
                continue

            if in_code_block:
                code_lines.append(raw_line)
                continue

            stripped = line.strip()
            if not stripped:
                flush_prose(stack[-1][1])
                continue

            # Check if line matches any pattern
            is_pattern = any([
                stripped.startswith("#"),
                re.match(r'^(\d+)[.)]\s+(.+)', stripped),
                re.match(r'^[-*]\s+(.+)', stripped),
                re.match(r'^\*\*(.+?)\*\*:\s*(.+)', stripped),
                re.match(r'^\*\*(.+?)\*\*\s*$', stripped),
            ])

            if is_pattern:
                # Flush prose buffer before processing pattern
                flush_prose(stack[-1][1])

            # ── Headings ────────────────────────────────────
            if stripped.startswith("#"):
                level = 0
                for ch in stripped:
                    if ch == "#":
                        level += 1
                    else:
                        break
                heading_text = stripped.lstrip("#").strip()

                heading_lower = heading_text.lower().strip()
                all_refs = re.findall(r'\b(pctech\d+|pcgene\d+|commontech\d+|pkgtech\d+|pctmeta\d+)\b', heading_text)
                # Filter out self-references (heading that IS the rule, e.g. "### pctech31" or "### commontech6 (desc)")
                internal_refs = [r for r in all_refs if not re.match(rf'^{re.escape(r)}(\s|$|\()', heading_lower)]

                # Pop stack until we find parent
                while len(stack) > 1 and stack[-1][1] >= level:
                    stack.pop()

                parent_id = stack[-1][0]
                nid = make_node(heading_text, heading_text, level, "section", line_number)
                edges.append({"from": parent_id, "to": nid, "relation": "has_section"})
                stack.append((nid, level))
                prev_step_id = None

                for ref in internal_refs:
                    ref_id = make_node(f"→ {ref}", f"Reference to rule {ref}", level + 1, "xref", line_number)
                    edges.append({"from": nid, "to": ref_id, "relation": "xref"})

                continue

            # ── Numbered steps (1. 2. etc) ─────────────────
            num_match = re.match(r'^(\d+)[.)]\s+(.+)', stripped)
            if num_match:
                step_num = num_match.group(1)
                step_text = num_match.group(2)
                parent_id = stack[-1][0]
                nid = make_node(f"Step {step_num}: {step_text}", f"Step {step_num}: {step_text}", stack[-1][1] + 1, "step", line_number)
                edges.append({"from": parent_id, "to": nid, "relation": "has_step"})

                # Chain sequential steps
                if prev_step_id:
                    edges.append({"from": prev_step_id, "to": nid, "relation": "then"})
                prev_step_id = nid
                continue

            # ── Bullet points (- or *) ──────────────────────
            bullet_match = re.match(r'^[-*]\s+(.+)', stripped)
            if bullet_match:
                bullet_text = bullet_match.group(1)
                parent_id = stack[-1][0]
                nid = make_node(bullet_text, bullet_text, stack[-1][1] + 1, "item", line_number)
                edges.append({"from": parent_id, "to": nid, "relation": "has_item"})
                prev_step_id = None
                continue

            # ── Cross-references [text](link) ──────────────
            refs = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', stripped)
            if refs:
                parent_id = stack[-1][0]
                for ref_label, ref_target in refs:
                    nid = make_node(ref_label, f"Link: {ref_label} ({ref_target})", stack[-1][1] + 1, "reference", line_number)
                    edges.append({"from": parent_id, "to": nid, "relation": "references"})

            # ── Key: Value patterns ─────────────────────────
            kv_match = re.match(r'^\*\*(.+?)\*\*:\s*(.+)', stripped)
            if kv_match:
                key = kv_match.group(1)
                value = kv_match.group(2)
                parent_id = stack[-1][0]
                nid = make_node(f"{key}: {value}", f"{key}: {value}", stack[-1][1] + 1, "property", line_number)
                edges.append({"from": parent_id, "to": nid, "relation": "has_property"})
                continue

            # ── Bold standalone lines (sub-sections) ────────
            bold_match = re.match(r'^\*\*(.+?)\*\*\s*$', stripped)
            if bold_match:
                bold_text = bold_match.group(1)
                parent_id = stack[-1][0]
                nid = make_node(bold_text, bold_text, stack[-1][1] + 1, "subsection", line_number)
                edges.append({"from": parent_id, "to": nid, "relation": "has_subsection"})
                prev_step_id = None
                continue

            # If not a pattern, accumulate as prose
            prose_buffer.append(raw_line)

        # Flush any remaining prose
        flush_prose(stack[-1][1])

        # Add frontmatter nodes
        if frontmatter:
            meta_id = make_node("Frontmatter", "Skill metadata", 1, "metadata", 0)
            edges.append({"from": root_id, "to": meta_id, "relation": "has_metadata"})
            for key, value in frontmatter.items():
                prop_id = make_node(f"{key}", f"{key}: {value}", 2, "property", 0)
                edges.append({"from": meta_id, "to": prop_id, "relation": "has_property"})

        if not nodes:
            return "No structure found in text."

        # Build lookup maps
        node_map = {n["id"]: n for n in nodes}

        # Build adjacency: node_id -> [(child_id, relation)]
        children = {}
        for e in edges:
            children.setdefault(e["from"], []).append((e["to"], e["relation"]))

        # ── JSON output ────────────────────────────────────────
        if output == "json":
            return json.dumps({"nodes": nodes, "edges": edges}, indent=2)

        # ── Adjacency list (best for LLM) ──────────────────────
        if output == "adjacency":
            lines_out = []
            for n in nodes:
                nid = n["id"]
                ntype = n["type"]
                content = n["content"]
                kids = children.get(nid, [])
                if not kids:
                    lines_out.append(f"{nid} [{ntype}]: {content}")
                else:
                    targets = ", ".join(
                        f"{cid}({rel})" for cid, rel in kids
                    )
                    lines_out.append(f"{nid} [{ntype}] -> {targets}")
                    lines_out.append(f"  content: {content}")
            return "\n".join(lines_out)

        # ── Structured natural language (best for reasoning) ───
        if output == "text":
            lines_out = []
            for n in nodes:
                nid = n["id"]
                ntype = n["type"]
                content = n["content"]
                level = n["level"]
                kids = children.get(nid, [])

                if ntype == "root":
                    lines_out.append(f"# {content}")
                elif ntype == "section":
                    lines_out.append(f"{'##' * min(level, 4)} {content}")
                elif ntype == "subsection":
                    lines_out.append(f"  Sub-topic: {content}")
                elif ntype == "item":
                    lines_out.append(f"  - {content}")
                elif ntype == "step":
                    lines_out.append(f"  {content}")
                elif ntype == "code":
                    lang_label = n["label"]
                    lines_out.append(f"  [{lang_label}]")
                    for code_line in content.split("\n")[:8]:
                        lines_out.append(f"    {code_line}")
                    if content.count("\n") > 8:
                        lines_out.append(f"    ... ({content.count(chr(10)) - 8} more lines)")
                elif ntype == "paragraph":
                    lines_out.append(f"  {content}")
                elif ntype == "property":
                    lines_out.append(f"  Property: {content}")
                elif ntype == "xref":
                    lines_out.append(f"  References: {content}")
                elif ntype == "reference":
                    lines_out.append(f"  Link: {content}")
                elif ntype == "metadata":
                    lines_out.append(f"Metadata: {content}")
            return "\n".join(lines_out)

        # ── DOT/Graphviz output ────────────────────────────────
        if output == "dot":
            dot_lines = ["digraph {"]
            for n in nodes:
                nid = n["id"]
                label = n["label"].replace('"', "'").replace("\n", " ")
                ntype = n["type"]
                if ntype == "root":
                    dot_lines.append(f'  {nid} [shape=doublecircle label="{label}"]')
                elif ntype == "code":
                    preview = label[:30]
                    dot_lines.append(f'  {nid} [shape=box style=filled label="{preview}"]')
                elif ntype in ("section", "subsection"):
                    dot_lines.append(f'  {nid} [shape=box label="{label}"]')
                elif ntype == "xref":
                    dot_lines.append(f'  {nid} [shape=diamond label="{label}"]')
                else:
                    dot_lines.append(f'  {nid} [label="{label}"]')
            for e in edges:
                rel = e["relation"]
                if rel == "has_section":
                    dot_lines.append(f'  {e["from"]} -> {e["to"]}')
                elif rel == "has_code":
                    dot_lines.append(f'  {e["from"]} -> {e["to"]} [style=bold]')
                elif rel == "then":
                    dot_lines.append(f'  {e["from"]} -> {e["to"]} [style=dashed label="then"]')
                elif rel == "xref":
                    dot_lines.append(f'  {e["from"]} -> {e["to"]} [style=dotted]')
                else:
                    dot_lines.append(f'  {e["from"]} -> {e["to"]} [style=dotted]')
            dot_lines.append("}")
            return "\n".join(dot_lines)

        # ── Mermaid output ─────────────────────────────────────
        mermaid_lines = ["graph TD"]
        for n in nodes:
            nid = n["id"]
            label = n["label"]
            ntype = n["type"]
            shape_map = {
                "root": f'{nid}{{"{label}"}}',
                "section": f'{nid}["{label}"]',
                "subsection": f'{nid}("{label}")',
                "step": f'{nid}["{label}"]',
                "item": f'{nid}["{label}"]',
                "property": f'{nid}[["{label}"]]',
                "reference": f'{nid}{{"{label}"}}',
                "metadata": f'{nid}{{"{label}"}}',
                "code": f'{nid}["{label}"]',
                "paragraph": f'{nid}["{label}"]',
                "xref": f'{nid}{{"{label}"}}',
            }
            mermaid_lines.append(f"    {shape_map.get(ntype, f'{nid}["{label}"]')}")

        mermaid_lines.append("")
        edge_styles = {
            "has_section": "-->",
            "has_subsection": "-.->",
            "has_step": "==>",
            "then": "-->",
            "has_item": "-.->",
            "has_property": "-.->",
            "references": "-.->",
            "has_metadata": "-.->",
            "has_code": "==>",
            "has_content": "-.->",
            "xref": "-.->",
        }
        for e in edges:
            style = edge_styles.get(e["relation"], "-->")
            mermaid_lines.append(f"    {e['from']} {style}|{e['relation']}| {e['to']}")

        result = "\n".join(mermaid_lines)

        if output == "both":
            result += "\n\n```json\n" + json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n```"

        return result

    except Exception as e:
        logger.error(f"textToGraph failed: {e}")
        return f"Error: {str(e)}"


_CONTENT_MARKERS = [
    r'^CLUSTERS\s*:',
    r'^COMPRESSED_RULES\s*:',
    r'^CODE\s*:',
    r'^#{2,}\s',
    r'^\*\*[^*]',
]

_PREAMBLE_PHRASES = [
    'i need to', 'let me', "i'll", 'i will', 'the user wants',
    'i want to', 'my approach', 'first, i', 'next, i', 'then i',
    'now i can', 'i can compress', 'here is how', 'to do this',
    'the goal', 'i should',
]


def strip_llm_artifacts(text: str) -> str:
    if not text or not text.strip():
        return text
    text = re.sub(r'<\?[\s\S]*?\?>', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*?</think\s*>', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*?</think\b', '', text)
    text = re.sub(r'<\?[\s\S]*$', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*$', '', text)
    text = re.sub(r'<think\b[\s\S]*$', '', text)
    text = text.strip()
    if not text:
        return text
    lines = text.split('\n')
    marker_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _CONTENT_MARKERS:
            if re.match(pattern, stripped, re.IGNORECASE):
                marker_positions.append(i)
                break
    if not marker_positions:
        return text
    for pos in marker_positions:
        next_line = ''
        for j in range(pos + 1, min(pos + 5, len(lines))):
            if lines[j].strip():
                next_line = lines[j].strip()
                break
        if not next_line:
            continue
        next_lower = next_line.lower()
        has_preamble = any(phrase in next_lower for phrase in _PREAMBLE_PHRASES)
        marker_word = lines[pos].strip().rstrip(':').strip().upper()
        if marker_word == 'CLUSTERS' and re.match(r'^\d+[\.\)]\s', next_line):
            continue
        if not has_preamble:
            if pos > 0:
                return '\n'.join(lines[pos:]).strip()
            return text
    last = marker_positions[-1]
    if last > 0:
        return '\n'.join(lines[last:]).strip()
    return text


SMARTGRAPH_SYSTEM_PROMPT = """You are compressing a technical skill/rules document for LLM context injection.

COMPRESSION STRATEGY:
- Compress ONLY explanatory prose, descriptions, and narrative text
- Treat as SACRED (never paraphrase, always verbatim): function names, macro names, constant names, parameter signatures, type constraints, negative constraints ("NEVER", "do NOT"), step sequencing, code examples
- ALSO SACRED: string literals used as enum values, dict keys/values, or default parameter values (e.g. "auto-delete", "code_pattern", "file_open") — treat exactly like function names, NEVER paraphrase or normalize them (not "auto_delete" or "code-pattern")
- Information gaps cause hallucination — the consumer LLM fills missing specifics with plausible guesses
- Your job is to eliminate prose WITHOUT creating gaps the consumer cannot safely fill

OUTPUT FORMAT — follow exactly, no other text:

VERBATIM_INDEX:
(Extract ALL technical identifiers from the source. List each one individually — never use wildcards like "VC_* macros" or "fc_* functions". Group by type.)
  macros: exact_name1, exact_name2, exact_name3
  functions: exact_name1, exact_name2
  constants: EXACT_CONST1, EXACT_CONST2
  types: exact_type1, exact_type2
  negative_constraints: "NEVER do X", "do NOT use Y"

CLUSTERS:
CLUSTER_NAME: rule1, rule2, rule3 — one-line summary

COMPRESSED_RULES:
(Compress prose explanations aggressively. Preserve ALL technical content verbatim — identifiers, parameter names, type constraints, negative rules, step order. If the source says "use tabs", write "use tabs" not "indent properly".)
ruleID: compressed-but-technically-exact content
ruleID1+ruleID2: merged content
[requires: r1, r2] ruleID: content

CODE:
(Preserve representative code patterns. Use ... only for boilerplate that isn't rule-specific.)
code snippet

Start with VERBATIM_INDEX: immediately. No preamble, no planning, no explanation."""

SMARTGRAPH_USER_TEMPLATE = """Compress this knowledge graph for LLM injection.

CRITICAL RULES:
1. Every function name, macro name, constant, parameter, and type from the source MUST appear in VERBATIM_INDEX
2. Every string literal used as an enum value, dict key/value, or default parameter MUST appear in VERBATIM_INDEX — preserve exact spelling, hyphens, underscores (e.g. "auto-delete" not "auto_delete")
3. Negative constraints ("NEVER", "do NOT", "No X") MUST appear verbatim in both VERBATIM_INDEX.negative_constraints AND the relevant COMPRESSED_RULES entry
4. Step sequencing and ordering MUST be preserved exactly
5. Compress ONLY the prose explanations — never compress technical content

Graph to compress:

{text}

Output:"""

_UNIVERSAL_ID_RE = re.compile(
    r'\b([A-Z][A-Z0-9_]{2,})\b'
    r'|\b([a-z][a-z0-9_]*_[a-z0-9_]+)\s*\('
    r'|\b([A-Z][A-Za-z0-9]+[a-z][a-z0-9]*)\s*\('
)
_QUOTED_STRING_RE = re.compile(
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
)
_QUOTED_MAPPING_RE = re.compile(
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
    r'\s*[=:]\s*'
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
)
_NOISE_NAMES = frozenset({
    'NULL', 'EXEC', 'SQL', 'BEGIN', 'END', 'DECLARE', 'SECTION', 'INTO',
    'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'SELECT', 'INSERT', 'UPDATE',
    'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'SET',
    'VALUES', 'INT', 'LONG', 'SHORT', 'CHAR', 'VOID', 'RETURN', 'IF',
    'ELSE', 'FOR', 'WHILE', 'DO', 'SWITCH', 'CASE', 'BREAK', 'CONTINUE',
    'DEFAULT', 'STRUCT', 'TYPEDEF', 'DEFINE', 'INCLUDE', 'PRINTF', 'SPRINTF',
    'MALLOC', 'FREE', 'SIZEOF', 'ATOL', 'ATOI', 'STRLEN', 'STRCPY', 'STRCAT',
    'MEMSET', 'MEMCPY', 'STDIN', 'STDOUT', 'STDERR', 'EOF', 'EXIT',
    'TRUE', 'FALSE', 'VARCHAR', 'STRING',
    'CONCEPT', 'LESSON', 'TRICK', 'PATTERN', 'IDEA', 'PLAN',
    'PUBLIC', 'STATIC', 'FINAL', 'ABSTRACT', 'PRIVATE', 'PROTECTED',
    'CLASS', 'INTERFACE', 'EXTENDS', 'IMPLEMENTS', 'PACKAGE', 'IMPORT',
    'FUNCTION', 'VAR', 'LET', 'CONST', 'TYPE', 'ASYNC', 'AWAIT',
    'MODULE', 'EXPORT', 'REQUIRE',
    'THE', 'THIS', 'THAT', 'WITH', 'USING', 'ARE', 'HAS', 'HAVE',
    'WAS', 'WERE', 'BEEN', 'BEING', 'WILL', 'WOULD', 'SHOULD', 'COULD',
    'MUST', 'SHALL', 'MAY', 'CAN', 'NEED', 'ALSO', 'BUT', 'HOWEVER',
    'NOTE', 'NOTES', 'SEE', 'USE', 'USED', 'WHEN', 'THEN', 'THAN',
    'SUCH', 'EACH', 'EVERY', 'ALL', 'ANY', 'SOME', 'MORE', 'MOST',
    'OTHER', 'ONLY', 'JUST', 'ALWAYS', 'NEVER', 'STILL', 'ALREADY',
    'BEFORE', 'AFTER', 'FIRST', 'LAST', 'NEXT',
})
_NOISE_STRINGS = frozenset({
    'the', 'and', 'for', 'not', 'yes', 'all', 'any', 'use', 'get', 'set',
    'add', 'new', 'old', 'put', 'key', 'val', 'one', 'two', 'out', 'off',
    'run', 'end', 'log', 'err', 'ok', 'true', 'false', 'null', 'none',
    'self', 'this', 'that', 'with', 'from', 'into', 'over', 'under',
    'items', 'data', 'text', 'name', 'type', 'list', 'dict', 'file',
    'args', 'kwargs', 'params', 'config', 'result', 'value', 'output',
    'input', 'index', 'count', 'size', 'length', 'width', 'height',
})


def _classify_name(name: str) -> str:
    if name.isupper() and '_' in name:
        return 'upper_case'
    if name.isupper():
        return 'upper_case'
    if '_' in name and name[0].islower():
        return 'lower_case'
    if name[0].isupper() and any(c.islower() for c in name):
        return 'mixed_case'
    if name.islower():
        return 'lower_case'
    return 'mixed_case'


def _extract_verified_names(text: str, max_names: int = 80) -> str:
    """Extract identifiers, quoted strings, and negative constraints — language-agnostic."""
    import collections
    names = collections.Counter()
    for match in _UNIVERSAL_ID_RE.finditer(text):
        name = match.group(0)
        if '(' in name:
            name = name.split('(')[0]
        name = name.strip('_')
        if len(name) < 3 or name.upper() in _NOISE_NAMES or name in _NOISE_NAMES:
            continue
        if name.isdigit():
            continue
        names[name] += 1

    mappings = collections.OrderedDict()
    for m in _QUOTED_MAPPING_RE.finditer(text):
        key, val = m.group(1), m.group(2)
        if key not in _NOISE_STRINGS and val not in _NOISE_STRINGS:
            if key not in mappings:
                mappings[key] = []
            if val not in mappings[key]:
                mappings[key].append(val)

    all_quoted = collections.Counter()
    for m in _QUOTED_STRING_RE.finditer(text):
        val = m.group(1)
        if len(val) >= 3 and val not in _NOISE_STRINGS and val.upper() not in _NOISE_NAMES:
            all_quoted[val] += 1
    standalone_strings = []
    seen = set()
    for val, _ in all_quoted.most_common(30):
        if val not in mappings and val not in seen:
            standalone_strings.append(val)
            seen.add(val)

    grouped = collections.defaultdict(list)
    for name, count in names.most_common(max_names):
        cat = _classify_name(name)
        if name not in grouped.get(cat, []):
            grouped[cat].append(name)

    negative_constraints = []
    for m in re.finditer(
        r'(?:NEVER|do\s+NOT|Do\s+not|never|Avoid)\s+([^\n.]{8,80})',
        text, re.IGNORECASE
    ):
        constraint = m.group(0).strip()
        if constraint and constraint not in negative_constraints:
            negative_constraints.append(constraint)
    for m in re.finditer(
        r'(?<!#\s)(?:No\s+(?:SQL\s+)?\w+[^\n]{8,80})',
        text, re.IGNORECASE
    ):
        constraint = m.group(0).strip()
        if re.match(r'^No\s+(?:use|JOIN|data)', constraint, re.IGNORECASE):
            if constraint and constraint not in negative_constraints:
                negative_constraints.append(constraint)

    has_content = (
        any(grouped.values())
        or mappings
        or standalone_strings
        or negative_constraints
    )
    if not has_content:
        return ""

    lines = ["LOSSLESS_INDEX (verified from source — overrides all other sections):"]
    for category in ['upper_case', 'mixed_case', 'lower_case']:
        items = grouped.get(category, [])
        if items:
            lines.append(f"  {category}: {', '.join(items)}")
    if mappings:
        lines.append("  mappings:")
        for key, vals in list(mappings.items())[:25]:
            lines.append(f'    "{key}" → {", ".join(repr(v) for v in vals)}')
    if standalone_strings:
        lines.append(f"  quoted_values: {', '.join(repr(v) for v in standalone_strings[:25])}")
    if negative_constraints:
        lines.append("  negative_constraints:")
        for nc in negative_constraints[:15]:
            lines.append(f'    - "{nc}"')
    return '\n'.join(lines)


_SYMBOL_CHARS = set('{}[]=;:-><')
_CONFIG_LINE_RE = re.compile(
    r'^[a-zA-Z_][\w.-]*\s*[=:]\s*\S'
)
_CODE_SYMBOL_THRESHOLD = 3
_CONFIG_BLOCK_MIN_LINES = 2


def _line_symbol_count(line: str) -> int:
    return sum(1 for c in line if c in _SYMBOL_CHARS)


def _fence_inline_code(text: str) -> str:
    """Pre-process text to fence code-like and config-like blocks as code blocks.

    Language-agnostic. Detects:
    1. Brace-delimited blocks (dicts, structs, objects) via { } matching
    2. Config blocks (YAML, TOML, properties) via consecutive key: value / key = value lines
    3. High symbol-density lines (code that isn't already fenced)
    """
    lines = text.split('\n')
    result = []
    i = 0
    in_fenced = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```'):
            in_fenced = not stripped == '```' and not in_fenced
            if stripped == '```':
                in_fenced = False
            result.append(line)
            i += 1
            continue

        if in_fenced:
            result.append(line)
            i += 1
            continue

        if '{' in line:
            brace_depth = line.count('{') - line.count('}')
            block = [line]
            j = i + 1
            while j < len(lines) and brace_depth > 0:
                brace_depth += lines[j].count('{') - lines[j].count('}')
                block.append(lines[j])
                j += 1
            result.append('```')
            result.extend(block)
            result.append('```')
            i = j
            continue

        if _CONFIG_LINE_RE.match(stripped) and not stripped.startswith('#'):
            config_block = [line]
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith('#') or s.startswith('```'):
                    break
                if _CONFIG_LINE_RE.match(s):
                    config_block.append(lines[j])
                    j += 1
                elif _line_symbol_count(s) >= _CODE_SYMBOL_THRESHOLD:
                    config_block.append(lines[j])
                    j += 1
                else:
                    break
            if len(config_block) >= _CONFIG_BLOCK_MIN_LINES:
                result.append('```')
                result.extend(config_block)
                result.append('```')
            else:
                result.extend(config_block)
            i = j
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


@mcp.tool()
async def textToSmartGraph(
    text: str,
    title: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 32000,
) -> str:
    """
    Convert text to a compressed, LLM-optimized knowledge graph using an LLM pass.

    Takes structured text (Markdown, skill files, docs), first runs textToGraph to extract
    structure, then sends it to an LLM for intelligent compression: cluster analysis,
    rule merging, dependency annotation, and example trimming.

    Reduces token count by 3-4x while preserving all technical semantics (function names,
    variable types, SQL patterns, file paths).

    Uses any OpenAI-compatible chat completions API. Configure via environment variables
    or pass parameters directly.

    Best for: SKILL.md files, coding rules, procedure docs — any structured document
    that will be injected into LLM context repeatedly.

    Args:
        text: The text content to convert (Markdown, plain text, etc.)
        title: Optional title for the graph root node
        api_key: API key (or set SMARTGRAPH_API_KEY env var)
        base_url: API base URL (or set SMARTGRAPH_BASE_URL env var, default: https://api.openai.com/v1)
        model: Model name (or set SMARTGRAPH_MODEL env var, default: gpt-4o-mini)
        max_tokens: Max tokens for LLM response (default: 4096)

    Returns:
        Compressed, cluster-organized knowledge graph optimized for LLM injection
    """
    try:
        import os
        import httpx

        # Step 1: Pre-fence inline Python dicts/signatures, then build graph
        fenced_text = _fence_inline_code(text)
        graph_text = await textToGraph(fenced_text, title=title, output="text")

        if graph_text.startswith("Error:"):
            return graph_text

        # Step 2: Resolve API config
        resolved_key = api_key or os.environ.get("SMARTGRAPH_API_KEY", "")
        resolved_url = base_url or os.environ.get("SMARTGRAPH_BASE_URL", "https://api.openai.com/v1")
        resolved_model = model or os.environ.get("SMARTGRAPH_MODEL", "gpt-4o-mini")

        if not resolved_key:
            return "Error: No API key configured. Set SMARTGRAPH_API_KEY env var or pass api_key parameter."

        # Ensure base_url ends correctly
        base = resolved_url.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = base.rstrip("/") + "/chat/completions"

        # Step 3: Call LLM API
        user_message = SMARTGRAPH_USER_TEMPLATE.format(text=graph_text)

        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": SMARTGRAPH_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.post(base, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]

        content = strip_llm_artifacts(content)

        verified = _extract_verified_names(text)
        if verified:
            content = verified + "\n\n" + content

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        header = f"# Smart Graph: {title or 'Document'}\n# Input: ~{len(graph_text)//4} tokens → Output: ~{output_tokens} tokens (LLM: {resolved_model})\n\n"
        return header + content

    except httpx.HTTPStatusError as e:
        logger.error(f"textToSmartGraph API error: {e.response.status_code} {e.response.text[:200]}")
        return f"Error: API returned {e.response.status_code}: {e.response.text[:200]}"
    except httpx.RequestError as e:
        logger.error(f"textToSmartGraph request error: {e}")
        return f"Error: Request failed: {str(e)}"
    except Exception as e:
        logger.error(f"textToSmartGraph failed: {e}")
        return f"Error: {str(e)}"
