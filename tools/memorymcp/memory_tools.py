#!/usr/bin/env python3
"""
Memory Tools - Core CRUD operations for memory management.

This module contains the main MCP tools for:
- upsertMemory: Store or update a memory
- queryMemory: Search memories with recency/usage weighting
- getMemory: Retrieve a specific memory by ID
- deleteMemory: Delete a memory
- listMemoryTypes: List available memory types
- decayOrExpire: Clean up expired memories
- attachProvenance: Add provenance metadata
- redactSensitive: Detect and redact PII
- getMemoryMetrics: Get system statistics
- onAgentAction: Capture agent action context
- getMemorySystemPrompt: Get LLM injection prompt
- reindexMemory: Re-index with new embedding model
- auditTrail: Get memory retrieval history
- mergeDuplicates: Find and merge duplicate memories

All tools use the FastMCP instance and utilities from memory_core.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_core import (
    mcp, logger, qdrant_client, pg_store,
    COLLECTION_NAME, SCRIPT_DIR,
    generate_embedding, get_now_iso, get_memory_id,
    scroll_all, parse_memory_type,
    memory_item_to_payload, payload_to_memory_hit,
    MemoryItem, RetentionPolicy, ScoringWeights, score_relevance,
    redact_sensitive_text, check_sensitivity,
    Filter, FieldCondition, MatchValue,
)

# For FEF V3 extensions
try:
    from launcher.tool_extensions import Extension, ExtensionType
    from tools.fef_integration import (
        ToolExtensionManager, register_common_extensions, setup_tool_extensions
    )
    FEF_V3_AVAILABLE = True
except ImportError:
    FEF_V3_AVAILABLE = False


# ============================================================================
# MCP Tools: CRUD Operations
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
    logger.info(f"upsertMemory: {'update' if memory_id else 'create'}, type={memory_type}, text_len={len(text)}")

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
        logger.info(f"{'Updated' if is_update else 'Stored'} memory {memory_id}")

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
        weights.alpha = 1.0 - recency_weight
        weights.beta = recency_weight

        now_iso = get_now_iso()
        usage_updates: dict[int, list[str]] = {}

        for result in results.points:
            payload = result.payload
            payload["id"] = str(result.id)

            relevance = score_relevance(payload, query_embedding, weights, semantic_score=result.score)

            hit = payload_to_memory_hit(payload, result.score)
            hit.score = relevance
            hits.append(hit)

            new_usage = payload.get("usage_count", 0) + 1
            usage_updates.setdefault(new_usage, []).append(str(result.id))

        for new_usage, point_ids in usage_updates.items():
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"last_accessed": now_iso, "usage_count": new_usage},
                points=point_ids,
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
        now_iso = get_now_iso()
        qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "last_accessed": now_iso,
                "usage_count": new_usage,
            },
            points=[memory_id],
        )

        payload["usage_count"] = new_usage
        payload["last_accessed"] = now_iso
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
    from memory_core import get_redactor
    
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
# FEF V3 Extensions
# ============================================================================

fef_setup_done = False

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

    from memory_core import MemoryType

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