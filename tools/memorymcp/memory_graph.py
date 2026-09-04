#!/usr/bin/env python3
"""
Memory Graph Tools - Knowledge graph visualization and manipulation.

This module contains MCP tools for:
- createMemoryEdge: Create a directed edge between two memories
- getMemoryGraph: Get the graph of connected memories
- exportGraphAsMarkdown: Export memories as Markdown with Mermaid diagrams
- memoryTypeChart: Show distribution of memories by type

Uses the FastMCP instance and utilities from memory_core.
"""

import json
from datetime import datetime, timezone

from memory_core import (
    mcp, logger, vector_store,
    COLLECTION_NAME,
    get_now_iso, scroll_all,
)


# ============================================================================
# MCP Tools: Graph Operations
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
    if not vector_store:
        return "Error: Qdrant client not initialized"

    try:
        if from_id == to_id:
            results = vector_store.retrieve(
                COLLECTION_NAME, [from_id],
                with_payload=True,
            )
            if not results:
                return f"Error: Memory not found: {from_id}"
            src = results[0]
            edge = {"to": to_id, "relation": relation, "label": label}
            current_edges = src.payload.get("edges", [])
            if edge not in current_edges:
                vector_store.set_payload(
                    COLLECTION_NAME,
                    {"edges": current_edges + [edge]},
                    ids=[from_id],
                )
            return f"Created self-loop edge: {from_id[:8]} --[{relation}]--> {from_id[:8]}"

        # Verify both memories exist
        results = vector_store.retrieve(
            COLLECTION_NAME, [from_id, to_id],
            with_payload=True,
        )
        if len(results) < 2:
            found = {str(r.id) for r in results}
            missing = [x for x in [from_id, to_id] if x not in found]
            return f"Error: Memory not found: {missing[0]}"

        edge = {"to": to_id, "relation": relation, "label": label}

        src = next(r for r in results if str(r.id) == from_id)
        current_edges = src.payload.get("edges", [])
        if edge not in current_edges:
            vector_store.set_payload(
                COLLECTION_NAME,
                {"edges": current_edges + [edge]},
                ids=[from_id],
            )

        rev_edge = {"to": from_id, "relation": f"back:{relation}", "label": label}
        dst = next(r for r in results if str(r.id) == to_id)
        current_dst_edges = dst.payload.get("edges", [])
        if rev_edge not in current_dst_edges:
            vector_store.set_payload(
                COLLECTION_NAME,
                {"edges": current_dst_edges + [rev_edge]},
                ids=[to_id],
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
    if not vector_store:
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

            results = vector_store.retrieve(
                COLLECTION_NAME, [current_id],
                with_payload=True,
            )
            if not results:
                continue

            payload = results[0].payload
            # Artifact-backed payloads carry text_preview instead of text.
            text = (payload.get("text") or payload.get("text_preview", ""))[:50].replace('"', "'")
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
                lines.append(f'    {safe}["{safe} [{info["type"]}]\n{info["text"]}"]')
            for src, dst, rel, lbl in edges:
                if dst not in nodes:
                    continue
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
                if dst not in nodes:
                    continue
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
    if not vector_store:
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
            # Artifact-backed payloads carry text_preview instead of text.
            text = p.get("text") or p.get("text_preview", "")
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
            # Artifact-backed payloads carry text_preview instead of text.
            preview = (point.payload.get("text") or point.payload.get("text_preview", ""))[:30].replace('"', "'")
            lines.append(f'    {mid}["{mid} [{mtype}]\n{preview}"]')
        point_ids = {str(p.id) for p in points}
        for point in points:
            mid = str(point.id)
            for edge in point.payload.get("edges", []):
                if edge.get("relation", "").startswith("back:"):
                    continue
                to_id = edge.get("to", "")
                if to_id in point_ids:
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
    if not vector_store:
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