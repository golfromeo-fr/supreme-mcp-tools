#!/usr/bin/env python3
"""
Memory Text Tools - Text processing and knowledge graph extraction.

This module contains MCP tools for:
- textToGraph: Convert structured text to a knowledge graph
- textToSmartGraph: LLM-powered compression of knowledge graphs

Uses the FastMCP instance and utilities from memory_core.
"""

import re
import json
import os
import logging
from datetime import datetime, timezone

from memory_core import (
    mcp, logger,
    get_now_iso,
)

logger = logging.getLogger("memorymcp")


# ============================================================================
# Constants for text processing
# ============================================================================

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
    'MODULE', 'EXPORT', 'REQUIRE', 'DEFAULT', 'ARGS', 'ERROR', 'INFO', 'DEBUG',
    'NONE', 'SELF', 'THIS', 'SUPER', 'TRAIT', 'ENUM', 'match', 'case',
    'impl', 'pub', 'mut', 'ref', 'static', 'async', 'await', 'yield',
    'raise', 'try', 'catch', 'finally', 'with', 'as', 'from', 'import',
    'lambda', 'map', 'filter', 'reduce', 'zip', 'enumerate', 'range',
    'open', 'close', 'read', 'write', 'seek', 'tell', 'flush', 'fileno',
})


# ============================================================================
# Helper Functions
# ============================================================================

def strip_llm_artifacts(text: str) -> str:
    """Strip LLM artifacts like <think> blocks and XML tags from text."""
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


def _extract_verified_names(text: str) -> str:
    """Extract verified technical names from source text."""
    found = set()
    for match in _UNIVERSAL_ID_RE.finditer(text):
        for g in match.groups():
            if g and g not in _NOISE_NAMES:
                found.add(g)
    for match in _QUOTED_STRING_RE.finditer(text):
        val = match.group(1)
        if val not in _NOISE_NAMES:
            found.add(val)
    for match in _QUOTED_MAPPING_RE.finditer(text):
        k, v = match.group(1), match.group(2)
        if k not in _NOISE_NAMES:
            found.add(k)
        if v not in _NOISE_NAMES:
            found.add(v)
    if not found:
        return ""
    lines = ["VERIFIED_NAMES:"]
    for name in sorted(found):
        lines.append(f"  {name}")
    return "\n".join(lines)


# ============================================================================
# MCP Tools: Text Processing
# ============================================================================

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
        text: The text content to convert
        title: Optional title for the output
        api_key: API key for the LLM provider (defaults to SMARTGRAPH_API_KEY env var)
        base_url: Base URL for the API (defaults to SMARTGRAPH_BASE_URL env var)
        model: Model name (defaults to SMARTGRAPH_MODEL env var)
        max_tokens: Maximum output tokens

    Returns:
        Compressed knowledge graph optimized for LLM context
    """
    try:
        import httpx

        # Step 1: Convert to graph first
        graph_text = await textToGraph(text, title or "Document", output="adjacency")
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