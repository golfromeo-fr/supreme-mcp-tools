#!/usr/bin/env python3
"""
memorymcp Auto-Use Policy - MCP shortcut tools.

This module exposes the auto-use policy to the LLM through MCP tools that
read repo-local policy files:

1. auto_use_policy.md — full policy body
2. auto_use_cheatsheet.md — short summary for tight context budgets

Both files live in tools/memorymcp/ alongside this module.

The pre-existing `getMemorySystemPrompt()` tool in memory_tools.py is also
patched at import time to prepend a short pointer that teaches the LLM the
names of these new tools.
"""

from pathlib import Path

from memory_core import mcp, logger, SCRIPT_DIR

# ---------------------------------------------------------------------------
# Paths — repo-local, always tracked in git
# ---------------------------------------------------------------------------

POLICY_FILE = SCRIPT_DIR / "auto_use_policy.md"
CHEATSHEET_FILE = SCRIPT_DIR / "auto_use_cheatsheet.md"

# Prepended at the top of getMemorySystemPrompt() output so the LLM
# learns the names of the new tools on first injection.
AUTOUSE_POLICY_POINTER = (
    "## Auto-Use Policy (read first)\n"
    "\n"
    "memorymcp is a reflex, not a lookup DB. Call `getMemoryCheatsheet()` at session start\n"
    "for a short summary, or `getMemoryAutousePolicy()` for the full policy body.\n"
    "\n"
    "---\n"
    "\n"
)

INLINE_POLICY_FALLBACK = (
    "# memorymcp - auto-use policy (inline fallback)\n"
    "\n"
    "Treat memorymcp as a reflex, not a lookup DB.\n"
    "\n"
    "**Contract**\n"
    "- Session start: queryMemory for project / agent context.\n"
    "- Before any non-trivial task: queryMemory for prior art.\n"
    "- After file_open / file_edit / test_run / commit / discovery: onAgentAction (one-liner).\n"
    "- End of task: upsertMemory for any pattern, decision, trick, or lesson.\n"
    "- Related to existing memory: createMemoryEdge (relation: refines / depends_on / follows / contradicts / example_of).\n"
    "\n"
    "**End-of-turn checklist:** query / store / link / self-check\n"
    "\n"
    "**Redact first if uncertain:** redactSensitive, then upsertMemory.\n"
    "\n"
    "**Anti-patterns:** do NOT skip query for small tasks, do NOT store secrets, do NOT store without a memory_type + tag, do NOT store generic LLM filler.\n"
)

INLINE_CHEATSHEET_FALLBACK = (
    "# memorymcp - cheatsheet (inline fallback)\n"
    "\n"
    "- Session start: queryMemory\n"
    "- Before non-trivial task: queryMemory\n"
    "- After file_open / file_edit / test_run / commit: onAgentAction\n"
    "- End of task: upsertMemory\n"
    "- Related memory: createMemoryEdge\n"
    "\n"
    "Redact secrets first (redactSensitive). Type + tag every memory.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file_or_fallback(path: Path, fallback: str, label: str) -> str:
    try:
        if path.exists():
            return path.read_text()
    except Exception as e:
        logger.warning(f"Could not read {label} file {path}: {e}")
    return fallback


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def getMemoryAutousePolicy() -> str:
    """
    Return the full memorymcp auto-use policy body.

    Tip: Call this at session start to load the reflex policy into your context.
    Prefer `getMemoryCheatsheet()` if your context budget is tight.

    Returns:
        Full auto-use policy as Markdown.
    """
    return _read_file_or_fallback(POLICY_FILE, INLINE_POLICY_FALLBACK, "policy")


@mcp.tool()
async def getMemoryCheatsheet() -> str:
    """
    Return the short memorymcp cheatsheet for tight context budgets.

    Tip: Call this at session start as the minimum-viable reflex primer.
    For the full policy, call `getMemoryAutousePolicy()` instead.

    Returns:
        Short cheatsheet as Markdown (~25 lines).
    """
    return _read_file_or_fallback(CHEATSHEET_FILE, INLINE_CHEATSHEET_FALLBACK, "cheatsheet")


# ---------------------------------------------------------------------------
# 2-level analysis shortcut
# ---------------------------------------------------------------------------

@mcp.tool()
async def getMetaDecisions(query: str, k: int = 5, priority: str = "A") -> str:
    """
    Query only the meta-level architectural decisions about a topic,
    filtered by priority tier.

    Tip: Use this when asking "why was X built this way?" or "what are the
    big design trade-offs for X?". Returns only memories tagged
    `level:meta` AND `priority:<priority>`, avoiding the noise of code
    patterns and less-critical decisions.

    Priority tier:
    - "A" (default): must-know, the curated set of high-signal decisions
    - "B": should-know, broader set including less central decisions
    - "C": nice-to-know, exhaustive including edge-case decisions

    For implementation details, use `queryMemory(query)` instead, which
    returns both `level:meta` and `level:detail` memories.

    Args:
        query: Search query (e.g., "authentication", "vector storage",
               "process model", "port allocation").
        k: Maximum number of meta decisions to return (default: 5).
        priority: Priority tier to filter by, "A" | "B" | "C" (default "A").

    Returns:
        Formatted list of meta-level architectural decisions, same
        format as `queryMemory`.
    """
    if priority not in ("A", "B", "C"):
        priority = "A"  # fail-safe default; don't trust user input

    # Defer import to avoid any chance of cycle (memory_autouse is imported
    # after memory_tools in memorymcp_fastmcp.py, so a top-level import
    # would also work — this is just defensive).
    from memory_tools import queryMemory

    return await queryMemory(
        query=query,
        k=k,
        tags=["level:meta", f"priority:{priority}"],
    )


# ---------------------------------------------------------------------------
# Prepend pointer to getMemorySystemPrompt()
# ---------------------------------------------------------------------------

def _patch_memory_system_prompt():
    try:
        import memory_tools
    except Exception as e:
        logger.debug(f"Could not import memory_tools to patch: {e}")
        return

    original = getattr(memory_tools, "getMemorySystemPrompt", None)
    if original is None:
        logger.debug("memory_tools.getMemorySystemPrompt not found, skipping patch")
        return
    if getattr(original, "_autouse_patched", False):
        return

    async def patched_get_memory_system_prompt() -> str:
        body = await original()
        return AUTOUSE_POLICY_POINTER + body

    patched_get_memory_system_prompt._autouse_patched = True
    memory_tools.getMemorySystemPrompt = patched_get_memory_system_prompt
    logger.info("getMemorySystemPrompt patched with auto-use policy pointer")


_patch_memory_system_prompt()
