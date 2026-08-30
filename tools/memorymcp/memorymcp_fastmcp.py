#!/usr/bin/env python3
"""
Memory MCP Server - FastMCP Entry Point

A memory system for agentic coding with semantic search, provenance, and lifecycle management.

This is the main entry point that:
1. Imports and coordinates all memory modules (core, tools, graph, text)
2. Sets up the FastMCP server with all tools registered
3. Handles FEF V3 extension setup
4. Provides the streamable HTTP app for the launcher

For tool implementations, see:
- memory_core.py: Shared configuration, utilities, and FastMCP instance
- memory_tools.py: CRUD operations (upsert, query, delete, etc.)
- memory_graph.py: Graph operations (edges, export, visualization)
- memory_text.py: Text processing (textToGraph, textToSmartGraph)

Usage:
    python memorymcp_fastmcp.py          # Standalone server
    python -m launcher --tools memorymcp  # Via launcher
"""

import sys
import os
from pathlib import Path

# Ensure tool directory is on sys.path
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Add parent (tools/) to path for shared imports
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ============================================================================
# Import Memory Modules
# ============================================================================

# Import core first (FastMCP instance, config, utilities)
from memory_core import (
    mcp, logger, TOOL_NAME, MCP_PORT, MGMT_PORT,
    SCRIPT_DIR,
)

# Import all tool modules (these register their tools with mcp)
import memory_tools  # noqa: F401 - registers MCP tools
import memory_graph  # noqa: F401 - registers MCP tools
import memory_text  # noqa: F401 - registers MCP tools
import memory_autouse  # noqa: F401 - registers auto-use policy tools and patches getMemorySystemPrompt. Must be imported AFTER memory_tools.

# Import FEF V3 setup function from memory_tools
from memory_tools import setup_fef_v3, setup_extensions  # noqa: F401 - launcher calls setup_extensions by name

# ============================================================================
# FEF V3 Setup
# ============================================================================
# Deliberately NO eager setup_fef_v3() call at import: it ran in standalone
# mode, consumed the fef_setup_done one-shot guard, and starved the launcher's
# later setup_extensions(registry=...) call — leaving memorymcp with an empty
# extensions list in the management API (and a rogue server on port 8105).
# Under the launcher, server_manager calls setup_extensions(registry=...)
# itself after this module loads.
# ============================================================================


# ============================================================================
# ASGI App (for launcher)
# Transports: streamable-http (default, /mcp) or sse (legacy compat, /sse + /messages)
# ============================================================================

from tools.shared.server_factory import get_transport_app, DEFAULT_HOST

app = get_transport_app(mcp)


# ============================================================================
# Run Server (standalone mode)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {TOOL_NAME} FastMCP server (multi-transport: /mcp, /mcp-stateless, /sse)")
    logger.info(f"  MCP port: {MCP_PORT}")

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )
