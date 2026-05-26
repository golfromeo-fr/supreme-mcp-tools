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

# Import FEF V3 setup function from memory_tools
from memory_tools import setup_fef_v3

# ============================================================================
# FEF V3 Setup
# ============================================================================

fef_setup_done = False

try:
    setup_fef_v3()
except Exception as e:
    logger.debug(f"Early FEF setup deferred: {e}")


# ============================================================================
# ASGI App (for launcher)
# Transport is selectable via MCP_TRANSPORT env var:
#   - "streamable-http" (default) → /mcp endpoint
#   - "sse"                    → /sse + /messages endpoints
# ============================================================================

from tools.shared.server_factory import get_transport_app, DEFAULT_HOST

app = get_transport_app(mcp)


# ============================================================================
# Run Server (standalone mode)
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

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )
