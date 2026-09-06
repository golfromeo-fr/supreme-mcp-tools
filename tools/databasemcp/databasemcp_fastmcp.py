#!/usr/bin/env python3
"""
DatabaseMCP Server - FastMCP Entry Point

Database workbench MCP (formerly the Oracle-only workbench): Oracle today; the multi-database
connection registry (Oracle pool / Postgres / libSQL) lands in P2 of
plans/databasemcp-overhaul-2026-09-06.md.

Port allocation from ports.json only — no hardcoded ports.
FEF V3 integration for distributed tool management.
"""
import sys
from pathlib import Path

# Ensure tool directory is on sys.path (flat module imports: core, connections, db_tools)
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Add parent (tools/) to path for shared imports
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ============================================================================
# Import core first (ports guard, FastMCP instance, metrics), then tools
# ============================================================================

from core import mcp, logger, TOOL_NAME, MCP_PORT, MGMT_PORT  # noqa: E402

import db_tools  # noqa: F401,E402 - registers MCP tools
from db_tools import setup_extensions, FEF_V3_AVAILABLE  # noqa: F401,E402 - launcher calls setup_extensions by name

# ============================================================================
# ASGI App (for launcher) — multi-transport: /mcp, /mcp-stateless, /sse
# ============================================================================

from tools.shared.server_factory import get_transport_app, DEFAULT_HOST  # noqa: E402
from tools.shared.function_masks import apply_function_masks  # noqa: E402

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
