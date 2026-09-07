"""
databasemcp core — identity, ports, logging, metrics, FastMCP instance.

Formerly the Oracle workbench monolith (renamed
2026-09-06 during the multi-database overhaul; see
plans/databasemcp-overhaul-2026-09-06.md).
"""
import os
import sys
import logging
from pathlib import Path

# ============================================================================
# Port Configuration (from ports.json only)
# ============================================================================

TOOL_NAME = "databasemcp"

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
# Logging
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "databasemcp.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(TOOL_NAME)

# ============================================================================
# FEF V3 Integration
# ============================================================================

try:
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    from launcher.tool_extensions import Extension, ExtensionType
    FEF_V3_AVAILABLE = True
    logger.info("FEF V3 modules loaded successfully")
except ImportError as e:
    FEF_V3_AVAILABLE = False
    logger.warning(f"FEF V3 not available: {e}")

# ============================================================================
# Metrics
# ============================================================================

metrics = {
    "query_count": 0,
    "query_errors": 0,
    "total_query_time_ms": 0.0,
    "min_query_time_ms": float("inf"),
    "max_query_time_ms": 0.0,
    "connection_count": 0,
    "connection_errors": 0,
    "schema_lookups": 0,
    "transactions_begun": 0,
    "transactions_committed": 0,
    "transactions_rolled_back": 0,
    "transactions_reaped": 0,
}

# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server, DEFAULT_HOST  # noqa: E402

mcp = create_fastmcp_server(TOOL_NAME)
