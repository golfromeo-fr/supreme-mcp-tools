#!/usr/bin/env python3
"""
SimpleMCP Server - FastMCP Implementation
Provides simple demonstration tools using FastMCP (Streamable HTTP primary, SSE legacy).

Port allocation from ports.json only — no hardcoded ports.
FEF V3 integration preserved from original implementation.
"""
import sys
import os
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

# ============================================================================
# Port Configuration (from ports.json only)
# ============================================================================

TOOL_NAME = "simplemcp"

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
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
    "double_count": 0,
    "square_count": 0,
    "greet_count": 0,
    "total_tool_calls": 0,
    "total_time_ms": 0.0,
    "min_time_ms": float("inf"),
    "max_time_ms": 0.0,
}

# ============================================================================
# FEF V3 Data Sources
# ============================================================================

def get_tool_usage(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get tool usage statistics."""
    return {
        "double_count": metrics["double_count"],
        "square_count": metrics["square_count"],
        "greet_count": metrics["greet_count"],
        "total_tool_calls": metrics["total_tool_calls"],
        "avg_time_ms": round(
            metrics["total_time_ms"] / metrics["total_tool_calls"]
            if metrics["total_tool_calls"] > 0 else 0.0, 2
        )
    }


def get_api_response_times(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get API response time statistics."""
    has_calls = metrics["total_tool_calls"] > 0
    return {
        "min_time_ms": round(metrics["min_time_ms"], 2) if has_calls else 0,
        "max_time_ms": round(metrics["max_time_ms"], 2) if has_calls else 0,
        "avg_time_ms": round(
            metrics["total_time_ms"] / metrics["total_tool_calls"]
            if has_calls else 0.0, 2
        )
    }


# ============================================================================
# FastMCP Instance (via shared factory — DualHeaderVerifier auth)
# ============================================================================

from tools.shared.server_factory import create_fastmcp_server, DEFAULT_HOST

mcp = create_fastmcp_server(TOOL_NAME)


fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


@mcp.tool()
async def double(value: float) -> str:
    """Doubles the value of a number."""
    logger.info(f"double(value={value})")
    start_time = time.perf_counter()
    try:
        result = str(value * 2)
        metrics["double_count"] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        metrics["total_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_time_ms"]:
            metrics["min_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_time_ms"]:
            metrics["max_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="double",
                success=True, duration_ms=elapsed_ms
            )
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="double",
                success=False, duration_ms=elapsed_ms
            )
        raise


@mcp.tool()
async def square(value: float) -> str:
    """Calculates the square of a number."""
    logger.info(f"square(value={value})")
    start_time = time.perf_counter()
    try:
        result = str(value ** 2)
        metrics["square_count"] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        metrics["total_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_time_ms"]:
            metrics["min_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_time_ms"]:
            metrics["max_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="square",
                success=True, duration_ms=elapsed_ms
            )
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="square",
                success=False, duration_ms=elapsed_ms
            )
        raise


@mcp.tool()
async def greet(name: str, greeting: str = "Hello") -> str:
    """Generates a greeting message."""
    logger.info(f"greet(name={name!r}, greeting={greeting!r})")
    start_time = time.perf_counter()
    try:
        result = f"{greeting}, {name}!"
        metrics["greet_count"] += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        metrics["total_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_time_ms"]:
            metrics["min_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_time_ms"]:
            metrics["max_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="greet",
                success=True, duration_ms=elapsed_ms
            )
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="greet",
                success=False, duration_ms=elapsed_ms
            )
        raise


@mcp.tool()
async def get_secret() -> str:
    """Returns the current value of SIMPLEMCP_SECRET env var for hot-reload testing.

    SECURITY NOTE (H-7 — by design):
        This tool intentionally exposes the value of SIMPLEMCP_SECRET to
        authenticated callers.  It exists solely as a development/test fixture
        for verifying that the management UI can hot-reload environment
        variables without restarting the server.  Do NOT place real production
        secrets in SIMPLEMCP_SECRET.  The .env file is git-ignored and scoped
        to the local developer machine.
    """
    logger.info("get_secret()")
    start_time = time.perf_counter()
    try:
        secret_value = os.environ.get("SIMPLEMCP_SECRET", "")
        is_set = bool(secret_value)
        ts = time.strftime("%H:%M:%S")
        result = (
            f"SIMPLEMCP_SECRET value (read at {ts}):\n"
            f"  Value: {secret_value if is_set else '(not set)'}\n"
            f"  Length: {len(secret_value)}\n"
            f"  Is Set: {is_set}\n\n"
            f"To test hot-reload:\n"
            f"1. Update SIMPLEMCP_SECRET via the WebUI\n"
            f"2. Call get_secret again\n"
            f"3. Verify the new value appears without restart"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        metrics["total_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_time_ms"]:
            metrics["min_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_time_ms"]:
            metrics["max_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_secret",
                success=True, duration_ms=elapsed_ms
            )
        return result
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["total_tool_calls"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_secret",
                success=False, duration_ms=elapsed_ms
            )
        raise


# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================


def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", MGMT_PORT))

    custom_extensions = [
        Extension(
            name="tool_usage",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "double_count": {"type": "integer"},
                        "square_count": {"type": "integer"},
                        "greet_count": {"type": "integer"},
                        "total_tool_calls": {"type": "integer"},
                    }
                }
            },
            handler=get_tool_usage,
            metadata={"description": "Tool usage statistics", "category": "metrics"}
        ),
        Extension(
            name="api_response_times",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "min_time_ms": {"type": "number"},
                        "max_time_ms": {"type": "number"},
                        "avg_time_ms": {"type": "number"}
                    }
                }
            },
            handler=get_api_response_times,
            metadata={"description": "API response time statistics", "category": "metrics"}
        ),
    ]

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


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{TOOL_NAME} FastMCP server starting on port {MCP_PORT}...")

    # Setup FEF V3 if not done by launcher
    if not fef_setup_done:
        setup_extensions(registry=None)

    # Start FEF V3 management server if standalone
    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.start()
            logger.info("FEF V3 management server started")
        except Exception as e:
            logger.warning(f"Failed to start FEF V3 management server: {e}")

    yield

    logger.info(f"{TOOL_NAME} FastMCP server shutting down...")
    if fef_http_server:
        try:
            await fef_http_server.stop()
        except Exception:
            pass


# ============================================================================
# App Export (transport selected via MCP_TRANSPORT env var)
# ============================================================================

from tools.shared.server_factory import get_transport_app

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

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http").lower()
    logger.info(f"Starting {TOOL_NAME} FastMCP server (transport: {transport})")
    logger.info(f"  MCP port: {MCP_PORT}")
    if transport == "sse":
        logger.info(f"  SSE endpoint: http://localhost:{MCP_PORT}/sse")
        logger.info(f"  Messages: http://localhost:{MCP_PORT}/messages")
    else:
        logger.info(f"  Streamable HTTP: http://localhost:{MCP_PORT}/mcp")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://localhost:{MGMT_PORT}")

    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )
