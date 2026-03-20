#!/usr/bin/env python3
"""
SimpleMCP Server - Basic MCP Tools
Provides simple demonstration tools for testing and development.

FEF V3 Integration:
- Management server on port 9002
- Extensions: tool_usage, api_response_times
- Mutators: timeout_config
"""
import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Check for required dependencies before importing
try:
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simplemcp")

# ============================================================================
# Server Initialization
# ============================================================================

# Verify server components
logger.info("Initializing SimpleMCP Server...")

try:
    server = Server("simplemcp")
    sse_transport = SseServerTransport("/messages/")
    logger.info("Server components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server components: {e}")
    sys.exit(1)


# ============================================================================
# FEF V3 Integration
# ============================================================================

# Add parent directory to path for FEF V3 imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    from launcher.tool_extensions import Extension, ExtensionType, ExtensionRegistry
    FEF_V3_AVAILABLE = True
    logger.info("FEF V3 modules loaded successfully")
except ImportError as e:
    FEF_V3_AVAILABLE = False
    logger.warning(f"FEF V3 not available: {e}")

# SimpleMCP-specific metrics
simplemcp_metrics = {
    "double_count": 0,
    "greet_count": 0,
    "total_tool_calls": 0,
    "total_time_ms": 0.0,
}

# Timeout configuration
timeout_config = {
    "default_timeout_ms": 30000,
    "max_timeout_ms": 120000,
}


def get_tool_usage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Data source: Get tool usage statistics."""
    return {
        "double_count": simplemcp_metrics["double_count"],
        "greet_count": simplemcp_metrics["greet_count"],
        "total_tool_calls": simplemcp_metrics["total_tool_calls"],
        "avg_time_ms": round(
            simplemcp_metrics["total_time_ms"] / simplemcp_metrics["total_tool_calls"]
            if simplemcp_metrics["total_tool_calls"] > 0 else 0.0, 2
        )
    }


def get_api_response_times(params: Dict[str, Any]) -> Dict[str, Any]:
    """Data source: Get API response time statistics."""
    return {
        "min_time_ms": 0,
        "max_time_ms": round(simplemcp_metrics["total_time_ms"], 2) if simplemcp_metrics["total_tool_calls"] > 0 else 0,
        "avg_time_ms": round(
            simplemcp_metrics["total_time_ms"] / simplemcp_metrics["total_tool_calls"]
            if simplemcp_metrics["total_tool_calls"] > 0 else 0.0, 2
        )
    }


def set_timeout_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Mutator: Update timeout configuration."""
    previous = timeout_config.copy()
    
    if "default_timeout_ms" in params:
        timeout_config["default_timeout_ms"] = int(params["default_timeout_ms"])
    if "max_timeout_ms" in params:
        timeout_config["max_timeout_ms"] = int(params["max_timeout_ms"])
    
    logger.info(f"[simplemcp] Timeout config updated: {timeout_config}")
    
    return {
        "success": True,
        "message": "Timeout configuration updated",
        "previous": previous,
        "new": timeout_config.copy()
    }


def setup_fef_v3():
    """Set up FEF V3 extensions for simplemcp."""
    if not FEF_V3_AVAILABLE:
        logger.warning("FEF V3 not available, skipping extension setup")
        return None, None, None
    
    # Create custom extensions for simplemcp
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
                        "greet_count": {"type": "integer"},
                        "total_tool_calls": {"type": "integer"}
                    }
                }
            },
            handler=get_tool_usage,
            metadata={
                "description": "Tool usage statistics",
                "category": "metrics"
            }
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
        Extension(
            name="timeout_config",
            ext_type=ExtensionType.MUTATOR,
            schema={
                "input": {
                    "type": "object",
                    "properties": {
                        "default_timeout_ms": {"type": "integer", "minimum": 1000},
                        "max_timeout_ms": {"type": "integer", "minimum": 5000}
                    }
                },
                "output": {"type": "object", "properties": {"success": {"type": "boolean"}}}
            },
            handler=set_timeout_config,
            metadata={"description": "Update timeout configuration", "category": "configuration"}
        ),
    ]
    
    return setup_tool_extensions(
        tool_name="simplemcp",
        mgmt_port=9002,
        custom_extensions=custom_extensions
    )


# Initialize FEF V3
fef_manager, fef_registry, fef_http_server = setup_fef_v3()


# ============================================================================
# SSE Handler
# ============================================================================

async def handle_sse(request):
    """Handle SSE connections for the MCP server."""
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


# ============================================================================
# Tool Definitions
# ============================================================================

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="double",
            description="Doubles the value of a number.",
            inputSchema={
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The number to double."
                    }
                }
            },
        ),
        types.Tool(
            name="square",
            description="Calculates the square of a number.",
            inputSchema={
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The number to square."
                    }
                }
            },
        ),
        types.Tool(
            name="greet",
            description="Generates a greeting message.",
            inputSchema={
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name to greet."
                    },
                    "greeting": {
                        "type": "string",
                        "description": "Optional custom greeting (default: 'Hello')."
                    }
                }
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "double":
            value = float(arguments.get("value", 0))
            result = value * 2
            logger.info(f"Double tool called with value={value}, result={result}")
            return [types.TextContent(type="text", text=str(result))]

        elif name == "square":
            value = float(arguments.get("value", 0))
            result = value ** 2
            logger.info(f"Square tool called with value={value}, result={result}")
            return [types.TextContent(type="text", text=str(result))]

        elif name == "greet":
            name_arg = arguments.get("name", "World")
            greeting = arguments.get("greeting", "Hello")
            result = f"{greeting}, {name_arg}!"
            logger.info(f"Greet tool called with name={name_arg}, greeting={greeting}")
            return [types.TextContent(type="text", text=result)]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in tool call '{name}': {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Resource Handlers
# ============================================================================

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List all available resources."""
    return []


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """List all available resource templates."""
    return []


# ============================================================================
# Create Starlette App
# ============================================================================

app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    async def start_with_fef():
        """Start MCP server with FEF V3 management server."""
        # Start FEF V3 management server if available
        if fef_http_server:
            logger.info("Starting FEF V3 management server on http://0.0.0.0:9002")
            await fef_http_server.start()
        
        # Start MCP server
        config = uvicorn.Config(app, host="0.0.0.0", port=8002)
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
    
    logger.info("Starting SimpleMCP Server on http://0.0.0.0:8002")
    if FEF_V3_AVAILABLE:
        logger.info("FEF V3 management server on http://0.0.0.0:9002")
    
    try:
        if FEF_V3_AVAILABLE:
            asyncio.run(start_with_fef())
        else:
            uvicorn.run(app, host="0.0.0.0", port=8002)
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        if fef_http_server:
            asyncio.run(fef_http_server.stop())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode Configuration Example:

{
  "mcpServers": {
    "simplemcp": {
      "type": "sse",
      "url": "http://localhost:8002/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
"""
