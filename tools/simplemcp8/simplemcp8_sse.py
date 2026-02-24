#!/usr/bin/env python3
"""
SimpleMCP8 Server - Basic MCP Tools
Provides simple demonstration tools for testing and development.
"""
import sys
import os
import logging
from pathlib import Path

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
logger = logging.getLogger("simplemcp8")

# ============================================================================
# Server Initialization
# ============================================================================

# Verify server components
logger.info("Initializing SimpleMCP8 Server...")

try:
    server = Server("simplemcp8")
    sse_transport = SseServerTransport("/messages/")
    logger.info("Server components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server components: {e}")
    sys.exit(1)


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
    logger.info("Starting SimpleMCP8 Server on http://0.0.0.0:8002")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8002)
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode Configuration Example:

{
  "mcpServers": {
    "simplemcp8": {
      "type": "sse",
      "url": "http://localhost:8002/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
"""
