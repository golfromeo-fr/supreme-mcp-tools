#!/usr/bin/env python3
"""
SimpleMCP Server - Streamable HTTP Transport
Provides simple demonstration tools for testing and development using Streamable HTTP transport.

FEF V3 Integration:
- Management server on port 9012
- Extensions: tool_usage, api_response_times
- Mutators: timeout_config
"""
import sys
import os
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from contextlib import asynccontextmanager

# Check for required dependencies before importing
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import uvicorn
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Add parent directories to path for importing StreamableHttpTransportBase
# The supreme-mcp-tools directory (parent of tools and launcher) needs to be in the path
# Script is at: tools/simplemcp/simplemcp_streamable.py
# supreme-mcp-tools is at: . (relative path)
supreme_mcp_tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if supreme_mcp_tools_dir not in sys.path:
    sys.path.insert(0, supreme_mcp_tools_dir)

# Import monitoring components (optional - tool should work without monitoring)
try:
    from monitoring.middleware import add_metrics_middleware
    from monitoring.exporters import add_metrics_routes
    from monitoring.collector import MetricsRegistry
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    add_metrics_middleware = None
    add_metrics_routes = None

try:
    from launcher.streamable_http.streamable_http_base import (
        StreamableHttpTransportBase,
        StreamableHttpConfig,
    )
except ImportError as e:
    print(f"ERROR: Cannot import StreamableHttpTransportBase: {e}", file=sys.stderr)
    print(f"Script location: {__file__}", file=sys.stderr)
    print(f"supreme_mcp_tools_dir: {supreme_mcp_tools_dir}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print("Please ensure the launcher/streamable_http module is available.", file=sys.stderr)
    print("Try running from the supreme-mcp-tools directory: python tools/simplemcp/simplemcp_streamable.py", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add protocol version compatibility
SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-11-25"]
logger = logging.getLogger("simplemcp_streamable")


# ============================================================================
# SimpleMCP Streamable HTTP Transport Implementation
# ============================================================================

class SimpleMCPStreamableHttp(StreamableHttpTransportBase):
    """
    SimpleMCP server implementation using Streamable HTTP transport.
    
    This class provides the three simple tools (double, square, greet) using
    the Streamable HTTP transport with JSON-RPC framing.
    """
    
    def __init__(self):
        """Initialize the SimpleMCP Streamable HTTP server."""
        config = StreamableHttpConfig(
            endpoint="/mcp",
            framing_format="newline-delimited",
            request_timeout=30.0,
        )
        super().__init__("simplemcp", config)
        logger.info("SimpleMCP Streamable HTTP transport initialized")
    
    async def _handle_initialize(self, params, session):
        """Handle initialize request - only tools are supported."""
        protocol_version = params.get("protocolVersion", "2024-11-05")
        # Support both old and new protocol versions
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            logger.warning(f"Client sent unsupported protocol version: {protocol_version}, using 2024-11-05")
            protocol_version = "2024-11-05"
        
        # Return server capabilities - only tools are supported (matching original simplemcp)
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {
                    "tools": {},  # Tools are supported
                    # resources and prompts are not included, indicating they're not supported
                },
                "serverInfo": {
                    "name": self.server_name,
                    "version": "1.0.0",
                },
            },
        }
    
    async def _handle_tools_list(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = [
            {
                "name": "double",
                "description": "Doubles the value of a number.",
                "inputSchema": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The number to double."
                        }
                    }
                }
            },
            {
                "name": "square",
                "description": "Calculates the square of a number.",
                "inputSchema": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The number to square."
                        }
                    }
                }
            },
            {
                "name": "greet",
                "description": "Generates a greeting message.",
                "inputSchema": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name to greet."
                        },
                        "greeting": {
                            "type": "string",
                            "description": "Optional custom greeting (default: 'Hello')"
                        }
                    }
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools,
            },
        }
    
    async def _handle_tool_call(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Tool call: {tool_name} with arguments: {arguments}")
        
        try:
            if tool_name == "double":
                value = float(arguments.get("value", 0))
                result = value * 2
                logger.info(f"Double tool: value={value}, result={result}")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                }
            
            elif tool_name == "square":
                value = float(arguments.get("value", 0))
                result = value ** 2
                logger.info(f"Square tool: value={value}, result={result}")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                }
            
            elif tool_name == "greet":
                name_arg = arguments.get("name", "World")
                greeting = arguments.get("greeting", "Hello")
                result = f"{greeting}, {name_arg}!"
                logger.info(f"Greet tool: name={name_arg}, greeting={greeting}")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }
            
            else:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        except ValueError as e:
            logger.error(f"Value error in tool call '{tool_name}': {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": str(e)
                }
            }
        
        except Exception as e:
            logger.error(f"Error in tool call '{tool_name}': {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }


# ============================================================================
# FastAPI Application
# ============================================================================

# Create the transport instance
transport = SimpleMCPStreamableHttp()

# Global FEF V3 variables (initialized lazily)
fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def setup_extensions(registry: Optional["ExtensionRegistry"] = None) -> None:
    """
    Set up FEF V3 extensions for simplemcp.
    
    This function can be called by the launcher after creating a registry,
    or it will be called lazily during lifespan startup.
    
    Args:
        registry: Optional existing registry to use (from launcher)
    """
    global fef_manager, fef_registry, fef_http_server, fef_setup_done
    
    if fef_setup_done:
        logger.info("FEF V3 extensions already set up")
        return
    
    if not FEF_V3_AVAILABLE:
        logger.warning("FEF V3 not available, skipping extension setup")
        fef_setup_done = True
        return
    
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
                        "total_tool_calls": {"type": "integer"}
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
    
    # If a registry is provided (from launcher), use it
    if registry is not None:
        fef_registry = registry
        fef_manager = ToolExtensionManager("simplemcp")
        
        # Register common extensions
        register_common_extensions("simplemcp", fef_registry, fef_manager)
        
        # Register custom extensions
        for ext in custom_extensions:
            fef_registry.register("simplemcp", ext)
            logger.info(f"[simplemcp] Registered custom extension: {ext.name}")
        
        # No HTTP server needed - launcher already has one
        fef_http_server = None
        logger.info("[simplemcp] FEF V3 extensions registered with launcher's registry")
    else:
        # Standalone mode - create our own registry and server
        mgmt_port = int(os.environ.get("MCP_MGMT_PORT", "9012"))
        fef_manager, fef_registry, fef_http_server = setup_tool_extensions(
            tool_name="simplemcp",
            mgmt_port=mgmt_port,
            custom_extensions=custom_extensions
        )
        logger.info(f"[simplemcp] FEF V3 standalone mode on port {mgmt_port}")
    
    fef_setup_done = True


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    global fef_http_server
    
    # Startup
    logger.info("SimpleMCP Streamable HTTP server starting up...")
    
    # Set up FEF V3 extensions if not already done by launcher
    if not fef_setup_done:
        setup_extensions(registry=None)
    
    # Start FEF V3 management server if we have one (standalone mode)
    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.start()
            logger.info("FEF V3 management server started")
        except Exception as e:
            logger.warning(f"Failed to start FEF V3 management server: {e}")
    
    yield
    
    # Shutdown
    logger.info("SimpleMCP Streamable HTTP server shutting down...")
    await transport.cleanup_sessions()
    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.stop()
        except Exception:
            pass

# Create FastAPI application with lifespan
app = FastAPI(
    title="SimpleMCP Streamable HTTP Server",
    description="Simple MCP tools using Streamable HTTP transport",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Monitoring Middleware and Routes
# ============================================================================

# Tool name for metrics labeling
TOOL_NAME = "simplemcp"

# Try to add monitoring middleware and routes (graceful degradation if monitoring unavailable)
if MONITORING_AVAILABLE and add_metrics_middleware is not None:
    try:
        # Get or create the metrics registry
        registry = MetricsRegistry.get_instance()
        
        # Add metrics middleware to track HTTP requests
        # Using collector_name=TOOL_NAME ensures metrics are labeled with "simplemcp"
        add_metrics_middleware(
            app,
            collector_name=TOOL_NAME,
            exclude_paths={"/metrics", "/health", "/stats"}
        )
        logger.info(f"Added metrics middleware for tool: {TOOL_NAME}")
        
        # Add metrics routes (/metrics, /health, /stats)
        add_metrics_routes(
            app,
            collector_name=TOOL_NAME,
            path="/metrics"
        )
        logger.info(f"Added metrics routes for tool: {TOOL_NAME}")
        
    except Exception as e:
        # Monitoring initialization failed - log warning but continue without metrics
        logger.warning(f"Failed to initialize monitoring middleware: {e}. Tool will run without metrics.")
else:
    logger.info("Monitoring not available - running without metrics collection")


# ============================================================================
# FEF V3 Integration
# ============================================================================

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
    "square_count": 0,
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
        "square_count": simplemcp_metrics["square_count"],
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


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "simplemcp",
        "version": "1.0.0",
        "transport": "streamable-http",
        "endpoint": "/mcp",
        "tools": ["double", "square", "greet"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": transport.get_session_count()
    }


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    """
    Handle Streamable HTTP MCP requests.
    
    This endpoint accepts JSON-RPC requests with proper framing and returns
    responses with the same framing format.
    """
    # Read request body
    body = await request.body()
    logger.debug(f"Received request body: {body}")
    
    # Extract headers
    headers = dict(request.headers)
    session_id = headers.get("Mcp-Session-Id")
    logger.debug(f"Request headers: {headers}, session_id: {session_id}")
    
    # Parse request data (expecting newline-delimited JSON)
    try:
        import json
        request_data = json.loads(body.decode("utf-8").strip())
        logger.info(f"Processing JSON-RPC request: method={request_data.get('method')}, id={request_data.get('id')}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse request: {e}")
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e)
                }
            }),
            status_code=400,
            media_type="application/json"
        )
    
    # Process the request
    async def response_generator():
        async for response in transport.handle_request(request_data, headers, session_id):
            # Format response based on framing configuration
            logger.debug(f"Generating response: {response}")
            if transport.config.framing_format == "newline-delimited":
                yield (json.dumps(response) + "\n").encode("utf-8")
            else:
                yield json.dumps(response).encode("utf-8")
    
    return StreamingResponse(
        response_generator(),
        media_type="application/json",
        headers={
            "Content-Type": "application/json",
            "X-MCP-Transport": "streamable-http",
            "X-MCP-Framing": transport.config.framing_format,
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SimpleMCP Streamable HTTP Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port to bind to (default: 8003)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger("simplemcp_streamable").setLevel(log_level)
    
    logger.info(f"Starting SimpleMCP Streamable HTTP Server on http://{args.host}:{args.port}")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode Configuration Example (Streamable HTTP):

{
  "mcpServers": {
    "simplemcp": {
      "type": "streamable-http",
      "url": "http://localhost:8003/mcp",
      "headers": {
        "Content-Type": "application/json"
      },
      "framing": "newline-delimited"
    }
  }
}
"""
