#!/usr/bin/env python3
"""
SimpleMCP8 Server - Streamable HTTP Transport
Provides simple demonstration tools for testing and development using Streamable HTTP transport.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict
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
# Script is at: tools/simplemcp8/simplemcp8_streamable.py
# supreme-mcp-tools is at: . (relative path)
supreme_mcp_tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if supreme_mcp_tools_dir not in sys.path:
    sys.path.insert(0, supreme_mcp_tools_dir)

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
    print("Try running from the supreme-mcp-tools directory: python tools/simplemcp8/simplemcp8_streamable.py", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add protocol version compatibility
SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-11-25"]
logger = logging.getLogger("simplemcp8_streamable")


# ============================================================================
# SimpleMCP8 Streamable HTTP Transport Implementation
# ============================================================================

class SimpleMCP8StreamableHttp(StreamableHttpTransportBase):
    """
    SimpleMCP8 server implementation using Streamable HTTP transport.
    
    This class provides the three simple tools (double, square, greet) using
    the Streamable HTTP transport with JSON-RPC framing.
    """
    
    def __init__(self):
        """Initialize the SimpleMCP8 Streamable HTTP server."""
        config = StreamableHttpConfig(
            endpoint="/mcp",
            framing_format="newline-delimited",
            request_timeout=30.0,
        )
        super().__init__("simplemcp8", config)
        logger.info("SimpleMCP8 Streamable HTTP transport initialized")
    
    async def _handle_initialize(self, params, session):
        """Handle initialize request - only tools are supported."""
        protocol_version = params.get("protocolVersion", "2024-11-05")
        # Support both old and new protocol versions
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            logger.warning(f"Client sent unsupported protocol version: {protocol_version}, using 2024-11-05")
            protocol_version = "2024-11-05"
        
        # Return server capabilities - only tools are supported (matching original simplemcp8)
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
transport = SimpleMCP8StreamableHttp()

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    logger.info("SimpleMCP8 Streamable HTTP server starting up...")
    yield
    # Shutdown
    logger.info("SimpleMCP8 Streamable HTTP server shutting down...")
    await transport.cleanup_sessions()

# Create FastAPI application with lifespan
app = FastAPI(
    title="SimpleMCP8 Streamable HTTP Server",
    description="Simple MCP tools using Streamable HTTP transport",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "simplemcp8",
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
    
    parser = argparse.ArgumentParser(description="SimpleMCP8 Streamable HTTP Server")
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
    logging.getLogger("simplemcp8_streamable").setLevel(log_level)
    
    logger.info(f"Starting SimpleMCP8 Streamable HTTP Server on http://{args.host}:{args.port}")
    
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
    "simplemcp8": {
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
