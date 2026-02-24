#!/usr/bin/env python3
"""
Example Streamable HTTP Server for Testing

This is a simple example server that demonstrates how to use the
Streamable HTTP transport base implementation.

Usage:
    python example_server.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import uvicorn
    
    from launcher.streamable_http import (
        StreamableHttpTransportBase,
        StreamableHttpConfig,
        StreamableHttpFraming,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("  pip install fastapi uvicorn httpx")
    sys.exit(1)


class ExampleToolTransport(StreamableHttpTransportBase):
    """Example tool transport implementation."""
    
    def __init__(self):
        config = StreamableHttpConfig(
            endpoint="/mcp",
            framing_format="newline-delimited",
        )
        super().__init__("example_tool", config)
    
    async def _handle_tools_list(self, params, session):
        """Return list of available tools."""
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the input text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to echo back"
                                }
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "add",
                        "description": "Add two numbers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["a", "b"]
                        }
                    },
                    {
                        "name": "greet",
                        "description": "Greet the user",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name to greet"
                                }
                            },
                            "required": ["name"]
                        }
                    }
                ]
            }
        }
    
    async def _handle_tool_call(self, params, session, request_id):
        """Handle tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "echo":
                result = await self._echo(arguments.get("text", ""))
            elif tool_name == "add":
                result = await self._add(arguments.get("a", 0), arguments.get("b", 0))
            elif tool_name == "greet":
                result = await self._greet(arguments.get("name", "World"))
            else:
                yield {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    },
                    "id": request_id
                }
                return
            
            yield {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                },
                "id": request_id
            }
        
        except Exception as e:
            yield {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                },
                "id": request_id
            }
    
    async def _echo(self, text: str) -> str:
        """Echo back the input text."""
        await asyncio.sleep(0.1)  # Simulate some work
        return f"Echo: {text}"
    
    async def _add(self, a: float, b: float) -> float:
        """Add two numbers."""
        await asyncio.sleep(0.1)  # Simulate some work
        return a + b
    
    async def _greet(self, name: str) -> str:
        """Greet the user."""
        await asyncio.sleep(0.1)  # Simulate some work
        return f"Hello, {name}!"


def create_app():
    """Create and configure the FastAPI application."""
    # Create transport
    transport = ExampleToolTransport()
    
    # Create FastAPI app
    app = FastAPI(
        title="Example Streamable HTTP Server",
        description="A simple example server for testing Streamable HTTP transport",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Example Streamable HTTP Server",
            "transport": "streamable-http",
            "endpoint": "/mcp"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.post("/mcp")
    async def handle_mcp(request: Request):
        """Handle MCP requests."""
        try:
            body = await request.json()
            headers = dict(request.headers)
            
            async def generate():
                async for response in transport.handle_request(body, headers):
                    yield StreamableHttpFraming.encode_message(response, transport.config)
            
            return StreamingResponse(
                generate(),
                media_type="application/json",
                headers={
                    "Transfer-Encoding": "chunked",
                }
            )
        
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Server error: {str(e)}"
                },
                "id": None
            }
    
    return app


def main():
    """Main entry point."""
    app = create_app()
    
    print("=" * 60)
    print("Example Streamable HTTP Server")
    print("=" * 60)
    print()
    print("Starting server on http://0.0.0.0:8000")
    print("MCP endpoint: http://0.0.0.0:8000/mcp")
    print("Health check: http://0.0.0.0:8000/health")
    print()
    print("Available tools:")
    print("  - echo: Echo back the input text")
    print("  - add: Add two numbers")
    print("  - greet: Greet the user")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
