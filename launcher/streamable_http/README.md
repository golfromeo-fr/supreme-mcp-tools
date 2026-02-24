# Streamable HTTP Transport

This module provides the base infrastructure for Streamable HTTP transport for MCP (Model Context Protocol) tools. It implements JSON-RPC framing with newline-delimited JSON, chunked transfer encoding, and connection management.

## Overview

Streamable HTTP is a transport layer for MCP that uses standard HTTP with JSON-RPC messages. It provides better proxy compatibility, bidirectional communication, and alignment with modern MCP specifications compared to Server-Sent Events (SSE).

## Features

- **JSON-RPC Framing**: Supports newline-delimited and length-prefixed JSON framing
- **Chunked Transfer Encoding**: Proper HTTP chunked encoding with explicit flushes
- **Session Management**: Built-in session tracking and timeout handling
- **Error Handling**: Comprehensive error handling with JSON-RPC error codes
- **Reconnection Logic**: Automatic reconnection with exponential backoff
- **Streaming Support**: Async streaming for multiple responses
- **Proxy Compatible**: Works with standard HTTP proxies and load balancers

## Installation

The Streamable HTTP transport is included in the MCP launcher. Ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

Required dependencies:
- `modelcontextprotocol>=1.0.0` - Official MCP SDK
- `fastapi>=0.104.0` - Modern async HTTP server framework
- `uvicorn>=0.30.0` - ASGI server
- `httpx>=0.27.0` - Async HTTP client

## Architecture

### Components

1. **StreamableHttpTransportBase**: Base class for server-side transport implementation
2. **StreamableHttpClient**: Client for communicating with Streamable HTTP servers
3. **StreamableHttpFraming**: Message framing and encoding/decoding
4. **StreamableHttpConfig**: Configuration dataclass
5. **ReconnectionManager**: Automatic reconnection handling

### Framing Format

#### Newline-Delimited JSON (Default)

Each JSON-RPC message ends with a newline character:

```
{"jsonrpc":"2.0","method":"initialize","params":{...}}
{"jsonrpc":"2.0","method":"tools/list","params":{...}}
```

#### Length-Prefixed JSON

Each message is prefixed with a 4-byte length header (big-endian):

```
<4-byte length>{"jsonrpc":"2.0","method":"initialize","params":{...}}
```

## Usage

### Server-Side Usage

#### Basic Server

```python
from fastapi import FastAPI, Request
from launcher.streamable_http import StreamableHttpTransportBase, StreamableHttpConfig

# Create configuration
config = StreamableHttpConfig(
    endpoint="/mcp",
    framing_format="newline-delimited",
)

# Create transport
transport = StreamableHttpTransportBase("my_tool", config)

# Create FastAPI app
app = FastAPI()

# Add MCP endpoint
@app.post("/mcp")
async def handle_mcp(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    
    # Stream responses
    from fastapi.responses import StreamingResponse
    
    async def generate():
        async for response in transport.handle_request(body, headers):
            yield StreamableHttpFraming.encode_message(response, config)
    
    return StreamingResponse(generate(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Custom Tool Implementation

```python
from launcher.streamable_http import StreamableHttpTransportBase, StreamableHttpConfig

class MyToolTransport(StreamableHttpTransportBase):
    async def _handle_tools_list(self, params, session):
        """Return list of available tools."""
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "my_tool",
                        "description": "A sample tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        }
    
    async def _handle_tool_call(self, params, session, request_id):
        """Handle tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        # Execute tool logic
        result = await self._execute_tool(tool_name, arguments)
        
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
    
    async def _execute_tool(self, name, arguments):
        """Execute the actual tool logic."""
        # Your tool implementation here
        return f"Tool {name} called with {arguments}"
```

### Client-Side Usage

#### Basic Client

```python
import asyncio
from launcher.streamable_http import StreamableHttpClient, ClientConfig

async def main():
    # Create client
    config = ClientConfig(
        base_url="http://localhost:8000",
        endpoint="/mcp"
    )
    client = StreamableHttpClient(config)
    
    # Connect to server
    await client.connect()
    
    try:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        
        # Call a tool
        async for response in client.call_tool("my_tool", {"input": "hello"}):
            print(f"Response: {response}")
    
    finally:
        # Disconnect
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Event Handling

```python
from launcher.streamable_http import StreamableHttpClient

async def on_connected():
    print("Connected to server")

async def on_disconnected():
    print("Disconnected from server")

async def on_connection_failed(error):
    print(f"Connection failed: {error}")

client = StreamableHttpClient()
client.on("connected", on_connected)
client.on("disconnected", on_disconnected)
client.on("connection_failed", on_connection_failed)
```

#### Reconnection Management

```python
from launcher.streamable_http import StreamableHttpClient, ReconnectionManager

client = StreamableHttpClient()
reconnect_manager = ReconnectionManager(
    client,
    max_attempts=10,
    initial_delay=2.0,
    backoff_multiplier=2.0
)

# Start reconnection manager
await reconnect_manager.start()

# Later, stop it
await reconnect_manager.stop()
```

## Configuration

### Server Configuration

```python
from launcher.streamable_http import StreamableHttpConfig

config = StreamableHttpConfig(
    # Endpoint paths
    endpoint="/mcp",
    messages_path="/messages/",
    
    # Framing
    framing_format="newline-delimited",  # or "length-prefixed"
    encoding="utf-8",
    
    # Timeouts
    request_timeout=30.0,
    connection_timeout=10.0,
    
    # Retry
    max_retries=3,
    retry_delay=1.0,
    
    # Chunked transfer
    chunk_size=8192,
    enable_chunked_encoding=True,
    
    # Session management
    enable_session_management=True,
    session_timeout=300.0,
    
    # Error handling
    include_stack_traces=False,
    max_error_message_length=1000,
)
```

### Client Configuration

```python
from launcher.streamable_http import ClientConfig

config = ClientConfig(
    # Server connection
    base_url="http://localhost:8000",
    endpoint="/mcp",
    
    # Timeouts
    request_timeout=30.0,
    connection_timeout=10.0,
    read_timeout=60.0,
    
    # Retry
    max_retries=3,
    retry_delay=1.0,
    backoff_multiplier=2.0,
    
    # Reconnection
    enable_auto_reconnect=True,
    reconnect_delay=2.0,
    max_reconnect_attempts=10,
    
    # Framing
    framing_format="newline-delimited",
    encoding="utf-8",
    
    # Session
    session_id=None,
    enable_session_persistence=True,
)
```

## JSON-RPC Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON was received |
| -32600 | Invalid Request | JSON-RPC request is invalid |
| -32601 | Method not found | Method does not exist |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal JSON-RPC error |

## Migration from SSE

### Key Differences

| Feature | SSE | Streamable HTTP |
|---------|-----|-----------------|
| Format | `text/event-stream` | JSON chunks |
| Request Model | Long-lived GET | HTTP POST |
| Proxy Support | Poor | Excellent |
| Bidirectional | Limited | Full duplex |

### Migration Steps

1. Replace `SseServerTransport` with `StreamableHttpTransportBase`
2. Update endpoint routes to use `/mcp` instead of `/sse`
3. Update message framing from SSE to newline-delimited JSON
4. Update client to use `StreamableHttpClient` instead of SSE client
5. Test all tool functionality

### Example Migration

**Before (SSE):**
```python
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette

sse_transport = SseServerTransport("/messages/")
app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse_transport.handle_post_message),
])
```

**After (Streamable HTTP):**
```python
from launcher.streamable_http import StreamableHttpTransportBase
from fastapi import FastAPI

streamable_transport = StreamableHttpTransportBase("my_tool")
app = FastAPI()

@app.post("/mcp")
async def handle_mcp(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    
    async def generate():
        async for response in streamable_transport.handle_request(body, headers):
            yield StreamableHttpFraming.encode_message(response, streamable_transport.config)
    
    return StreamingResponse(generate(), media_type="application/json")
```

## Testing

### Running Tests

```bash
# Run all tests
pytest launcher/streamable_http/tests/

# Run specific test
pytest launcher/streamable_http/tests/test_streamable_http.py::test_framing
```

### Test Utilities

The module includes test utilities for validating the implementation:

```python
from launcher.streamable_http import StreamableHttpFraming, StreamableHttpConfig

# Test framing
config = StreamableHttpConfig()
message = {"jsonrpc": "2.0", "method": "test", "params": {}}

encoded = StreamableHttpFraming.encode_message(message, config)
decoded = StreamableHttpFraming.decode_message(encoded, config)

assert decoded == message
```

## Troubleshooting

### Connection Issues

**Problem**: Client cannot connect to server

**Solutions**:
1. Check server is running: `curl http://localhost:8000/mcp`
2. Verify firewall settings
3. Check logs for error messages
4. Ensure httpx is installed: `pip install httpx`

### Framing Issues

**Problem**: Messages not being parsed correctly

**Solutions**:
1. Verify framing format matches between client and server
2. Check encoding settings
3. Ensure messages end with newline for newline-delimited format
4. Use length-prefixed format for binary data

### Session Issues

**Problem**: Sessions expiring too quickly

**Solutions**:
1. Increase `session_timeout` in configuration
2. Implement session refresh logic
3. Check session ID is being sent in headers

## Performance Considerations

- **Chunk Size**: Default 8192 bytes. Increase for larger payloads, decrease for lower latency.
- **Timeouts**: Adjust based on expected operation duration.
- **Connection Pooling**: httpx automatically manages connection pooling.
- **Compression**: Consider enabling HTTP compression for large payloads.

## Security Considerations

- Always use HTTPS in production
- Validate all incoming requests
- Implement rate limiting
- Use authentication/authorization headers
- Sanitize error messages in production

## API Reference

See inline documentation in source files for complete API reference:

- [`streamable_http_base.py`](streamable_http_base.py) - Server-side implementation
- [`streamable_http_client.py`](streamable_http_client.py) - Client-side implementation

## Contributing

When contributing to the Streamable HTTP transport:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

This module is part of the MCP launcher project.

## Related Documentation

- [SSE to Streamable HTTP Migration Plan](../../../SSE_TO_STREAMABLE_HTTP_MIGRATION.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
