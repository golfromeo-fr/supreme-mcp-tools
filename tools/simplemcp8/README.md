# Simple MCP 8 Server

A simple Model Context Protocol (MCP) server for testing and demonstration purposes. This server supports both SSE (Server-Sent Events) and Streamable HTTP transports.

## Features

- Simple number tools: double, square, and greet
- Minimal dependencies
- Easy to understand code structure
- Perfect for MCP tool development reference
- **NEW**: Streamable HTTP transport support with JSON-RPC framing
- Backward compatible with SSE transport

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. No additional configuration required

## Transports

SimpleMCP8 supports two transport methods:

### SSE Transport (Original)

The original SSE-based transport using Starlette.

**Standalone Mode:**
```bash
python simplemcp8.py
```

The server will start on `http://0.0.0.0:8002` by default.

### Streamable HTTP Transport (New)

The new Streamable HTTP transport using FastAPI with JSON-RPC framing.

**Standalone Mode:**
```bash
python simplemcp8_streamable.py
```

The server will start on `http://0.0.0.0:8003` by default.

**With Custom Port:**
```bash
python simplemcp8_streamable.py --host 0.0.0.0 --port 8080
```

**With Debug Logging:**
```bash
python simplemcp8_streamable.py --log-level debug
```

**Important:** The Streamable HTTP server requires the `launcher` module to be available. Run from the supreme-mcp-tools directory:
```bash
cd supreme-mcp-tools
python tools/simplemcp8/simplemcp8_streamable.py
```

### With Unified Launcher

```bash
python launchmcp.py simplemcp8
```

## Available Tools

### `double`

Doubles the value of a number.

**Parameters:**
- `value` (number, required): The number to double.

**Returns:**
- The doubled value as a string.

**Example:**
```json
{
  "name": "double",
  "arguments": {
    "value": 5
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "10.0"
    }
  ]
}
```

### `square`

Calculates the square of a number.

**Parameters:**
- `value` (number, required): The number to square.

**Returns:**
- The squared value as a string.

**Example:**
```json
{
  "name": "square",
  "arguments": {
    "value": 4
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "16.0"
    }
  ]
}
```

### `greet`

Generates a greeting message.

**Parameters:**
- `name` (string, required): The name to greet.
- `greeting` (string, optional): Custom greeting (default: "Hello").

**Returns:**
- A greeting message as a string.

**Example:**
```json
{
  "name": "greet",
  "arguments": {
    "name": "World",
    "greeting": "Welcome"
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Welcome, World!"
    }
  ]
}
```

## Dependencies

### Core Dependencies

- `mcp>=1.0.0` - MCP framework
- `anyio>=4.0.0` - Async I/O
- `click>=8.0.0` - CLI framework

### SSE Transport Dependencies

- `starlette>=0.27.0` - ASGI framework
- `uvicorn>=0.27.0` - ASGI server

### Streamable HTTP Transport Dependencies

- `fastapi>=0.104.0` - Modern web framework
- `uvicorn>=0.27.0` - ASGI server

## Transport Comparison

| Feature | SSE Transport | Streamable HTTP Transport |
|---------|--------------|---------------------------|
| File | `simplemcp8.py` | `simplemcp8_streamable.py` |
| Port | 8002 | 8003 |
| Framework | Starlette | FastAPI |
| Protocol | Server-Sent Events | JSON-RPC over HTTP |
| Framing | SSE events | Newline-delimited JSON |
| Session Management | Built-in | Configurable |
| Backward Compatible | Yes | New feature |

## VSCode Configuration

### SSE Transport

```json
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
```

### Streamable HTTP Transport

```json
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
```

## Migration from SSE to Streamable HTTP

### Why Migrate?

Streamable HTTP transport offers several advantages:
- **Better compatibility**: Works with more HTTP clients and proxies
- **Simpler debugging**: Standard JSON-RPC over HTTP is easier to debug
- **Session management**: Built-in session support with configurable timeouts
- **Error handling**: Standard JSON-RPC error codes
- **Framing flexibility**: Supports multiple framing formats

### Migration Steps

1. **Install new dependencies:**
   ```bash
   pip install fastapi>=0.104.0
   ```

2. **Update your VSCode configuration:**
   - Change `"type"` from `"sse"` to `"streamable-http"`
   - Change `"url"` from `http://localhost:8002/sse` to `http://localhost:8003/mcp`
   - Add `"framing": "newline-delimited"` to headers

3. **Start the new server:**
   ```bash
   cd supreme-mcp-tools
   python tools/simplemcp8/simplemcp8_streamable.py
   ```

4. **Test the connection:**
   - Open VSCode and verify the server connects
   - Test each tool to ensure proper functionality

### Rollback

If you encounter issues, you can easily rollback to SSE:
1. Stop the Streamable HTTP server
2. Start the SSE server: `python simplemcp8.py`
3. Revert your VSCode configuration to use SSE transport

## MCP Tool Structure

### SSE Transport Structure

The SSE transport demonstrates the standard MCP tool structure:

1. **Server Instance**: Creates an MCP Server instance
2. **SSE Transport**: Configures Server-Sent Events transport
3. **SSE Handler**: Handles SSE connections
4. **Tool Registration**: Registers tools using decorators
5. **Starlette App**: Creates ASGI application with routes
6. **Server Startup**: Runs Uvicorn server

### Streamable HTTP Transport Structure

The Streamable HTTP transport uses a different pattern:

1. **Transport Base**: Extends `StreamableHttpTransportBase`
2. **Tool Handlers**: Implements `_handle_tools_list()` and `_handle_tool_call()`
3. **FastAPI App**: Creates FastAPI application with endpoints
4. **JSON-RPC Framing**: Uses newline-delimited JSON for messages
5. **Session Management**: Built-in session support
6. **Error Handling**: Standard JSON-RPC error codes

## Development

### Adding New Tools (Streamable HTTP)

To add a new tool to the Streamable HTTP version:

1. Add the tool to the `_handle_tools_list()` method
2. Implement the handler in `_handle_tool_call()` method
3. Define input schema for parameters

Example:
```python
async def _handle_tools_list(self, params, session):
    tools = [
        # ... existing tools ...
        {
            "name": "my_tool",
            "description": "Description of my tool",
            "inputSchema": {
                "type": "object",
                "required": ["param"],
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                }
            }
        }
    ]
    return {"jsonrpc": "2.0", "result": {"tools": tools}}

async def _handle_tool_call(self, params, session, request_id):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name == "my_tool":
        param = arguments.get("param", "")
        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": f"Result: {param}"
                    }
                ]
            }
        }
```

## Troubleshooting

### Port Already in Use

If you get a port conflict error:
- Change the port using command-line arguments: `--port 8080`
- Or stop the process using the port

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### StreamableHttpTransportBase Not Found

If you get an import error for `StreamableHttpTransportBase`:
- Ensure you're running from the correct directory
- The launcher module should be available in the parent directories
- Check that the path is correctly added in the script

### Connection Refused

If VSCode cannot connect to the server:
- Verify the server is running
- Check the port number matches your configuration
- Ensure the URL is correct (include `/mcp` for Streamable HTTP)
- Check firewall settings

### Import Error: Cannot import StreamableHttpTransportBase

If you get an import error when running the Streamable HTTP server:
- Ensure you're running from the supreme-mcp-tools directory:
  ```bash
  cd supreme-mcp-tools
  python tools/simplemcp8/simplemcp8_streamable.py
  ```
- Verify the launcher/streamable_http module exists
- Check that the launcher directory has an `__init__.py` file

## License

This tool is part of the MCP tools ecosystem.
