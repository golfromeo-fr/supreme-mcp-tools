# Converter MCP Server

A Model Context Protocol (MCP) server that provides document conversion tools, specifically for converting DOCX files to plain text.

## Features

- **DOCX to Text Conversion**: Convert Microsoft Word documents (.docx) to plain text format
- **Multiple Input Sources**: Support for both local file paths and HTTP/HTTPS URLs
- **SharePoint Support**: Automatic fallback to SharePoint REST API for Doc.aspx URLs
- **Output Options**: Return text directly to LLM or write to a specified output file
- **Path Security**: File access restricted to allowed root directories
- **Size Limits**: Configurable maximum file size for downloads (default: 20MB)
- **Authentication**: Support for custom HTTP headers for authenticated downloads
- **Streamable HTTP Transport**: Modern HTTP-based transport with JSON-RPC framing

## Installation

1. Navigate to the convertermcp directory:
   ```bash
   cd supreme-mcp-tools/tools/convertermcp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The server configuration is stored in `config.json`:

```json
{
  "name": "convertermcp",
  "version": "1.0.0",
  "server": {
    "name": "convertermcp",
    "port": 8003,
    "host": "0.0.0.0",
    "transport": "streamable-http",
    "endpoint": "/mcp",
    "framing_format": "newline-delimited",
    "request_timeout": 60.0,
    "sse_endpoint": "/sse",
    "messages_endpoint": "/messages/"
  },
  "allowed_roots": ["/workspaces"],
  "max_docx_size_mb": 20
}
```

### Configuration Options

- **port**: Server port (default: 8003)
- **host**: Server host (default: 0.0.0.0)
- **transport**: Transport type - "streamable-http" (recommended) or "sse"
- **endpoint**: Streamable HTTP endpoint path (default: /mcp)
- **framing_format**: Message framing format - "newline-delimited" or "length-prefixed"
- **request_timeout**: Request timeout in seconds (default: 60.0)
- **allowed_roots**: List of allowed root directories for file access
- **max_docx_size_mb**: Maximum allowed DOCX file size in megabytes

## Usage

### Starting the Server

#### Streamable HTTP (Recommended)

Run the Streamable HTTP version:
```bash
python convertermcp_streamable.py
```

The server will start on `http://0.0.0.0:8003` with the MCP endpoint at `/mcp`

#### SSE (Legacy)

Run the SSE version:
```bash
python convertermcp.py
```

The server will start on `http://0.0.0.0:8003` with the SSE endpoint at `/sse`

### VSCode Integration

#### Streamable HTTP (Recommended)

Add the following to your VSCode `settings.json`:

```json
{
  "mcpServers": {
    "convertermcp": {
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

#### SSE (Legacy)

For backward compatibility, you can still use SSE:

```json
{
  "mcpServers": {
    "convertermcp": {
      "type": "sse",
      "url": "http://localhost:8003/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

### Using with MCP Launcher

The server can also be launched using the MCP launcher system. The launcher will automatically discover and start the convertermcp server.

## Tools

### convert_docx_to_text

Converts a DOCX file (local path or URL) to plain text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | Local .docx path or http/https URL |
| `output_path` | string | No | Optional path to write the output text file |
| `headers` | object | No | Optional HTTP headers for authenticated URL download (e.g., Authorization, Cookie) |

**Examples:**

1. Convert local DOCX file:
   ```json
   {
     "source": "/workspaces/mydocs/document.docx"
   }
   ```

2. Convert DOCX from URL:
   ```json
   {
     "source": "https://example.com/docs/document.docx"
   }
   ```

3. Convert and save to file:
   ```json
   {
     "source": "/workspaces/mydocs/document.docx",
     "output_path": "/workspaces/mydocs/document.txt"
   }
   ```

4. Convert from authenticated URL:
   ```json
   {
     "source": "https://sharepoint.example.com/sites/docs/Doc.aspx?sourcedoc={GUID}",
     "headers": {
       "Authorization": "Bearer your-token-here"
     }
   }
   ```

**Output:**

- If `output_path` is provided: Success message with file sizes
- If `output_path` is not provided: The extracted text content

## Dependencies

- `mcp>=1.0.0` - Model Context Protocol library
- `httpx>=0.27.0` - Async HTTP client
- `fastapi>=0.109.0` - FastAPI framework for Streamable HTTP
- `starlette>=0.27.0` - ASGI framework (for SSE version)
- `uvicorn>=0.27.0` - ASGI server
- `anyio>=4.0.0` - Async I/O library
- `python-docx>=1.1.0` - DOCX file processing

## Migration from SSE to Streamable HTTP

### Why Migrate?

Streamable HTTP transport offers several advantages over SSE:
- **Better Performance**: More efficient HTTP-based communication
- **Improved Compatibility**: Works better with modern HTTP clients and proxies
- **Simpler Debugging**: Easier to test with standard HTTP tools
- **JSON-RPC Framing**: Standard JSON-RPC 2.0 protocol with proper error codes

### Migration Steps

1. **Install Additional Dependencies**:
   ```bash
   pip install fastapi>=0.109.0
   ```

2. **Update VSCode Configuration**:
   
   Change from SSE:
   ```json
   {
     "mcpServers": {
       "convertermcp": {
         "type": "sse",
         "url": "http://localhost:8003/sse",
         "headers": {
           "Content-Type": "application/json"
         }
       }
     }
   }
   ```
   
   To Streamable HTTP:
   ```json
   {
     "mcpServers": {
       "convertermcp": {
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

3. **Switch Server Implementation**:
   
   Stop the SSE version:
   ```bash
   # If running convertermcp.py, stop it with Ctrl+C
   ```
   
   Start the Streamable HTTP version:
   ```bash
   python convertermcp_streamable.py
   ```

### Rollback Instructions

If you need to rollback to SSE:

1. **Revert VSCode Configuration**:
   ```json
   {
     "mcpServers": {
       "convertermcp": {
         "type": "sse",
         "url": "http://localhost:8003/sse",
         "headers": {
           "Content-Type": "application/json"
         }
       }
     }
   }
   ```

2. **Switch Back to SSE Server**:
   ```bash
   python convertermcp.py
   ```

### Compatibility Notes

- Both implementations share the same tool interfaces
- All tool parameters remain unchanged
- The original `convertermcp.py` (SSE version) is preserved for backward compatibility
- You can run both versions simultaneously on different ports if needed

## Security Considerations

- File access is restricted to directories specified in `allowed_roots`
- Maximum file size limits prevent memory issues
- Temporary files are automatically cleaned up
- Path validation prevents directory traversal attacks
- Streamable HTTP uses standard JSON-RPC 2.0 error codes for better error handling

## Troubleshooting

### Import Error: python-docx not installed

```bash
pip install python-docx
```

### Import Error: fastapi not installed

```bash
pip install fastapi>=0.109.0
```

### File not found error

Ensure the file path is within the allowed roots directory. By default, only files under `/workspaces` are accessible.

### HTTP error downloading DOCX

- Check that the URL is accessible
- Verify authentication headers if required
- Ensure the file size is under the limit (20MB by default)

### Connection refused when using Streamable HTTP

- Ensure you're running `convertermcp_streamable.py` instead of `convertermcp.py`
- Check that the server is running on the expected port (default: 8003)
- Verify the endpoint path is `/mcp` for Streamable HTTP

### VSCode cannot connect to the server

- Verify the transport type matches the server implementation
- Check that the URL matches the endpoint ( `/sse` for SSE, `/mcp` for Streamable HTTP)
- Ensure the server is running and accessible
- Check VSCode MCP logs for detailed error messages

## License

This tool is part of the MCP Tools project.

## Contributing

To add new document conversion tools:

1. Add the tool handler function to `convertermcp.py`
2. Register the tool in the `list_tools()` function
3. Add the tool to the handler mapping in `tool_router()`
4. Update this README with the new tool documentation
