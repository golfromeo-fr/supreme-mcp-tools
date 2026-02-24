# MCP Launcher

> My need is being able to launch MCP server tools in a flexible manner depending on my current needs.

**Status**: Tested as single user server (not yet tested as multi-user)

**Available Tools**: `simplemcp8` `webmcp` `ragmcp` `convertermcp` `oraclemcp`

A unified launcher system for running multiple MCP (Model Context Protocol) tools in a single Python process.
This reduces memory usage by approximately 50% compared to running each tool as a separate process.

---

## ✅ WORKING

| Tool | Description |
|------|-------------|
| `simplemcp8` | Simple test tools (double, square, greet) |
| `webmcp` | Web search (Brave Search, Google API), URL fetch, HTTP POST |

---

## 🚧 NOT WORKING Yet

| Tool | Description |
|------|-------------|
| `ragmcp` | RAG-like codebase indexing using local or API embeddings |
| `convertermcp` | Document conversion (DOCX to TXT) |
| `oraclemcp` | Oracle database tools to feed the LLM |

---

## 📖 Example

```bash
# Terminal session
$ python launchmcp.py simplemcp8
```

**Output:**
```
2026-02-24 02:05:29,013 - root - INFO - ============================================================
2026-02-24 02:05:29,013 - root - INFO - MCP Launcher Starting
2026-02-24 02:05:29,013 - root - INFO - ============================================================
2026-02-24 02:05:29,013 - root - INFO - Searching only in directories for requested tools: ['path/to/supreme-mcp-tools/tools/simplemcp8']
2026-02-24 02:05:29,013 - launcher.tool_discovery - INFO - Searching for MCP tools in: path/to/supreme-mcp-tools/tools/simplemcp8
2026-02-24 02:05:29,591 - launcher.streamable_http.streamable_http_base - INFO - StreamableHttpTransport initialized for 'simplemcp8'
2026-02-24 02:05:29,591 - simplemcp8_streamable - INFO - SimpleMCP8 Streamable HTTP transport initialized
2026-02-24 02:05:29,592 - launcher.tool_discovery - INFO - Discovered tool: simplemcp8 from path/to/supreme-mcp-tools/tools/simplemcp8/simplemcp8_streamable.py
2026-02-24 02:05:29,592 - root - INFO - Discovered 1 MCP tools: ['simplemcp8']
2026-02-24 02:05:29,592 - launcher.port_manager - INFO - Allocated port 8002 for tool simplemcp8
2026-02-24 02:05:29,592 - root - INFO - Allocated ports: {'simplemcp8': 8002}
2026-02-24 02:05:29,592 - root - INFO - Starting 1 servers...
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Starting server for simplemcp8 on port 8002
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Server for simplemcp8 starting on port 8002
2026-02-24 02:05:29,592 - root - INFO - Server for simplemcp8 started on port 8002
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Running server for simplemcp8 on port 8002
INFO:     Started server process [2850649]
INFO:     Waiting for application startup.
2026-02-24 02:05:29,606 - simplemcp8_streamable - INFO - SimpleMCP8 Streamable HTTP server starting up...
2026-02-24 02:05:29,606 - root - INFO - Successfully started 1/1 servers

============================================================
MCP Launcher Running
============================================================
  simplemcp8: http://0.0.0.0:8002
============================================================
Press Ctrl+C to stop all servers

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
2026-02-24 02:05:29,702 - simplemcp8_streamable - INFO - Processing JSON-RPC request: method=initialize, id=0
INFO:     127.0.0.1:35188 - "POST /mcp HTTP/1.1" 200 OK
2026-02-24 02:05:29,724 - simplemcp8_streamable - INFO - Processing JSON-RPC request: method=notifications/initialized, id=None
INFO:     127.0.0.1:35192 - "POST /mcp HTTP/1.1" 200 OK
2026-02-24 02:05:29,733 - simplemcp8_streamable - INFO - Processing JSON-RPC request: method=tools/list, id=1
INFO:     127.0.0.1:35188 - "POST /mcp HTTP/1.1" 200 OK
```

## Features

- **Memory Efficient**: Run multiple MCP tools in a single process using asyncio
- **Zero Modification**: Existing MCP tools work without any code changes
- **Automatic Discovery**: Automatically discover MCP tools from configured directories
- **Flexible Port Management**: Manual or automatic port allocation with conflict detection
- **CLI Interface**: Easy-to-use command-line interface
- **Error Handling**: Best-effort approach - continues with other tools if one fails
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### List Available Tools

```bash
python launchmcp.py --list-tools
```

### Launch Specific Tools

```bash
python launchmcp.py webmcp oraclemcp convertermcp ragmcp
```

### Launch All Discovered Tools

```bash
python launchmcp.py
```

### Use Custom Configuration

```bash
python launchmcp.py --config custom_config.json webmcp oraclemcp
```

### Verbose Mode

```bash
python launchmcp.py --verbose webmcp oraclemcp
```

### Dry Run (Preview Without Starting)

```bash
python launchmcp.py --dry-run webmcp oraclemcp
```

## Configuration

The launcher uses a JSON configuration file (`config.json`) for settings. You can override configuration values using environment variables or CLI arguments.

### Configuration Options

```json
{
  "toolDirectories": [
    "/path/to/tools1",
    "/path/to/tools2"
  ],
  "portAllocation": {
    "mode": "manual",
    "basePort": 8000,
    "portRange": [8000, 9000],
    "ports": {
      "tool1": 8000,
      "tool2": 8001,
      "tool3": 8002
    }
  },
  "server": {
    "host": "0.0.0.0",
    "logLevel": "info"
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": null
  },
  "errorHandling": {
    "continueOnError": true,
    "failFast": false
  }
}
```

### Configuration Details

#### `toolDirectories`
List of directories to search for MCP tools. Each directory is scanned for Python files that export the required MCP objects.

#### `portAllocation.mode`
- `"manual"`: Use port assignments from the `ports` dictionary
- `"auto"`: Automatically allocate ports starting from `basePort`

#### `portAllocation.ports`
Dictionary mapping tool names to port numbers (used in manual mode).

#### `server.host`
Host address for all servers (default: `"0.0.0.0"`).

#### `server.logLevel`
Log level for Uvicorn servers (debug, info, warning, error, critical).

#### `logging.level`
Log level for the launcher itself.

#### `logging.file`
Optional path to a log file. If `null`, logs to console only.

#### `errorHandling.continueOnError`
If `true`, the launcher continues with other tools if one fails. If `false`, it stops immediately on error.

#### `errorHandling.failFast`
If `true`, the launcher fails fast on the first error.

### Environment Variables

You can override configuration using environment variables:

- `LAUNCHER_TOOL_DIRECTORIES`: Comma-separated list of tool directories
- `LAUNCHER_PORT_MODE`: Port allocation mode (auto/manual)
- `LAUNCHER_BASE_PORT`: Base port for auto allocation
- `LAUNCHER_PORT_RANGE`: Port range (e.g., "8000,9000")
- `LAUNCHER_SERVER_HOST`: Server host address
- `LAUNCHER_LOG_LEVEL`: Server log level
- `LAUNCHER_LOGGING_LEVEL`: Launcher log level
- `LAUNCHER_CONTINUE_ON_ERROR`: Continue on error (true/false)
- `LAUNCHER_FAIL_FAST`: Fail fast on error (true/false)

Example:
```bash
export LAUNCHER_PORT_MODE=auto
export LAUNCHER_BASE_PORT=9000
python launchmcp.py webmcp oraclemcp
```

## CLI Arguments

```
usage: launchmcp.py [-h] [--config CONFIG] [--list-tools] [--verbose] [--dry-run]
                    [--host HOST] [--log-level {debug,info,warning,error,critical}]
                    [tools ...]

positional arguments:
  tools                 Names of tools to launch (if not specified, launches all
                        discovered tools)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (default: config.json)
  --list-tools          List all available MCP tools and exit
  --verbose             Enable verbose logging
  --dry-run             Preview actions without actually starting servers
  --host HOST           Override server host address
  --log-level {debug,info,warning,error,critical}
                        Override log level
```

## MCP Tool Requirements

For a Python module to be recognized as a valid MCP tool, it must export the following objects:

1. `server`: An instance of `mcp.server.lowlevel.Server`
2. `app`: An instance of `starlette.applications.Starlette`
3. `sse_transport`: An instance of `mcp.server.sse.SseServerTransport`

Example:
```python
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount

# Create MCP server
server = Server("my_tool")

# Create SSE transport
sse_transport = SseServerTransport("/messages/")

# Create Starlette app
app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse_transport.handle_post_message),
])
```

## Available MCP Tools

The launcher currently supports the following MCP tools:

### webmcp (Port 8001)
A web search and URL fetch MCP server that provides:
- **brave_search_web**: Enhanced web search using Brave Search with language support and metadata
- **brave_search_api**: Enhanced web search using Brave Search API with structured results
- **google_search_api**: Google Search API using SerpAPI with comprehensive results
- **fetch_url**: Enhanced web reader that fetches and processes web content with pagination, caching, and content filtering
- **post_url**: HTTP POST request tool for sending data to URLs with JSON payload support

**Documentation**: [`tools/webmcp/README.md`](tools/webmcp/README.md)

### oraclemcp (Port 8000)
An Oracle database MCP server that provides:
- Database query execution and schema introspection
- SQL optimization with AI assistance
- Explain plan analysis
- Pro*C coding rules reference

**Documentation**: [`tools/oraclemcp/README.md`](tools/oraclemcp/README.md)

### convertermcp (Port 8003)
A document conversion MCP server that provides:
- **convert_docx_to_text**: Convert Microsoft Word documents (.docx) to plain text
- Support for both local file paths and HTTP/HTTPS URLs
- SharePoint REST API fallback for Doc.aspx URLs
- Path security with configurable allowed root directories

**Documentation**: [`tools/convertermcp/README.md`](tools/convertermcp/README.md)

### ragmcp (Port 8004)
A RAG (Retrieval-Augmented Generation) and Code Indexing MCP server that provides:
- **search_code**: Semantic search using vector embeddings with natural language queries
- **search_code_sparse**: BM25-style lexical search for exact identifiers, table names, and function names
- **get_copilot_context**: Copilot context injection for GitHub Copilot integration
- **start_indexing**: Background indexing of code files into Qdrant vector database
- **check_indexing_progress**: Check indexing status and statistics
- **clear_index**: Clear indexed code collections
- **list_collections**: List all collections with statistics

**Documentation**: [`tools/ragmcp/README.md`](tools/ragmcp/README.md)

## Usage Examples

### Example 1: Launch All Tools

```bash
python launchmcp.py webmcp oraclemcp convertermcp ragmcp
```

Output:
```
============================================================
MCP Launcher Running
============================================================
  oracleMCP: http://0.0.0.0:8000
  webmcp: http://0.0.0.0:8001
  convertermcp: http://0.0.0.0:8003
  ragmcp: http://0.0.0.0:8004
============================================================
Press Ctrl+C to stop all servers
```

### Example 2: Launch Specific Tools

```bash
python launchmcp.py webmcp oraclemcp
```

Output:
```
============================================================
MCP Launcher Running
============================================================
  webmcp: http://0.0.0.0:8001
  oracleMCP: http://0.0.0.0:8000
============================================================
Press Ctrl+C to stop all servers
```

### Example 3: Auto Port Allocation

Configure `config.json` with:
```json
{
  "portAllocation": {
    "mode": "auto",
    "basePort": 8000,
    "portRange": [8000, 9000]
  }
}
```

Then run:
```bash
python launchmcp.py webmcp oraclemcp convertermcp ragmcp
```

The launcher will automatically allocate ports 8000, 8001, 8003, 8004.

### Example 4: Custom Host and Log Level

```bash
python launchmcp.py --host 127.0.0.1 --log-level debug webmcp
```

## Troubleshooting

### Port Already in Use

If you see a port conflict error:
1. Check which process is using the port: `lsof -i :8000`
2. Kill the process or use a different port
3. Or configure the launcher to use a different port range

### Tool Not Found

If a tool is not discovered:
1. Verify the tool directory is in `toolDirectories`
2. Check that the tool exports the required objects (server, app, sse_transport)
3. Use `--list-tools` to see all discovered tools

### Import Errors

If you see import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that tool-specific dependencies are installed
3. Verify Python version compatibility (3.9+)

### Server Fails to Start

If a server fails to start:
1. Check the logs for detailed error messages
2. Verify the tool's configuration is correct
3. Ensure the tool's dependencies are met
4. Try running the tool standalone to isolate the issue

## Architecture

The launcher consists of the following components:

- **Tool Discovery**: Scans directories and loads MCP tool modules
- **Port Manager**: Allocates and manages ports for each tool
- **Server Manager**: Manages lifecycle of Uvicorn servers
- **Configuration**: Loads and validates configuration
- **Error Handling**: Provides custom exceptions and error recovery

For detailed architecture information, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Memory Efficiency

Running multiple MCP tools in a single process provides significant memory savings:

| Scenario | Separate Processes | Unified Launcher | Savings |
|----------|-------------------|------------------|---------|
| 3 Tools  | ~300 MB           | ~150 MB          | ~50%    |
| 5 Tools  | ~500 MB           | ~200 MB          | ~60%    |
| 10 Tools | ~1000 MB          | ~350 MB          | ~65%    |

## Contributing

To add a new MCP tool:

1. Create your tool following the MCP tool pattern
2. Export `server`, `app`, and `sse_transport` objects
3. Place the tool file in a configured directory
4. Run `python launchmcp.py --list-tools` to verify discovery
5. Launch with `python launchmcp.py your_tool_name`

## License

This project is part of the MCP tools ecosystem.

## Support

For issues, questions, or contributions, please refer to the MCP documentation.
