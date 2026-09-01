# MCP Launcher

> My need is being able to launch MCP server tools in a flexible manner depending on my current needs.

**Status**: Tested as single user server (not yet tested as multi-user)

**Available Tools**: `oraclemcp` `webmcp` `simplemcp` `convertermcp` `ragmcp` `memorymcp`

A unified launcher system for running multiple MCP (Model Context Protocol) tools in a single Python process.
This reduces memory usage by approximately 50% compared to running each tool as a separate process.

There are two entry points: `python -m launcher` is the primary one; `python launchmcp.py` is the
legacy interface (still maintained, with extra flags like `--list-tools`, `--dry-run`, and `--verbose`
that the primary CLI doesn't expose).

---

## ✅ WORKING

| Tool | Port | Description |
|------|------|-------------|
| `oraclemcp` | 8000 | Oracle database tools to feed the LLM |
| `webmcp` | 8001 | Web search (Brave Search, Google API), URL fetch, HTTP POST |
| `simplemcp` | 8002 | Simple test tools (double, square, greet) |
| `convertermcp` | 8003 | Document conversion (DOCX to TXT) |
| `ragmcp` | 8004 | RAG-like codebase indexing using local or API embeddings |
| `memorymcp` | 8005 | Persistent memory store with knowledge graph and semantic search |

---

## 📖 Example

```bash
# Terminal session
$ python launchmcp.py simplemcp
```

**Output:**
```
2026-02-24 02:05:29,013 - root - INFO - ============================================================
2026-02-24 02:05:29,013 - root - INFO - MCP Launcher Starting
2026-02-24 02:05:29,013 - root - INFO - ============================================================
2026-02-24 02:05:29,013 - root - INFO - Searching only in directories for requested tools: ['path/to/supreme-mcp-tools/tools/simplemcp']
2026-02-24 02:05:29,013 - launcher.tool_discovery - INFO - Searching for MCP tools in: path/to/supreme-mcp-tools/tools/simplemcp
2026-02-24 02:05:29,591 - simplemcp - INFO - Starting simplemcp FastMCP server (transport: streamable-http)
2026-02-24 02:05:29,592 - launcher.tool_discovery - INFO - Discovered tool: simplemcp from path/to/supreme-mcp-tools/tools/simplemcp/simplemcp_fastmcp.py
2026-02-24 02:05:29,592 - root - INFO - Discovered 1 MCP tools: ['simplemcp']
2026-02-24 02:05:29,592 - launcher.port_manager - INFO - Allocated port 8002 for tool simplemcp
2026-02-24 02:05:29,592 - root - INFO - Allocated ports: {'simplemcp': 8002}
2026-02-24 02:05:29,592 - root - INFO - Starting 1 servers...
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Starting server for simplemcp on port 8002
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Server for simplemcp starting on port 8002
2026-02-24 02:05:29,592 - root - INFO - Server for simplemcp started on port 8002
2026-02-24 02:05:29,592 - launcher.server_manager - INFO - Running server for simplemcp on port 8002
INFO:     Started server process [2850649]
INFO:     Waiting for application startup.
2026-02-24 02:05:29,606 - simplemcp_streamable - INFO - SimpleMCP8 Streamable HTTP server starting up...
2026-02-24 02:05:29,606 - root - INFO - Successfully started 1/1 servers

============================================================
MCP Launcher Running
============================================================
  simplemcp: http://0.0.0.0:8002
============================================================
Press Ctrl+C to stop all servers

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
2026-02-24 02:05:29,702 - simplemcp_streamable - INFO - Processing JSON-RPC request: method=initialize, id=0
INFO:     127.0.0.1:35188 - "POST /mcp HTTP/1.1" 200 OK
2026-02-24 02:05:29,724 - simplemcp_streamable - INFO - Processing JSON-RPC request: method=notifications/initialized, id=None
INFO:     127.0.0.1:35192 - "POST /mcp HTTP/1.1" 200 OK
2026-02-24 02:05:29,733 - simplemcp_streamable - INFO - Processing JSON-RPC request: method=tools/list, id=1
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

### Launch with the Primary Entry Point

```bash
python -m launcher --tools webmcp,simplemcp    # start specific tools (comma-separated)
python -m launcher                             # start all discovered tools
python -m launcher --management-port 8200     # management API (started by default)
python -m launcher --no-management            # skip the central management server
python -m launcher --debug                    # DEBUG logging
```

The `python -m launcher` CLI takes a comma-separated `--tools` list; the legacy entry point below
takes tool names as positional arguments and has additional flags
(`--list-tools`, `--dry-run`, `--verbose`, `--no-sync-tools`, `--health-check-logs`, `--health-check-interval`).

### List Available Tools (legacy entry point)

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

## Ports and Management

Port ranges and tool assignments live in `config/ports.json`:

| Range | Purpose |
|-------|---------|
| 8000-8099 | MCP tool endpoints (mcp) |
| 8100-8199 | Tool management servers (mgmt) |
| 8200-8299 | System (central management: 8200) |
| 8300-8399 | Metrics (metrics_server: 8300) |
| 8400-8499 | UI (management_ui: 8400) |

Tool ports: oraclemcp 8000, webmcp 8001, simplemcp 8002, convertermcp 8003, ragmcp 8004, memorymcp 8005.

The launcher starts the central management API (port 8200) by default — disable with `--no-management`.
The NiceGUI management UI is a separate process: `python -m mcp_ui` (port 8400).

## Configuration

The launcher uses a JSON configuration file (`config/launcher_config.json` is the checked-in default;
pass a custom file with `--config`). Port ranges and tool→port assignments live in `config/ports.json`.
You can override configuration values using environment variables or CLI arguments.

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

Legacy entry point (`python launchmcp.py --help`):

```
usage: launchmcp.py [-h] [--config CONFIG] [--list-tools] [--verbose]
                    [--dry-run] [--host HOST]
                    [--log-level {debug,info,warning,error,critical}]
                    [--no-management] [--management-port MANAGEMENT_PORT]
                    [--no-sync-tools] [--transport {streamable-http,sse}]
                    [--health-check-logs {enable,disable,errors-only}]
                    [--health-check-interval HEALTH_CHECK_INTERVAL]
                    [tools ...]

Launch multiple MCP tools in a single process

positional arguments:
  tools                 Names of tools to launch (if not specified, launches
                        all discovered tools)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (default: config.json)
  --list-tools          List all available MCP tools and exit
  --verbose             Enable verbose logging
  --dry-run             Preview actions without actually starting servers
  --host HOST           Override server host address
  --log-level {debug,info,warning,error,critical}
                        Override log level
  --no-management       Disable the centralized management API server (enabled
                        by default, port from ports.json)
  --management-port MANAGEMENT_PORT
                        Port for the management server (default: from
                        ports.json)
  --no-sync-tools       Disable automatic tools config sync after server
                        startup
  --transport {streamable-http,sse}
                        Transport protocol for all tools (default: streamable-
                        http; sse is legacy compatibility for outdated
                        harnesses)
  --health-check-logs {enable,disable,errors-only}
                        Health check logging mode (default: enable)
  --health-check-interval HEALTH_CHECK_INTERVAL
                        Health check interval in seconds (default: 30)
```

Run `python -m launcher --help` for the primary entry point's flags
(`--tools`, `--tools-dir`, `--api-key`, `--debug`, ...).

## MCP Tool Requirements

For a Python module to be recognized as a valid MCP tool, place it at `tools/<name>/<name>_fastmcp.py`
(the launcher's discovery scans top-level files in each tool directory and derives the tool name from
the `_fastmcp` suffix). The module must export an `app` object — either a FastAPI app or (the
standard path) a Starlette app built via the shared server factory:

```python
from tools.shared.server_factory import create_fastmcp_server, get_transport_app

mcp = create_fastmcp_server("my_tool")   # registers @mcp.tool() functions
app = get_transport_app(mcp)             # multi-transport app (default)
```

By default the app serves every client dialect simultaneously from one FastMCP instance —
`/mcp` (streamable HTTP, stateful), `/mcp-stateless` (streamable HTTP, stateless), and
`/sse` + `/messages` (legacy SSE). Client selection is by URL, nothing to configure.
Single-transport pinning remains available as an escape hatch via `--transport`,
the `MCP_TRANSPORT` env var, or the `"transport"` key in `tools/<name>/config.json`.

## Available MCP Tools

The launcher currently supports the following six MCP tools:

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
- **search**: Unified code search with automatic detection of collection capabilities (dense/sparse/hybrid)
- **search_code**: Semantic search using vector embeddings with natural language queries
- **search_code_sparse**: BM25-style lexical search for exact identifiers, table names, and function names
- **get_copilot_context**: Copilot context injection for GitHub Copilot integration
- **index_code**: Index code files into Qdrant for semantic search
- **start_indexing**: Background indexing of code files into Qdrant vector database
- **check_indexing_progress**: Check indexing status and statistics
- **clear_index**: Clear indexed code collections
- **list_collections**: List all collections with statistics

**Documentation**: [`tools/ragmcp/README.md`](tools/ragmcp/README.md)

### memorymcp (Port 8005)
A persistent memory MCP server that provides:
- **upsertMemory / queryMemory / getMemory / deleteMemory**: Store and retrieve memories with semantic search, recency weighting, and PII redaction
- **textToGraph / textToSmartGraph**: Convert structured text into LLM-friendly knowledge graphs
- **createMemoryEdge / getMemoryGraph / exportGraphAsMarkdown**: Knowledge-graph linking, neighborhood queries, and export
- **getMetaDecisions / mergeDuplicates / decayOrExpire**: Curated architectural-decision lookup, dedup, and TTL cleanup
- Pluggable SQL + vector storage backends (Postgres, Turso/libSQL, Qdrant, pgvector)

**Documentation**: [`tools/memorymcp/BACKENDS.md`](tools/memorymcp/BACKENDS.md) (backend architecture and migration)

## Usage Examples

### Example 1: Launch All Tools

```bash
python launchmcp.py oraclemcp webmcp simplemcp convertermcp ragmcp memorymcp
```

Or with the primary entry point, just run `python -m launcher` (no args = all discovered tools).

Output:
```
============================================================
MCP Launcher Running
============================================================
  oraclemcp: http://0.0.0.0:8000
  webmcp: http://0.0.0.0:8001
  simplemcp: http://0.0.0.0:8002
  convertermcp: http://0.0.0.0:8003
  ragmcp: http://0.0.0.0:8004
  memorymcp: http://0.0.0.0:8005
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
  oraclemcp: http://0.0.0.0:8000
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

By default (no custom config), tool ports come from the fixed assignments in `config/ports.json`.

### Example 4: Custom Host and Log Level

```bash
python launchmcp.py --host 127.0.0.1 --log-level debug webmcp
```

## Monitoring

The launcher includes a built-in monitoring system for collecting metrics.

### Enable Monitoring

A `config/monitoring_config.json` is already checked in with monitoring enabled. To customize it:

1. Copy the example config (if starting fresh):
   ```bash
   cp config/monitoring_config.example.json config/monitoring_config.json
   ```
2. Edit `config/monitoring_config.json` and set `"enabled": true`

### Available Metrics

- **Request metrics**: HTTP request count, duration, status codes
- **Tool execution metrics**: Tool call count, success/error rates, duration
- **Server metrics**: Server health and uptime

### Metrics Endpoints

When monitoring is enabled:
- `GET /metrics` - Prometheus-formatted metrics
- `GET /metrics/health` - Health status
- `GET /metrics/stats` - Basic statistics

### Configuration

See [`config/monitoring_config.example.json`](config/monitoring_config.example.json) for all available options.

## Troubleshooting

### Port Already in Use

If you see a port conflict error:
1. Check which process is using the port: `lsof -i :8000`
2. Kill the process or use a different port
3. Or configure the launcher to use a different port range

### Tool Not Found

If a tool is not discovered:
1. Verify the tool directory is in `toolDirectories`
2. Check that the tool exports an `app` object (see MCP Tool Requirements above)
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

1. Create your tool following the MCP tool pattern (see MCP Tool Requirements above)
2. Export an `app` built via `create_fastmcp_server()` + `get_transport_app()`
3. Place it at `tools/<name>/<name>_fastmcp.py` (the `_fastmcp` suffix is what discovery looks for)
4. Run `python launchmcp.py --list-tools` to verify discovery
5. Launch with `python -m launcher --tools your_tool_name` (or `python launchmcp.py your_tool_name`)

## License

This project is part of the MCP tools ecosystem.

## Support

For issues, questions, or contributions, please refer to the MCP documentation.
