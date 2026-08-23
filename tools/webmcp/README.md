# Web MCP Server

A powerful MCP (Model Context Protocol) server that provides web search and URL fetch capabilities. Supports both SSE (Server-Sent Events) and Streamable HTTP transports for maximum compatibility.

## Search provider candidates — cheaper/free (dig in later)

Status quo: `BRAVE_SEARCH_API_KEY` + `SERPAPI_API_KEY` are already configured in `.env`
(`brave_search_api`, `google_search_api` are live). `brave_search_web` is the only
keyless fallback (HTML scraping — brittle; excluded from live sweeps since 2026-08-23).

Candidates worth an account, cheapest-first:

| Provider | Why | Price ballpark |
|---|---|---|
| **Tavily** | Built for LLM agents: clean JSON + content extraction, relevance-tuned | free ~1k calls/mo |
| **Serper.dev** | Same Google SERP data as SerpAPI at a fraction of the price | 2.5k free, then ~$0.3–1.0 /1k |
| **Exa** | Semantic/neural search (finds by meaning, not keywords) | free tier (~$10 credits/mo) |
| **Kagi** | Highest result quality, no SEO spam | paid only (~$25/10k) |
| Perplexity Sonar | search + synthesized answers (links not primary) | usage-based |

Notes:
- SerpAPI (current `google_search_api`) is the most expensive per call here;
  Serper.dev returns comparable Google data far cheaper — first swap candidate.
- Bing Web Search API was retired by Microsoft in 2025 — don't bother.
- DuckDuckGo has no official API; wrapper libraries exist but they're scraping,
  i.e. the same fragility class as `brave_search_web`.

Wiring recipe when a key exists (~30 lines): add `<PROVIDER>_SEARCH_API_KEY` to
`.env` + `config.json` `environment_variables`; add a tool function in
`webmcp_fastmcp.py` following the `brave_search_api` pattern; register it in
`config.json` `tools` + the mcp-live-tool-test skill expectations; restart launcher.
Candidate for the same change: remove `brave_search_web`.

## Features

- **brave_search_web**: Enhanced web search using Brave Search with language support and metadata (web scraping)
- **brave_search_api**: Enhanced web search using Brave Search API with structured results (requires API key)
- **google_search_api**: Google Search API using SerpAPI with comprehensive results (requires API key)
- **fetch_url**: Enhanced web reader that fetches and processes web content with pagination, caching, and content filtering
- **post_url**: HTTP POST request tool for sending data to URLs with JSON payload support

## Features

- **brave_search_web**: Enhanced web search using Brave Search with language support and metadata (web scraping)
- **brave_search_api**: Enhanced web search using Brave Search API with structured results (requires API key)
- **google_search_api**: Google Search API using SerpAPI with comprehensive results (requires API key)
- **fetch_url**: Enhanced web reader that fetches and processes web content with pagination, caching, and content filtering
- **post_url**: HTTP POST request tool for sending data to URLs with JSON payload support

## Installation

1. Activate your Python virtual environment:
   ```bash
   c:\DEV\python_env\Scripts\activate.bat
   ```

2. Install dependencies:
   ```bash
   cd c:\DEV\test\mcphome
   pip install -r requirements.txt
   ```

## Running the Server

### SSE Transport (Original)

The server will start on `http://0.0.0.0:8001` with SSE endpoint at `/sse`

#### Option 1: Using the startup script (recommended)
```bash
start_webmcp.bat
```

#### Option 2: Direct Python command
```bash
python webmcp.py
```

### Streamable HTTP Transport (Recommended)

The Streamable HTTP server will start on `http://0.0.0.0:8002` with endpoint at `/mcp`

#### Option 1: Using the startup script (if available)
```bash
start_webmcp_streamable.bat
```

#### Option 2: Direct Python command
```bash
python webmcp_streamable.py
```

#### Command-line options:
```bash
python webmcp_streamable.py --host 0.0.0.0 --port 8002 --log-level info
```

## VSCode Configuration

### SSE Transport (Original)

Add this to your VSCode `settings.json`:

```json
{
  "mcpServers": {
    "webmcp": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

### Streamable HTTP Transport (Recommended)

Add this to your VSCode `settings.json`:

```json
{
  "mcpServers": {
    "webmcp": {
      "type": "streamable-http",
      "url": "http://localhost:8002/mcp",
      "headers": {
        "Content-Type": "application/json"
      },
      "framing": "newline-delimited"
    }
  }
}
```

**Note:** The Streamable HTTP transport provides better performance, proper JSON-RPC 2.0 compliance, and is the recommended transport for modern MCP clients.

## VSCode Configuration

Add this to your VSCode `settings.json`:

```json
{
  "mcpServers": {
    "webmcp": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

## Tools

### brave_search_web

Enhanced web search using Brave Search with language support and metadata (web scraping).

**Parameters:**
- `query` (required): The search query
- `count` (optional): Number of results to return (default: 10)
- `timeout` (optional): Search timeout in seconds (default: 30.0)
- `language` (optional): Search language code (default: "en")

**Example:**
```
brave_search_web(query="Python MCP server tutorial", count=5, language="en")
```

### brave_search_api

Enhanced web search using Brave Search API with structured results. Requires `BRAVE_SEARCH_API_KEY` in `.env` file.

**Parameters:**
- `query` (required): The search query
- `count` (optional): Number of results to return (1-20, default: 10)
- `timeout` (optional): Search timeout in seconds (default: 30.0)
- `language` (optional): Search language code (default: "en")
- `country` (optional): Country code for search (default: "US")
- `text_decorations` (optional): Whether to include text decorations (default: True)
- `fresh` (optional): Whether to prefer fresh results (default: False)
- `use_post` (optional): Use POST method instead of GET (default: False)
- `llm_mode` (optional): Return concise JSON-like structure optimized for LLM consumption (default: False)

**Example:**
```
brave_search_api(query="Python MCP server tutorial", count=5, llm_mode=true)
```

### google_search_api

Google Search API using SerpAPI with comprehensive results. Requires `SERPAPI_API_KEY` in `.env` file.

**Parameters:**
- `query` (required): The search query
- `engine` (optional): Search engine to use (default: "google")
- `google_domain` (optional): Google domain to use (default: "google.com")
- `hl` (optional): Language code for search results (default: "en")
- `gl` (optional): Country code for search results (default: "us")
- `location` (optional): Location for localized search results
- `start` (optional): Pagination offset (default: 0)
- `num` (optional): Number of results to return (1-100, default: 10)
- `safe` (optional): Safe search setting: "active" or "off" (default: "active")
- `device` (optional): Device type: "desktop", "mobile", or "tablet"
- `llm_mode` (optional): Return concise JSON-like structure optimized for LLM consumption (default: False)

**Example:**
```
google_search_api(query="Python MCP server tutorial", num=5, llm_mode=true)
```

### fetch_url

Enhanced web reader that fetches and processes web content with pagination, caching, and content filtering. Supports auto-detection of response type (json/html/text).

**Parameters:**
- `url` (required): The URL to fetch and process
- `timeout` (optional): Request timeout in seconds (default: 30.0, range: 1-300)
- `headers` (optional): Custom HTTP headers
- `follow_redirects` (optional): Whether to follow redirects (default: True)
- `format` (optional): Output format (markdown/text/raw, default: markdown)
- `response_type` (optional): Response type handling (default: "auto"). Options: "auto", "json", "html", "text"
- `max_length` (optional): Maximum number of characters to return (default: 50000, range: 100-1000000)
- `start_index` (optional): Start content from this character index for pagination (default: 0)
- `include_images` (optional): Whether to include images in output (default: True)
- `include_tables` (optional): Whether to include tables in output (default: True)
- `include_links` (optional): Whether to include links in output (default: True)
- `max_size` (optional): Maximum download size in bytes (default: 10485760 = 10MB, range: 1024-52428800)
- `use_cache` (optional): Enable caching for repeated URL requests (default: False)
- `cache_ttl` (optional): Cache time-to-live in seconds (default: 3600 = 1 hour, range: 60-86400)
- `retainImages` (optional): Legacy parameter: Whether to retain images in output (default: True)
- `noCache` (optional): Legacy parameter: Disable caching (default: False)

**Example:**
```
fetch_url(url="https://api.example.com/data.json", response_type="auto", format="markdown")
```

### post_url

Sends an HTTP POST request to the specified URL with optional data and headers. Supports JSON payloads and returns the response as text.

**Parameters:**
- `url` (required): The URL to POST to
- `data` (optional): The POST body (as a string, e.g. JSON or form data)
- `headers` (optional): Optional HTTP headers as a JSON object
- `timeout` (optional): Request timeout in seconds (default: 30.0)

**Example:**
```
post_url(url="https://api.example.com/endpoint", data='{"key": "value"}', headers={"Content-Type": "application/json"})
```

## Architecture

### SSE Transport (Original)

The SSE version uses:
- **mcp.server.lowlevel.Server**: Core MCP server implementation
- **SseServerTransport**: SSE (Server-Sent Events) transport for HTTP
- **Starlette**: ASGI web framework for HTTP handling
- **Uvicorn**: ASGI server to run the application
- **httpx**: Async HTTP client for web requests

### Streamable HTTP Transport

The server (single transport — SSE was removed 2026-08) uses:
- **FastMCP**: server framework (`tools/shared/server_factory.py`)
- **FastAPI**: Modern ASGI web framework for HTTP handling
- **Uvicorn**: ASGI server to run the application
- **httpx**: Async HTTP client for web requests
- **JSON-RPC 2.0**: Proper JSON-RPC framing over streamable HTTP at `/mcp`

## Migration from SSE to Streamable HTTP

### Why Migrate?

- **Better Performance**: Streamable HTTP uses chunked transfer encoding and proper JSON-RPC framing
- **Modern Standard**: Streamable HTTP is the recommended transport for MCP going forward
- **Improved Compatibility**: Works with modern MCP clients that support the latest protocol
- **Simpler Architecture**: No separate SSE endpoint and message endpoint; single `/mcp` endpoint

### Migration Steps

1. **Ensure Streamable HTTP version is running**:
   ```bash
   python webmcp_streamable.py
   ```
   The server will start on `http://0.0.0.0:8002/mcp`

2. **Update VSCode settings.json**:
   Change from SSE configuration:
   ```json
   {
     "mcpServers": {
       "webmcp": {
         "type": "sse",
         "url": "http://localhost:8001/sse"
       }
     }
   }
   ```
   
   To Streamable HTTP configuration:
   ```json
   {
     "mcpServers": {
       "webmcp": {
         "type": "streamable-http",
         "url": "http://localhost:8002/mcp",
         "headers": {
           "Content-Type": "application/json"
         },
         "framing": "newline-delimited"
       }
     }
   }
   ```

3. **Restart VSCode** or reload the window to apply the new configuration.

4. **Verify the migration**:
   - Check that the server is running: `http://localhost:8002/health`
   - Check server info: `http://localhost:8002/`
   - Test tools to ensure they work correctly

### Rollback Instructions

If you need to rollback to SSE:

1. Stop the Streamable HTTP server
2. Revert your `settings.json` to the SSE configuration
3. Start the SSE server: `python webmcp.py`
4. Restart VSCode

### Configuration Differences

| Aspect | SSE | Streamable HTTP |
|--------|-----|-----------------|
| Port | 8001 | 8002 |
| Endpoint | `/sse` (GET) + `/messages/` (POST) | `/mcp` (POST only) |
| Framing | SSE events | Newline-delimited JSON |
| Headers | Standard SSE | `Content-Type: application/json` |
| Session Management | Automatic via SSE | Optional session headers |

## Environment Variables

Create a `.env` file in the `webmcp` directory with the following optional API keys:

```bash
# Brave Search API key (for brave_search_api tool)
BRAVE_SEARCH_API_KEY=your_brave_api_key_here

# SerpAPI key (for google_search_api tool)
SERPAPI_API_KEY=your_serpapi_key_here
```

**Note:** These keys are optional. The `brave_search_web` tool works without an API key by scraping Brave Search, and `post_url` and `fetch_url` don't require API keys.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**: Change the port using `--port` argument:
   ```bash
   python webmcp_streamable.py --port 8003
   ```

3. **API key errors**: Verify your `.env` file is in the correct location and contains valid API keys.

4. **Connection refused**: Ensure the server is running and the port is correct in your VSCode settings.

5. **Streamable HTTP not working**: Make sure your MCP client supports Streamable HTTP transport (GitHub Copilot and Roo Code both support it).

### Logging

Both servers support log level configuration:
```bash
python webmcp_streamable.py --log-level debug
```

Log levels: `debug`, `info`, `warning`, `error`

## Development

### Testing Tools

You can test the tools using `curl` or any HTTP client:

```bash
# Test initialize
curl -X POST http://localhost:8002/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}},"id":1}'

# List tools
curl -X POST http://localhost:8002/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":2}'

# Call a tool (example: fetch_url)
curl -X POST http://localhost:8002/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"fetch_url","arguments":{"url":"https://example.com"}},"id":3}'
```

### Adding New Tools

To add a new tool:

1. Add the tool handler method in the `WebMCPStreamableHttp` class
2. Add the tool definition in `_handle_tools_list()`
3. Add the tool routing in `_handle_tool_call()`
4. Follow the pattern of existing tools for proper error handling and response formatting

## License

This project is open source. Please check the repository for license details.
