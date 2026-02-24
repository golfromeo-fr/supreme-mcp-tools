# RAG MCP Server

RAG (Retrieval-Augmented Generation) and Code Indexing MCP server that provides semantic code search, sparse search, and code indexing capabilities using Qdrant vector database.

## Overview

This MCP server provides tools for:
- **Semantic Code Search**: Search indexed code using natural language queries with vector embeddings
- **Sparse Code Search**: BM25-style lexical search for exact identifiers, table names, and function names
- **Copilot Context Injection**: Get formatted code context for GitHub Copilot integration
- **Code Indexing**: Background indexing of code files into Qdrant vector database
- **Index Management**: List collections, check indexing progress, and clear indexes


## Transport

This MCP server supports **two transport types**:

### Streamable HTTP (Recommended)

The new Streamable HTTP transport provides:
- **Better proxy compatibility**: Works better with HTTP proxies and load balancers
- **Bidirectional communication**: Full duplex communication between client and server
- **JSON-RPC alignment**: Standard HTTP-based JSON-RPC implementation
- **Reduced complexity**: Single endpoint for both requests and responses

**Endpoint**: `http://0.0.0.0:8004/mcp`

**Framing**: Newline-delimited JSON-RPC messages

### SSE (Legacy)

The SSE transport is maintained for backward compatibility.

**Endpoint**: `http://0.0.0.0:8004/sse`

**Note**: SSE is deprecated and will be removed in a future version.

### Migration from SSE to Streamable HTTP

If you're currently using the SSE transport, migrating to Streamable HTTP is straightforward:

1. **Update your VSCode settings**:
   ```json
   {
     "mcpServers": {
       "ragmcp": {
         "type": "streamable-http",
         "url": "http://localhost:8004/mcp"
       }
     }
   }
   ```

2. **Update launcher configuration** (if using unified launcher):
   The launcher will automatically detect and use the Streamable HTTP transport.

3. **No code changes required**: All tool functionality remains the same.

See [`migrate_to_streamable_http.sh`](migrate_to_streamable_http.sh) for a helper script.

## Features

### 7 Available Tools

1. **search_code** - Semantic search using vector embeddings
   - Natural language queries (e.g., "functions that process customer orders")
   - Supports filtering by file type and function name
   - Returns relevant code chunks with relevance scores

2. **search_code_sparse** - Lexical search using sparse vectors
   - BM25-style search for exact matches
   - Works offline without API costs
   - Excellent for finding identifiers, table names, function names

3. **get_copilot_context** - Copilot context injection
   - Retrieves relevant code using sparse search
   - Formats as inline comments or markdown sidebar
   - Makes Copilot project-aware

4. **start_indexing** - Start background indexing
   - Indexes Pro*C, PL/SQL, Java, and other files
   - Supports sparse (BM25), dense (embeddings), or hybrid modes
   - Returns process ID for monitoring

5. **check_indexing_progress** - Check indexing status
   - Returns files processed, chunks indexed, errors
   - Shows recent log entries
   - Displays collection statistics

6. **clear_index** - Clear indexed code
   - Delete specific collections or all collections
   - Requires confirmation for safety
   - Useful for reindexing

7. **list_collections** - List all collections
   - Shows collection names and statistics
   - Displays vector dimensions and distance metrics
   - Useful for checking what's indexed

## Installation

### Prerequisites

- Python 3.8+
- Qdrant vector database running (default: `qdrant:6333`)
- Optional: Azure OpenAI API key for embeddings

### Install Dependencies

```bash
cd supreme-mcp-tools/tools/ragmcp
pip install -r requirements.txt
```

### Optional Dependencies

For full functionality, install these optional modules:

```bash
# For sparse search (BM25)
# Place sparse_vector_gen.py in the indexer directory

# For Copilot context injection
# Place copilot_context_injector.py in the tools directory

# For local embeddings (offline mode)
pip install sentence-transformers
```

## Configuration

### Environment Variables

Create a `.env` file in the ragmcp directory or set these environment variables:

```bash
# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Embedding Provider (azure or local)
EMBEDDING_PROVIDER=azure

# Local Embedding Model (if using local provider)
# Options:
#   'sbert-large' (1024d) - SBERT Large - English semantic similarity
#   'e5-large' (1024d) - Multilingual E5 Large - Multilingual semantic similarity (default)
#   'small' (384d) - BGE Small - Fast English embeddings
#   'base' (768d) - BGE Base - Balanced English embeddings
LOCAL_EMBEDDING_MODEL=e5-large

# Azure OpenAI Embeddings (if using azure provider)
AZURE_EMBEDDING_API_URL=https://put.your.API.gateway.ai/v1/embeddings
AZURE_EMBEDDING_MODEL=text-embedding-3-large
AI_API_KEY=your_api_key_here
```

### Configuration File

The [`config.json`](config.json) file contains detailed configuration options including:
- Qdrant connection settings
- Embedding provider settings
- Indexing parameters
- Search defaults
- Copilot integration settings

## Usage

### Standalone Execution

Run the server directly:

```bash
# Streamable HTTP (recommended)
python ragmcp_streamable.py

# SSE (legacy, for backward compatibility)
python ragmcp.py
```

The server will start on `http://0.0.0.0:8004`

**Note**: Use `ragmcp_streamable.py` for the Streamable HTTP transport.

### VSCode Integration

Add to your VSCode settings.json:

**Streamable HTTP (recommended)**:
```json
{
  "mcpServers": {
    "ragmcp": {
      "type": "streamable-http",
      "url": "http://localhost:8004/mcp"
    }
  }
}
```

**SSE (legacy)**:
```json
{
  "mcpServers": {
    "ragmcp": {
      "type": "sse",
      "url": "http://localhost:8004/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

### Unified Launcher Integration

The server can be launched using the unified MCP launcher:

```bash
cd supreme-mcp-tools
python launchmcp.py ragmcp
```

## Tool Usage Examples

### Semantic Code Search

```python
search_code(
    query="functions that process customer orders",
    limit=5,
    file_type="proc",
    collection_name="folder.to.index-database-code"
)
```

### Sparse Code Search

```python
search_code_sparse(
    query="STOMVT table",
    limit=5,
    file_type="proc"
)
```

### Get Copilot Context

```python
get_copilot_context(
    current_context="EXEC SQL INSERT INTO STOMVT VALUES (...)",
    format="comment",
    language="c",
    limit=3,
    max_lines=50
)
```

### Start Indexing

```python
start_indexing(
    workspace_root="/path/to/your/workspace",
    collection_name="your-database-code",
    mode="sparse"  # or "dense" or "hybrid"
)
```

### Check Indexing Progress

```python
check_indexing_progress(pid=12345)
```

### List Collections

```python
list_collections()
```

### Clear Index

```python
# Clear specific collection
clear_index(
    collection_name="folder.to.index-database-code",
    confirm=True
)

# Clear all collections
clear_index(
    collection_name="ALL",
    confirm=True
)
```

## Indexing Modes

### Sparse Mode (BM25)
- **Cost**: Free (offline)
- **Use Case**: Exact identifier matches, table names, function names
- **Best For**: Finding specific code patterns

### Dense Mode (Embeddings)
- **Cost**: API costs apply
- **Use Case**: Semantic understanding, natural language queries
- **Best For**: Finding code by functionality or intent

### Hybrid Mode
- **Cost**: API costs apply
- **Use Case**: Best of both worlds
- **Best For**: Maximum search quality

## Local Embedding Models

When using `EMBEDDING_PROVIDER=local`, you can choose from the following models:

### SBERT Large (`sbert-large`)
- **Model**: `stsb-bert-large`
- **Dimensions**: 1024
- **Description**: English semantic similarity model
- **Best For**: English code with semantic understanding
- **Performance**: Slower, higher quality

### Multilingual E5 Large (`e5-large`) - **Default**
- **Model**: `intfloat/multilingual-e5-large`
- **Dimensions**: 1024
- **Description**: Multilingual semantic similarity model
- **Best For**: Multilingual codebases, general purpose
- **Performance**: Balanced, supports multiple languages

### BGE Small (`small`)
- **Model**: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Description**: Fast English embeddings
- **Best For**: Quick searches, English-only code
- **Performance**: Fastest, good quality

### BGE Base (`base`)
- **Model**: `BAAI/bge-base-en-v1.5`
- **Dimensions**: 768
- **Description**: Balanced English embeddings
- **Best For**: Balanced performance and quality
- **Performance**: Balanced speed and quality

**Configuration**:
```bash
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=e5-large  # or sbert-large, small, base
```

## Architecture

### Components

1. **MCP Server**: Uses `mcp.server.lowlevel.Server` for MCP protocol
2. **Transport**: 
   - **Streamable HTTP**: JSON-RPC over HTTP with newline-delimited framing (recommended)
   - **SSE**: Server-Sent Events for real-time communication (legacy)
3. **Qdrant Client**: Vector database for storing and searching embeddings
4. **Embedding Generator**: Azure API or local sentence-transformers
5. **Indexer**: Background process for indexing code files

### Data Flow

```
Code Files → Indexer → Qdrant (vectors) → Search Tools → Results
```

## Logging

Logs are written to:
- **Console**: Standard output with INFO level
- **File**: `ragmcp.log` with DEBUG level

Monitor logs in real-time:
```bash
tail -f ragmcp.log
```

Indexing logs are written to:
- **File**: `logs/indexing_{collection_name}.log`

## Troubleshooting

### Qdrant Connection Failed

**Error**: `Qdrant client not initialized`

**Solution**:
1. Check Qdrant is running: `docker ps | grep qdrant`
2. Verify environment variables: `QDRANT_HOST`, `QDRANT_PORT`
3. Check network connectivity to Qdrant

### Sparse Search Not Available

**Error**: `Sparse vector search not available`

**Solution**:
1. Install `sparse_vector_gen.py` in the indexer directory
2. Check the module is importable

### Local Embeddings Not Available

**Error**: `Local embeddings module not available`

**Solution**:
```bash
pip install sentence-transformers
```

### API Key Not Set

**Error**: `AI_API_KEY not set`

**Solution**:
Set the `AI_API_KEY` environment variable in `.env` file or your shell profile.

### Indexing Process Not Starting

**Error**: `Indexer script not found`

**Solution**:
1. Verify the indexer script exists at the expected path
2. Check the path in the error message
3. Ensure the oraclemcp directory structure is correct

## Dependencies

### Required Dependencies

- `mcp` - Model Context Protocol framework
- `fastapi` - Modern ASGI framework (for Streamable HTTP)
- `starlette` - ASGI framework (for SSE)
- `uvicorn` - ASGI server
- `httpx` - Async HTTP client
- `qdrant-client` - Qdrant vector database client
- `python-dotenv` - Environment variable management
- `psutil` - Process management

### Optional Dependencies

- `sentence-transformers` - Local embedding generation
- `sparse_vector_gen` - Sparse vector generation (custom module)
- `copilot_context_injector` - Copilot context injection (custom module)

## File Structure

```
ragmcp/
├── ragmcp.py          # Main MCP server implementation
├── config.json         # Configuration file
├── requirements.txt     # Python dependencies
├── README.md          # This file
├── .env              # Environment variables (create this)
├── ragmcp.log        # Server logs (auto-generated)
└── logs/             # Indexing logs (auto-generated)
    └── indexing_*.log
```

## Development

### Adding New Tools

1. Define the tool in `@server.list_tools()` decorator
2. Implement the handler function
3. Add the handler to the `tool_router` dictionary
4. Update `config.json` with tool metadata

### Testing

Test the server locally:

```bash
# Start the Streamable HTTP server
python ragmcp_streamable.py

# In another terminal, test with curl
curl http://localhost:8004/mcp

# Or test the SSE server (legacy)
python ragmcp.py
curl http://localhost:8004/sse
```

## License

This tool is part of the MCP tools collection.

## Support

For issues and questions:
1. Check the logs: `tail -f ragmcp.log`
2. Review this README for common solutions
3. Verify configuration in `config.json`
4. Check environment variables are set correctly

## Related Tools

- **oracleMCP**: Oracle database tools and SQL execution
- **webmcp**: Web search and URL fetching
- **convertermcp**: Document conversion tools
