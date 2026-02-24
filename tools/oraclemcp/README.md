# Oracle MCP Server

A Model Context Protocol (MCP) server that provides access to Oracle databases with query execution, schema introspection, and AI-powered SQL optimization.

## Features

- Oracle database query execution
- Schema introspection with detailed table information
- SQL optimization with AI assistance
- Explain plan analysis
- Pro*C coding rules reference
- Valid language codes lookup
- **Support for both SSE and Streamable HTTP transports**

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env` file or your shell:
```bash
export USERID="your_username/your_password"
export DB_HOST="your_db_host"
export DB_PORT="1521"
export DB_SERVICE_NAME="your_service_name"
export AI_API_KEY="your_openai_api_key"  # For SQL optimization feature
```

## Configuration

### Required Configuration

- `USERID` - Oracle database credentials in format `username/password`
- `DB_HOST` - Oracle database host
- `DB_PORT` - Oracle database port (default: 1521)
- `DB_SERVICE_NAME` - Oracle database service name

### Optional Configuration

- `AI_API_KEY` - OpenAI API key for SQL optimization feature (uses gpt-4.1)

## Usage

### Transport Options

The Oracle MCP server supports two transport types:

1. **SSE (Server-Sent Events)** - Original transport using Starlette
2. **Streamable HTTP** - New transport using FastAPI with JSON-RPC framing

### SSE Transport (Original)

#### Standalone Mode

```bash
python oracleMCP.py
```

#### With Unified Launcher

```bash
python launchmcp.py oraclemcp
```

#### VSCode Configuration (SSE)

```json
{
  "mcpServers": {
    "oraclemcp": {
      "type": "sse",
      "url": "http://localhost:8000/sse/",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
```

### Streamable HTTP Transport (New)

#### Standalone Mode

```bash
python oracleMCP_streamable.py
```

#### With Unified Launcher

```bash
python launchmcp.py oraclemcp --transport streamable-http
```

#### VSCode Configuration (Streamable HTTP)

```json
{
  "mcpServers": {
    "oraclemcp": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Content-Type": "application/json"
      },
      "framing": "newline-delimited"
    }
  }
}
```

### Migration from SSE to Streamable HTTP

To migrate from SSE to Streamable HTTP transport:

1. **Update your VSCode configuration** - Change the transport type from `"sse"` to `"streamable-http"` and update the URL from `"/sse/"` to `"/mcp"`
2. **Add framing configuration** - Add `"framing": "newline-delimited"` to your configuration
3. **Verify dependencies** - Ensure `fastapi>=0.104.0` is installed (included in requirements.txt)
4. **Update startup command** - Use `oracleMCP_streamable.py` instead of `oracleMCP.py`

See the [Migration Script](#migration-script) section for an automated migration helper.

### Command Line Options (Streamable HTTP)

```bash
python oracleMCP_streamable.py --help
```

Options:
- `--host` - Host to bind to (default: 0.0.0.0)
- `--port` - Port to bind to (default: 8000)
- `--log-level` - Log level: debug, info, warning, error (default: info)

## Available Tools

- `schemas` - Get schema information for specified tables including columns, constraints, and foreign keys
- `query` - Execute SQL SELECT queries and return results
- `execute_sql` - Execute SQL statements for INSERT, UPDATE, DELETE operations
- `get_valid_languages` - Get valid language codes from LANGUES table
- `list_user_tables_with_descriptions` - List all user tables and their functional descriptions
- `get_sql_optimization_rules` - Get the list of rules for SQL query optimization from optimization.json
- `explain_plan` - Send an EXPLAIN PLAN query to Oracle and return the execution plan
- `optimize_sql_with_ai` - Accept a SQL query and use AI (gpt-4.1) to suggest optimizations based on optimization rules
- `get_proc_rules` - Get the Pro*C coding rules from proc_rules.md

## Tool Details

### schemas
Get detailed schema information for one or more tables.

**Parameters:**
- `table_names` (required): Array of table names

**Returns:**
- Column names, data types, lengths, precision, scale, nullable status, defaults, and comments
- Constraint types and search conditions
- Foreign key relationships

### query
Execute a SQL SELECT query.

**Parameters:**
- `sql_query` (required): The SQL query to execute

**Returns:**
- Query results as an array of rows
- Oracle error details with error code, message, and offset if query fails

### execute_sql
Execute SQL statements for data modification (INSERT, UPDATE, DELETE).

**Parameters:**
- `sql_statement` (required): The SQL statement to execute

**Returns:**
- Success message if statement executes successfully
- Oracle error details if execution fails

### get_valid_languages
Get valid language codes from the LANGUES table.

**Parameters:**
- `limit` (optional): Maximum number of languages to return (default: 10)

**Returns:**
- Language codes and descriptions

### list_user_tables_with_descriptions
List all user tables with their functional descriptions.

**Returns:**
- Table names and their comments from user_tab_comments

### get_sql_optimization_rules
Get the list of rules for SQL query optimization.

**Returns:**
- JSON-formatted optimization rules from optimization.json

### explain_plan
Generate and return the execution plan for a SQL query.

**Parameters:**
- `sql_query` (required): The SQL query to explain

**Returns:**
- Formatted execution plan from DBMS_XPLAN.DISPLAY()

### optimize_sql_with_ai
Use AI (gpt-4.1) to suggest SQL query optimizations.

**Parameters:**
- `sql_query` (required): The SQL query to optimize
- `add_comments` (optional): Include optimization comments in response (default: true)
- `table_descriptions` (optional): JSON or text describing table structures for context

**Returns:**
- Optimized SQL query with or without explanatory comments

### get_proc_rules
Get the Pro*C coding rules reference.

**Returns:**
- Pro*C coding rules from proc_rules.md

## Dependencies

### Core Dependencies

- `mcp>=1.0.0` - MCP framework
- `anyio>=4.0.0` - Async I/O
- `click>=8.0.0` - CLI framework
- `starlette>=0.27.0` - ASGI framework (for SSE transport)
- `fastapi>=0.104.0` - FastAPI framework (for Streamable HTTP transport)
- `uvicorn>=0.27.0` - ASGI server
- `oracledb>=1.0.0` - Oracle database client
- `httpx>=0.27.0` - HTTP client (for AI API)
- `python-dotenv>=1.0.0` - Environment variable loading
- `openai>=1.0.0` - OpenAI API client (for SQL optimization)

## Migration Script

A helper script is available to assist with migrating from SSE to Streamable HTTP transport:

```bash
./migrate_to_streamable_http.sh
```

This script will:
1. Check if FastAPI is installed
2. Display the current SSE configuration
3. Show the new Streamable HTTP configuration
4. Provide instructions for updating your VSCode settings
5. Offer to create a backup of your current configuration

### Manual Migration Steps

If you prefer to migrate manually:

1. **Stop the SSE server** (if running)
2. **Update VSCode configuration**:
   ```json
   {
     "mcpServers": {
       "oraclemcp": {
         "type": "streamable-http",
         "url": "http://localhost:8000/mcp",
         "headers": {
           "Content-Type": "application/json"
         },
         "framing": "newline-delimited"
       }
     }
   }
   ```
3. **Start the Streamable HTTP server**:
   ```bash
   python oracleMCP_streamable.py
   ```
4. **Verify the server is running**:
   ```bash
   curl http://localhost:8000/health
   ```

### Rollback Instructions

To rollback to SSE transport:

1. **Stop the Streamable HTTP server**
2. **Restore your VSCode configuration**:
   ```json
   {
     "mcpServers": {
       "oraclemcp": {
         "type": "sse",
         "url": "http://localhost:8000/sse/",
         "headers": {
           "Content-Type": "application/json"
         }
       }
     }
   }
   ```
3. **Start the SSE server**:
   ```bash
   python oracleMCP.py
   ```

## Troubleshooting

### Oracle Connection Issues

- Verify Oracle client is installed
- Check connection string format
- Ensure database is accessible
- Verify USERID format is `username/password`

### AI Optimization Issues

- Verify AI_API_KEY is set correctly
- Check that the API key has access to gpt-4.1 model
- Ensure network access to put.your.API.gateway.ai

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.9+ required)
- Verify Oracle client libraries are installed

### Streamable HTTP Transport Issues

#### Connection Refused
- Ensure the Streamable HTTP server is running: `python oracleMCP_streamable.py`
- Check that the port (default: 8000) is not already in use
- Verify the URL in VSCode configuration matches the server endpoint

#### Framing Errors
- Ensure `"framing": "newline-delimited"` is included in your VSCode configuration
- Check that the Content-Type header is set to `"application/json"`

#### Timeout Errors
- Increase the request timeout if executing long-running queries
- Check database connection settings and network latency
- Verify that the Oracle database is responsive

### SSE vs Streamable HTTP Comparison

| Feature | SSE Transport | Streamable HTTP Transport |
|---------|--------------|---------------------------|
| Protocol | Server-Sent Events | HTTP with JSON-RPC |
| Framing | Event-stream format | Newline-delimited JSON |
| Server Framework | Starlette | FastAPI |
| Endpoint | `/sse/` | `/mcp` |
| VSCode Config | `type: "sse"` | `type: "streamable-http"` |
| Status | Stable (Legacy) | Recommended (New) |

### Log Files

- **SSE Transport**: `oracleMCP.log`
- **Streamable HTTP Transport**: `oracleMCP_streamable.log`

Monitor logs in real-time:
```bash
tail -f oracleMCP_streamable.log
```

## License

This tool is part of the MCP tools ecosystem.
