"""
databasemcp MCP tools — all @mcp.tool() registrations + FEF setup_extensions.

Ported verbatim (behavior-preserving) from the former oraclemcp monolith;
P2/P3 add the connection-registry tools and per-dialect dispatch.
"""
import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Any

import oracledb
import openai

from core import (
    mcp, logger, metrics, SCRIPT_DIR, TOOL_NAME,
    FEF_V3_AVAILABLE, ToolExtensionManager, register_common_extensions,
    setup_tool_extensions, Extension, ExtensionType,
)
import connections
from connections import REGISTRY
from dialects import DIALECTS, get_dialect


# ============================================================================
# Connection Registry Tools (P2)
# ============================================================================

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


@mcp.tool()
async def connect_database(name: str, db_type: str, params: dict) -> str:
    """Connect to a database and register it under a name.

    db_type: oracle | postgres | libsql. params per type:
    - oracle: user, password, host, port, service_name
    - postgres: host, dbname, user, password (optional: port, connect_timeout)
    - libsql: url ("file:/path.db", "file::memory:", or "libsql://..." with
      optional auth_token). NOTE: a file: URL to a nonexistent path creates
      an empty database (SQLite semantics).
    The first connection becomes the active one for query/execute_sql/etc.
    Params are never echoed back.
    """
    start_time = time.perf_counter()
    if not _NAME_RE.match(name or ""):
        _timing_update(start_time, "connect_database", False)
        return "Invalid connection name: use 1-32 chars of letters, digits, '_' or '-'"
    try:
        dialect = get_dialect(db_type)
    except ValueError as e:
        _timing_update(start_time, "connect_database", False)
        return f"Error: {e}"
    missing = [p for p in dialect.REQUIRED_PARAMS if not params.get(p)]
    if missing:
        _timing_update(start_time, "connect_database", False)
        return f"Missing required params for {db_type}: {', '.join(missing)}"

    def _connect():
        REGISTRY.connect(name, db_type, params)

    try:
        await asyncio.to_thread(_connect)
    except Exception as e:
        _timing_update(start_time, "connect_database", False)
        logger.error(f"Connect failed for '{name}' ({db_type}): {type(e).__name__}")
        return f"Connect failed: {str(e).splitlines()[0] if str(e) else type(e).__name__}"
    _timing_update(start_time, "connect_database", True)
    active = REGISTRY._active
    return f"Connected '{name}' ({db_type}). Active: '{active}'"


@mcp.tool()
async def disconnect_database(name: str) -> str:
    """Disconnect and remove a named database connection."""
    start_time = time.perf_counter()
    try:
        result = await asyncio.to_thread(REGISTRY.disconnect, name)
    except LookupError as e:
        _timing_update(start_time, "disconnect_database", False)
        return f"Error: {e}"
    _timing_update(start_time, "disconnect_database", True)
    if result.startswith("Connection '") and "busy" in result:
        _timing_update(start_time, "disconnect_database", False)
        return result
    return f"Disconnected '{name}'. Active now: {result}"


@mcp.tool()
async def list_connections() -> str:
    """List all registered database connections (never shows credentials)."""
    start_time = time.perf_counter()
    conns = REGISTRY.list()
    if not conns:
        _timing_update(start_time, "list_connections", True)
        return "none — use connect_database(name, db_type, params)"
    lines = []
    for c in conns:
        active = " *ACTIVE*" if c["active"] else ""
        last_err = f" last_error={c['last_error']}" if c["last_error"] else ""
        lines.append(
            f"- {c['name']} ({c['dialect']}, {c['state']}, cached={c['cached_tables']}){active}{last_err}"
        )
    _timing_update(start_time, "list_connections", True)
    return "\n".join(lines)


@mcp.tool()
async def use_database(name: str) -> str:
    """Switch the active database connection used by query/execute_sql/etc."""
    start_time = time.perf_counter()
    try:
        entry = await asyncio.to_thread(REGISTRY.set_active, name)
    except LookupError as e:
        _timing_update(start_time, "use_database", False)
        return f"Error: {e}"
    _timing_update(start_time, "use_database", True)
    return f"Active connection: '{entry.name}' ({entry.dialect})"


# ============================================================================
# FEF V3 State + Extension Handlers
# ============================================================================

fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def _record_request(tool_name: str, success: bool, elapsed_ms: float) -> None:
    """FEF per-request metrics (factored from the monolith's repeated blocks)."""
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name=tool_name,
            success=success, duration_ms=elapsed_ms
        )


def _timing_update(start_time: float, tool_name: str, success: bool) -> float:
    """Shared metrics bookkeeping; returns elapsed ms."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if success:
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
    else:
        metrics["query_errors"] += 1
    _record_request(tool_name, success, elapsed_ms)
    return elapsed_ms


def get_query_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get query statistics."""
    avg_query_time = (
        metrics["total_query_time_ms"] / metrics["query_count"]
        if metrics["query_count"] > 0 else 0.0
    )
    return {
        "total_queries": metrics["query_count"],
        "query_errors": metrics["query_errors"],
        "avg_query_time_ms": round(avg_query_time, 2),
        "schema_lookups": metrics["schema_lookups"]
    }


def get_connection_pool_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get connection pool statistics."""
    return {
        "active_connections": metrics["connection_count"],
        "connection_errors": metrics["connection_errors"],
        "config": connections.get_pool_config()
    }


def get_schema_cache_stats(params: dict[str, Any]) -> dict[str, Any]:
    """Data source: Get schema cache statistics."""
    return {
        "cached_tables": len(connections.table_columns_cache),
        "schema_lookups": metrics["schema_lookups"]
    }


def reset_connections(params: dict[str, Any]) -> dict[str, Any]:
    """Action: Reset database connections."""
    metrics["connection_count"] = 0
    metrics["connection_errors"] = 0
    logger.info("[databasemcp] Connection counters reset")
    return {
        "success": True,
        "message": "Connection counters have been reset"
    }


def clear_cache(params: dict[str, Any]) -> dict[str, Any]:
    """Action: Clear schema cache."""
    with connections._db_lock:
        connections.table_columns_cache = {}
        connections.schema_cache = {}
    logger.info("[databasemcp] Schema cache cleared")
    return {
        "success": True,
        "message": "Schema cache cleared"
    }


# ============================================================================
# MCP Tools (ported verbatim; error rendering identical)
# ============================================================================

def _format_db_error(error_msg: dict) -> str:
    if error_msg.get("code"):
        formatted = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
        if error_msg.get("offset"):
            formatted += f"\nAt position: {error_msg['offset']}"
    else:
        formatted = f"Error: {error_msg['message']}"
    return formatted


@mcp.tool()
async def get_schemas(table_name: str | None = None) -> str:
    """Get schema information for a specified table."""
    start_time = time.perf_counter()
    if not table_name:
        _timing_update(start_time, "get_schemas", False)
        return "Error: table_name is required"

    table_name = table_name.upper()
    schema = connections.fetch_schema_from_cache(table_name)

    if not schema or not isinstance(schema, dict):
        _timing_update(start_time, "get_schemas", False)
        return f"Invalid schema for table '{table_name}'."

    _timing_update(start_time, "get_schemas", True)
    return str({
        "columns": schema["columns"],
        "constraints": schema["constraints"]
    })


@mcp.tool()
async def get_valid_languages() -> str:
    """Get valid language codes from LANGUES table."""
    start_time = time.perf_counter()
    sql = """
        SELECT LANCODE, LANLIBC, LANLIBL, LANUSED
        FROM LANGUES
        WHERE ROWNUM <= 10
        ORDER BY LANCODE
    """
    result = connections.execute_query(sql)
    if not result["success"]:
        _timing_update(start_time, "get_valid_languages", False)
        return _format_db_error(result["error"])
    _timing_update(start_time, "get_valid_languages", True)
    return str(result["data"])


@mcp.tool()
async def query(sql: str, max_rows: int = 100) -> str:
    """Executes a SQL query and returns the results."""
    start_time = time.perf_counter()
    if not sql:
        _timing_update(start_time, "query", False)
        return "Error: sql query is required"

    result = connections.execute_query(sql)

    if not result["success"]:
        _timing_update(start_time, "query", False)
        return _format_db_error(result["error"])

    data = result["data"]
    if len(data) > max_rows:
        data = data[:max_rows]
        _timing_update(start_time, "query", True)
        return f"{str(data)}\n\n[Results truncated - showing {max_rows} of {len(result['data'])} rows]"
    _timing_update(start_time, "query", True)
    return str(data)


@mcp.tool()
async def execute_sql(sql: str) -> str:
    """Executes an SQL statement for INSERT or UPDATE operations."""
    start_time = time.perf_counter()
    if not sql:
        _timing_update(start_time, "execute_sql", False)
        return "Error: sql statement is required"

    try:
        logger.info(f"[SQL] Executing statement: {sql[:200]}{'...' if len(sql) > 200 else ''}")
        conn = connections.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        _timing_update(start_time, "execute_sql", True)
        return "SQL statement executed successfully."
    except oracledb.DatabaseError as e:
        _timing_update(start_time, "execute_sql", False)
        error_details = connections.format_oracle_error(e)
        logger.error(f"Oracle error executing SQL statement: {error_details}")
        formatted_error = f"Oracle Error {error_details.get('code', 'Unknown')}: {error_details['message']}"
        if error_details.get('offset'):
            formatted_error += f"\nAt position: {error_details['offset']}"
        return formatted_error
    except Exception as e:
        _timing_update(start_time, "execute_sql", False)
        logger.error(f"Error executing SQL statement: {e}")
        return f"Error executing SQL statement: {str(e)}"


@mcp.tool()
async def list_user_tables_with_descriptions() -> str:
    """Lists all user tables and their functional descriptions."""
    start_time = time.perf_counter()
    try:
        conn = connections.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, NVL(comments, 'No description available')
            FROM user_tab_comments
            ORDER BY table_name
        """)
        results = cursor.fetchall()
        table_list = [
            {"table_name": row[0], "description": row[1]}
            for row in results
        ]
        _timing_update(start_time, "list_user_tables_with_descriptions", True)
        return str(table_list)
    except Exception as e:
        _timing_update(start_time, "list_user_tables_with_descriptions", False)
        logger.error(f"Error fetching user tables with descriptions: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_sql_optimization_rules() -> str:
    """Returns the list of rules for optimization of SQL queries from optimization.json."""
    start_time = time.perf_counter()
    try:
        optimization_path = SCRIPT_DIR / "optimization.json"
        with Path(optimization_path).open("r", encoding="utf-8") as f:
            rules = json.load(f)
        _timing_update(start_time, "get_sql_optimization_rules", True)
        return json.dumps(rules, ensure_ascii=False, indent=2)
    except Exception as e:
        _timing_update(start_time, "get_sql_optimization_rules", False)
        logger.error(f"Error reading optimization.json: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def explain_plan(sql: str) -> str:
    """Sends an EXPLAIN PLAN query to Oracle and returns the execution plan for the provided SQL query."""
    start_time = time.perf_counter()
    if not sql:
        _timing_update(start_time, "explain_plan", False)
        return "Error: sql query is required"

    try:
        conn = connections.get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM PLAN_TABLE")
        except Exception:
            pass

        cursor.execute(f"EXPLAIN PLAN FOR {sql}")

        try:
            cursor.execute("SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY())")
            plan_rows = cursor.fetchall()
            plan_text = "\n".join(row[0] for row in plan_rows)
        except Exception:
            cursor.execute("SELECT * FROM PLAN_TABLE")
            plan_rows = cursor.fetchall()
            plan_text = str(plan_rows)
        _timing_update(start_time, "explain_plan", True)
        return plan_text
    except Exception as e:
        _timing_update(start_time, "explain_plan", False)
        logger.error(f"Error executing EXPLAIN PLAN: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def optimize_sql_with_ai(sql: str) -> str:
    """Accepts a SQL query, references optimization rules from optimization.json, and calls an AI to suggest or apply optimizations."""
    start_time = time.perf_counter()
    if not sql:
        _timing_update(start_time, "optimize_sql_with_ai", False)
        return "Error: sql query is required"

    try:
        def extract_table_names_from_sql(sql_query):
            pattern = r"(?:from|join|into|update|with)\s+([a-zA-Z0-9_]+)"
            return list(set(re.findall(pattern, sql_query, re.IGNORECASE)))

        table_names = extract_table_names_from_sql(sql)
        table_descriptions = None
        if table_names:
            try:
                schemas_result = {}
                for table_name in table_names:
                    schema = connections.fetch_schema_from_cache(table_name.upper())
                    if schema and isinstance(schema, dict):
                        schemas_result[table_name] = {
                            "columns": schema["columns"],
                            "constraints": schema["constraints"]
                        }
                if schemas_result:
                    table_descriptions = str(schemas_result)
            except Exception as e:
                logger.error(f"Error fetching table schemas for AI optimization: {e}")

        optimization_path = SCRIPT_DIR / "optimization.json"
        with Path(optimization_path).open("r", encoding="utf-8") as f:
            rules = json.load(f)

        prompt = "You are an expert SQL query optimizer."
        if table_descriptions:
            prompt += "\n\nHere are the relevant table structures and comments for context:\n"
            prompt += f"{table_descriptions}\n"
        prompt += (
            "\nGiven the following SQL query and a set of optimization rules, "
            "suggest improvements or rewrite the query to be as efficient as possible.\n\n"
            "Optimization Rules:\n"
            f"{json.dumps(rules, ensure_ascii=False, indent=2)}\n\n"
            "SQL Query:\n"
            f"{sql}\n\n"
            "Optimized SQL and/or suggestions (include comments explaining optimizations):"
        )

        api_key = os.getenv('AI_API_KEY')
        if not api_key or api_key == "put_your_api_key_here":
            _timing_update(start_time, "optimize_sql_with_ai", False)
            logger.error("AI_API_KEY environment variable not properly configured")
            return "Error: The AI optimization feature requires a valid API key. Please update the AI_API_KEY environment variable."

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://put.your.API.gateway.ai/"
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        _timing_update(start_time, "optimize_sql_with_ai", True)
        return response.choices[0].message.content
    except Exception as e:
        _timing_update(start_time, "optimize_sql_with_ai", False)
        logger.error(f"Error optimizing SQL with AI: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_proc_rules() -> str:
    """Returns the Pro*C coding rules from proc_rules.md."""
    start_time = time.perf_counter()
    try:
        proc_rules_path = SCRIPT_DIR / "proc_rules.md"
        with Path(proc_rules_path).open("r", encoding="utf-8") as f:
            rules = f.read()
        _timing_update(start_time, "get_proc_rules", True)
        return rules
    except Exception as e:
        _timing_update(start_time, "get_proc_rules", False)
        logger.error(f"Error reading proc_rules.md: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================

def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", "8100"))

    custom_extensions = [
        Extension(
            name="query_stats",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "total_queries": {"type": "integer"},
                        "query_errors": {"type": "integer"},
                        "avg_query_time_ms": {"type": "number"},
                        "schema_lookups": {"type": "integer"}
                    }
                }
            },
            handler=get_query_stats,
            metadata={"description": "Database query execution statistics", "category": "metrics"}
        ),
        Extension(
            name="connection_pool",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "active_connections": {"type": "integer"},
                        "connection_errors": {"type": "integer"},
                        "config": {"type": "object"}
                    }
                }
            },
            handler=get_connection_pool_stats,
            metadata={"description": "Connection pool statistics", "category": "metrics"}
        ),
        Extension(
            name="schema_cache",
            ext_type=ExtensionType.DATA_SOURCE,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "cached_tables": {"type": "integer"},
                        "schema_lookups": {"type": "integer"}
                    }
                }
            },
            handler=get_schema_cache_stats,
            metadata={"description": "Schema cache statistics", "category": "metrics"}
        ),
        Extension(
            name="reset_connections",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=reset_connections,
            metadata={"description": "Reset database connection counters", "category": "maintenance"}
        ),
        Extension(
            name="clear_cache",
            ext_type=ExtensionType.ACTION,
            schema={
                "input": {"type": "object", "properties": {}},
                "output": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"}
                    }
                }
            },
            handler=clear_cache,
            metadata={"description": "Clear schema cache", "category": "maintenance"}
        ),
    ]

    if registry is not None:
        fef_registry = registry
        fef_manager = ToolExtensionManager(TOOL_NAME)
        register_common_extensions(TOOL_NAME, fef_registry, fef_manager)
        for ext in custom_extensions:
            fef_registry.register(TOOL_NAME, ext)
        fef_http_server = None
        logger.info(f"[{TOOL_NAME}] FEF V3 registered with launcher's registry")
    else:
        fef_manager, fef_registry, fef_http_server = setup_tool_extensions(
            tool_name=TOOL_NAME,
            mgmt_port=mgmt_port,
            custom_extensions=custom_extensions
        )
        logger.info(f"[{TOOL_NAME}] FEF V3 standalone mode on port {mgmt_port}")

    fef_setup_done = True
