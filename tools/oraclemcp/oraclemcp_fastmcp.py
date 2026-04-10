#!/usr/bin/env python3
"""
OracleMCP Server - FastMCP Implementation
Provides Oracle database query and schema exploration capabilities using FastMCP.

Port allocation from ports.json only — no hardcoded ports.
FEF V3 integration for distributed tool management.
"""
import os
import sys
import logging
import time
import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from pathlib import Path

import oracledb
import openai

# ============================================================================
# Port Configuration (from ports.json only)
# ============================================================================

TOOL_NAME = "oraclemcp"

try:
    from launcher.launcher_config import load_ports_config
    ports_config = load_ports_config()
    MCP_PORT = int(os.environ.get(
        "MCP_PORT",
        ports_config["assignments"]["mcp"][TOOL_NAME]
    ))
    MGMT_PORT = int(os.environ.get(
        "MCP_MGMT_PORT",
        ports_config["assignments"]["mgmt"][TOOL_NAME]
    ))
except Exception as e:
    print(f"ERROR: Failed to load ports.json: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Logging
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "oraclemcp.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(TOOL_NAME)

# ============================================================================
# FEF V3 Integration
# ============================================================================

try:
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    from launcher.tool_extensions import Extension, ExtensionType
    FEF_V3_AVAILABLE = True
    logger.info("FEF V3 modules loaded successfully")
except ImportError as e:
    FEF_V3_AVAILABLE = False
    logger.warning(f"FEF V3 not available: {e}")

# ============================================================================
# Metrics
# ============================================================================

metrics = {
    "query_count": 0,
    "query_errors": 0,
    "total_query_time_ms": 0.0,
    "min_query_time_ms": float("inf"),
    "max_query_time_ms": 0.0,
    "connection_count": 0,
    "connection_errors": 0,
    "schema_lookups": 0,
}

# ============================================================================
# FEF V3 Data Sources
# ============================================================================

def get_query_stats(params: Dict[str, Any]) -> Dict[str, Any]:
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


def get_connection_pool_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """Data source: Get connection pool statistics."""
    return {
        "active_connections": metrics["connection_count"],
        "connection_errors": metrics["connection_errors"],
        "config": get_pool_config()
    }


def get_schema_cache_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """Data source: Get schema cache statistics."""
    return {
        "cached_tables": len(table_columns_cache) if 'table_columns_cache' in globals() else 0,
        "schema_lookups": metrics["schema_lookups"]
    }


def reset_connections(params: Dict[str, Any]) -> Dict[str, Any]:
    """Action: Reset database connections."""
    metrics["connection_count"] = 0
    metrics["connection_errors"] = 0
    logger.info("[oraclemcp] Connection counters reset")
    return {
        "success": True,
        "message": "Connection counters have been reset"
    }


def clear_cache(params: Dict[str, Any]) -> Dict[str, Any]:
    """Action: Clear schema cache."""
    global table_columns_cache, schema_cache
    table_columns_cache = {}
    schema_cache = {}
    logger.info("[oraclemcp] Schema cache cleared")
    return {
        "success": True,
        "message": "Schema cache cleared"
    }


# ============================================================================
# Connection Pool Configuration
# ============================================================================

def get_pool_config() -> dict:
    """Get pool config from env vars (hot-reload)."""
    return {
        "min_connections": int(os.environ.get("ORACLE_MIN_CONNECTIONS", "1")),
        "max_connections": int(os.environ.get("ORACLE_MAX_CONNECTIONS", "10")),
        "increment": 1,
        "query_timeout_seconds": int(os.environ.get("ORACLE_QUERY_TIMEOUT", "30")),
    }


# ============================================================================
# Database Connection Management
# ============================================================================

connection = None
schema_cache = {}
table_columns_cache = {}


def get_db_connection():
    """Get database connection with automatic reconnection."""
    global connection
    try:
        if connection is None:
            raise oracledb.DatabaseError("Connection is not established")
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
    except oracledb.DatabaseError:
        try:
            user_id = os.getenv('USERID')
            if not user_id:
                raise EnvironmentError("USERID environment variable not set")

            login, password = user_id.split('/')

            db_host = os.getenv('DB_HOST')
            db_port = int(os.getenv('DB_PORT') or 1521)
            db_service_name = os.getenv('DB_SERVICE_NAME')
            if not db_host or not db_port or not db_service_name:
                raise EnvironmentError("Database connection environment variables not set")

            dsn_tns = oracledb.makedsn(db_host, db_port, service_name=db_service_name)
            connection = oracledb.connect(user=login, password=password, dsn=dsn_tns)
            metrics["connection_count"] += 1
            logger.info("Database connection re-established successfully.")
        except Exception as e:
            logger.error(f"Error re-establishing database connection: {e}")
            metrics["connection_errors"] += 1
            connection = None
            raise
    return connection


def fetch_schema_from_cache(table_name):
    """Fetch schema for a table, using cache when possible."""
    global schema_cache
    logger.info(f"Querying schema for table: {table_name}")
    if table_name not in schema_cache:
        return "Table not found"

    if schema_cache[table_name] is None:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT utc.column_name, utc.data_type, utc.data_length, utc.data_precision, utc.data_scale, utc.nullable, utc.data_default, ucc.comments
                FROM user_tab_columns utc
                LEFT JOIN user_col_comments ucc
                ON utc.table_name = ucc.table_name AND utc.column_name = ucc.column_name
                WHERE utc.table_name = '{table_name}'
            """)
            columns = cursor.fetchall()

            cursor.execute(f"""
                SELECT cols.column_name, cons.constraint_type, cons.search_condition
                FROM user_constraints cons, user_cons_columns cols
                WHERE cols.table_name = '{table_name}'
                  AND cons.constraint_type IN ('P', 'R', 'C', 'U')
                  AND cons.constraint_name = cols.constraint_name
            """)
            constraints = cursor.fetchall()

            cursor.execute(f"""
                SELECT a.constraint_name, a.column_name, c_pk.table_name AS referenced_table, b.column_name AS referenced_column
                FROM user_cons_columns a
                JOIN user_constraints c ON a.constraint_name = c.constraint_name
                JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
                JOIN user_cons_columns b ON c_pk.constraint_name = b.constraint_name AND a.position = b.position
                WHERE c.constraint_type = 'R' AND a.table_name = '{table_name}'
            """)
            foreign_keys = cursor.fetchall()

            schema_cache[table_name] = {
                "columns": columns,
                "constraints": constraints,
                "foreign_keys": foreign_keys
            }
            metrics["schema_lookups"] += 1
            logger.info(f"Schema details for table {table_name} cached successfully.")
        except Exception as e:
            logger.error(f"Error fetching schema details for table {table_name}: {e}")
            connection = None
            return "Error fetching schema details"

    return schema_cache[table_name]


def format_oracle_error(e):
    """Format Oracle error details into a structured response."""
    try:
        if isinstance(e, oracledb.DatabaseError):
            error_obj = e.args[0]
            error_msg = str(error_obj)
            error_code = None
            if error_msg.startswith('ORA-'):
                error_code = error_msg[4:9]

            return {
                "error": "ORA_ERROR",
                "code": error_code,
                "message": error_msg,
                "offset": getattr(error_obj, 'offset', None)
            }
    except Exception as format_error:
        logger.error(f"Error formatting Oracle error: {format_error}")

    return {
        "error": "DB_ERROR",
        "code": None,
        "message": str(e)
    }


def execute_query(sql_query):
    """Execute a SQL query and return results or error details."""
    global metrics
    start_time = time.time()
    logger.info(f"[SQL] Executing query: {sql_query[:200]}{'...' if len(sql_query) > 200 else ''}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        metrics["query_count"] += 1
        return {"success": True, "data": results}
    except oracledb.DatabaseError as e:
        metrics["query_errors"] += 1
        error_details = format_oracle_error(e)
        logger.error(f"Oracle error executing query: {error_details}")
        return {"success": False, "error": error_details}
    except Exception as e:
        metrics["query_errors"] += 1
        logger.error(f"Error executing query: {e}")
        return {"success": False, "error": {"error": "EXECUTION_ERROR", "message": str(e)}}
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        metrics["total_query_time_ms"] += elapsed_ms


# ============================================================================
# FastMCP Instance
# ============================================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    TOOL_NAME,
    sse_path="/sse",
    streamable_http_path="/mcp",
)


@mcp.tool()
async def get_schemas(table_name: str | None = None) -> str:
    """Get schema information for a specified table."""
    start_time = time.perf_counter()
    if not table_name:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_schemas",
                success=False, duration_ms=elapsed_ms
            )
        return "Error: table_name is required"

    table_name = table_name.upper()
    schema = fetch_schema_from_cache(table_name)

    if not schema or not isinstance(schema, dict):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_schemas",
                success=False, duration_ms=elapsed_ms
            )
        return f"Invalid schema for table '{table_name}'."

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metrics["query_count"] += 1
    metrics["total_query_time_ms"] += elapsed_ms
    if elapsed_ms < metrics["min_query_time_ms"]:
        metrics["min_query_time_ms"] = elapsed_ms
    if elapsed_ms > metrics["max_query_time_ms"]:
        metrics["max_query_time_ms"] = elapsed_ms
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name="get_schemas",
            success=True, duration_ms=elapsed_ms
        )
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
    result = execute_query(sql)
    if not result["success"]:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_valid_languages",
                success=False, duration_ms=elapsed_ms
            )
        error_msg = result["error"]
        if error_msg.get("code"):
            formatted_error = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
            if error_msg.get("offset"):
                formatted_error += f"\nAt position: {error_msg['offset']}"
        else:
            formatted_error = f"Error: {error_msg['message']}"
        return formatted_error
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metrics["query_count"] += 1
    metrics["total_query_time_ms"] += elapsed_ms
    if elapsed_ms < metrics["min_query_time_ms"]:
        metrics["min_query_time_ms"] = elapsed_ms
    if elapsed_ms > metrics["max_query_time_ms"]:
        metrics["max_query_time_ms"] = elapsed_ms
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name="get_valid_languages",
            success=True, duration_ms=elapsed_ms
        )
    return str(result["data"])


@mcp.tool()
async def query(sql: str, max_rows: int = 100) -> str:
    """Executes a SQL query and returns the results."""
    start_time = time.perf_counter()
    if not sql:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="query",
                success=False, duration_ms=elapsed_ms
            )
        return "Error: sql query is required"

    result = execute_query(sql)

    if not result["success"]:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="query",
                success=False, duration_ms=elapsed_ms
            )
        error_msg = result["error"]
        if error_msg.get("code"):
            formatted_error = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
            if error_msg.get("offset"):
                formatted_error += f"\nAt position: {error_msg['offset']}"
        else:
            formatted_error = f"Error: {error_msg['message']}"
        return formatted_error

    data = result["data"]
    if len(data) > max_rows:
        data = data[:max_rows]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="query",
                success=True, duration_ms=elapsed_ms
            )
        return f"{str(data)}\n\n[Results truncated - showing {max_rows} of {len(result['data'])} rows]"
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    metrics["query_count"] += 1
    metrics["total_query_time_ms"] += elapsed_ms
    if elapsed_ms < metrics["min_query_time_ms"]:
        metrics["min_query_time_ms"] = elapsed_ms
    if elapsed_ms > metrics["max_query_time_ms"]:
        metrics["max_query_time_ms"] = elapsed_ms
    if fef_manager is not None:
        fef_manager.metrics.record_request(
            endpoint="tools/call", tool_name="query",
            success=True, duration_ms=elapsed_ms
        )
    return str(data)


@mcp.tool()
async def execute_sql(sql: str) -> str:
    """Executes an SQL statement for INSERT or UPDATE operations."""
    start_time = time.perf_counter()
    if not sql:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="execute_sql",
                success=False, duration_ms=elapsed_ms
            )
        return "Error: sql statement is required"

    try:
        logger.info(f"[SQL] Executing statement: {sql[:200]}{'...' if len(sql) > 200 else ''}")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="execute_sql",
                success=True, duration_ms=elapsed_ms
            )
        return "SQL statement executed successfully."
    except oracledb.DatabaseError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="execute_sql",
                success=False, duration_ms=elapsed_ms
            )
        error_details = format_oracle_error(e)
        logger.error(f"Oracle error executing SQL statement: {error_details}")
        formatted_error = f"Oracle Error {error_details.get('code', 'Unknown')}: {error_details['message']}"
        if error_details.get('offset'):
            formatted_error += f"\nAt position: {error_details['offset']}"
        return formatted_error
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="execute_sql",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error executing SQL statement: {e}")
        return f"Error executing SQL statement: {str(e)}"


@mcp.tool()
async def list_user_tables_with_descriptions() -> str:
    """Lists all user tables and their functional descriptions."""
    start_time = time.perf_counter()
    try:
        conn = get_db_connection()
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
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="list_user_tables_with_descriptions",
                success=True, duration_ms=elapsed_ms
            )
        return str(table_list)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="list_user_tables_with_descriptions",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error fetching user tables with descriptions: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_sql_optimization_rules() -> str:
    """Returns the list of rules for optimization of SQL queries from optimization.json."""
    start_time = time.perf_counter()
    try:
        optimization_path = SCRIPT_DIR / "optimization.json"
        with open(optimization_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_sql_optimization_rules",
                success=True, duration_ms=elapsed_ms
            )
        return json.dumps(rules, ensure_ascii=False, indent=2)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_sql_optimization_rules",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error reading optimization.json: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def explain_plan(sql: str) -> str:
    """Sends an EXPLAIN PLAN query to Oracle and returns the execution plan for the provided SQL query."""
    start_time = time.perf_counter()
    if not sql:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="explain_plan",
                success=False, duration_ms=elapsed_ms
            )
        return "Error: sql query is required"

    try:
        conn = get_db_connection()
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
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="explain_plan",
                success=True, duration_ms=elapsed_ms
            )
        return plan_text
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="explain_plan",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error executing EXPLAIN PLAN: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def optimize_sql_with_ai(sql: str) -> str:
    """Accepts a SQL query, references optimization rules from optimization.json, and calls an AI to suggest or apply optimizations."""
    start_time = time.perf_counter()
    if not sql:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="optimize_sql_with_ai",
                success=False, duration_ms=elapsed_ms
            )
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
                    schema = fetch_schema_from_cache(table_name.upper())
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
        with open(optimization_path, "r", encoding="utf-8") as f:
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
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            metrics["query_errors"] += 1
            if fef_manager is not None:
                fef_manager.metrics.record_request(
                    endpoint="tools/call", tool_name="optimize_sql_with_ai",
                    success=False, duration_ms=elapsed_ms
                )
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
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="optimize_sql_with_ai",
                success=True, duration_ms=elapsed_ms
            )
        return response.choices[0].message.content
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="optimize_sql_with_ai",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error optimizing SQL with AI: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def get_proc_rules() -> str:
    """Returns the Pro*C coding rules from proc_rules.md."""
    start_time = time.perf_counter()
    try:
        proc_rules_path = SCRIPT_DIR / "proc_rules.md"
        with open(proc_rules_path, "r", encoding="utf-8") as f:
            rules = f.read()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_count"] += 1
        metrics["total_query_time_ms"] += elapsed_ms
        if elapsed_ms < metrics["min_query_time_ms"]:
            metrics["min_query_time_ms"] = elapsed_ms
        if elapsed_ms > metrics["max_query_time_ms"]:
            metrics["max_query_time_ms"] = elapsed_ms
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_proc_rules",
                success=True, duration_ms=elapsed_ms
            )
        return rules
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics["query_errors"] += 1
        if fef_manager is not None:
            fef_manager.metrics.record_request(
                endpoint="tools/call", tool_name="get_proc_rules",
                success=False, duration_ms=elapsed_ms
            )
        logger.error(f"Error reading proc_rules.md: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# FEF V3 Extensions Setup
# ============================================================================

fef_manager = None
fef_registry = None
fef_http_server = None
fef_setup_done = False


def setup_extensions(registry=None) -> None:
    """Set up FEF V3 extensions. Called by launcher or on startup."""
    global fef_manager, fef_registry, fef_http_server, fef_setup_done

    if fef_setup_done:
        return

    if not FEF_V3_AVAILABLE:
        fef_setup_done = True
        return

    mgmt_port = int(os.environ.get("MCP_MGMT_PORT", MGMT_PORT))

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
            metadata={"description": "Oracle query execution statistics", "category": "metrics"}
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


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"{TOOL_NAME} FastMCP server starting on port {MCP_PORT}...")

    if not fef_setup_done:
        setup_extensions(registry=None)

    if FEF_V3_AVAILABLE and fef_http_server:
        try:
            await fef_http_server.start()
            logger.info("FEF V3 management server started")
        except Exception as e:
            logger.warning(f"Failed to start FEF V3 management server: {e}")

    yield

    logger.info(f"{TOOL_NAME} FastMCP server shutting down...")
    if fef_http_server:
        try:
            await fef_http_server.stop()
        except Exception:
            pass


# ============================================================================
# App Export
# FastMCP's streamable_http_app handles SSE at /sse and HTTP at /mcp
# internally via its own routing. We export it directly without wrapping in
# Starlette Mount, which would cause 307 redirects (Mount requires trailing
# slash in path patterns, breaking FastMCP's routes at /mcp).
# ============================================================================

app = mcp.streamable_http_app()


# ============================================================================
# Exports for Launcher
# ============================================================================

__all__ = ["app", "setup_extensions", "mcp"]


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {TOOL_NAME} FastMCP server")
    logger.info(f"  MCP port: {MCP_PORT}")
    logger.info(f"  SSE endpoint: http://0.0.0.0:{MCP_PORT}/sse")
    logger.info(f"  Streamable HTTP: http://0.0.0.0:{MCP_PORT}/mcp")
    if FEF_V3_AVAILABLE:
        logger.info(f"  FEF V3 mgmt: http://0.0.0.0:{MGMT_PORT}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MCP_PORT,
        log_level="info",
        lifespan="on",
    )
