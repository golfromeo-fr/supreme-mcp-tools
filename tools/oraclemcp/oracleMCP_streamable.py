#!/usr/bin/env python3
"""
Oracle MCP Server - Streamable HTTP Transport
Provides access to Oracle databases with query execution, schema introspection, and AI-powered SQL optimization.
"""
import sys
import os
import logging
import json
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict
from contextlib import asynccontextmanager

# Check for required dependencies before importing
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import uvicorn
    import oracledb
    import openai
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Add parent directories to path for importing StreamableHttpTransportBase
# The supreme-mcp-tools directory (parent of tools and launcher) needs to be in the path
# Script is at: tools/oraclemcp/oracleMCP_streamable.py
# supreme-mcp-tools is at: . (relative path)
supreme_mcp_tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if supreme_mcp_tools_dir not in sys.path:
    sys.path.insert(0, supreme_mcp_tools_dir)

try:
    from launcher.streamable_http.streamable_http_base import (
        StreamableHttpTransportBase,
        StreamableHttpConfig,
    )
except ImportError as e:
    print(f"ERROR: Cannot import StreamableHttpTransportBase: {e}", file=sys.stderr)
    print(f"Script location: {__file__}", file=sys.stderr)
    print(f"supreme_mcp_tools_dir: {supreme_mcp_tools_dir}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print("Please ensure the launcher/streamable_http module is available.", file=sys.stderr)
    print("Try running from the supreme-mcp-tools directory: python tools/oraclemcp/oracleMCP_streamable.py", file=sys.stderr)
    sys.exit(1)

from dotenv import load_dotenv

# Configure logging with file and console output
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "oracleMCP_streamable.log"

# Configure root logger first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("oracleMCP_streamable")

# Log startup information
logger.info("="*80)
logger.info("OracleMCP Streamable HTTP Server Starting")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Log file can be monitored with: tail -f {LOG_FILE}")
logger.info("="*80)

# Add protocol version compatibility
SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-11-25"]

# ============================================================================
# Database Connection and Schema Management
# ============================================================================

# Cache for schema information
schema_cache = {}

# Maintain a persistent database connection
connection = None

# Function to reset the connection
def reset_db_connection():
    global connection
    connection = None

def get_db_connection():
    """Get or establish a database connection with automatic reconnection."""
    global connection
    try:
        if connection is None:
            raise oracledb.DatabaseError("Connection is not established")
        # Test the connection by executing a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
    except oracledb.DatabaseError:
        try:
            user_id = os.getenv('USERID')
            if not user_id:
                raise EnvironmentError("USERID environment variable not set")

            login, password = user_id.split('/')

            db_host = os.getenv('DB_HOST')
            db_port = int(os.getenv('DB_PORT') or 1521)  # Default to 1521 if DB_PORT is not set
            db_service_name = os.getenv('DB_SERVICE_NAME')
            if not db_host or not db_port or not db_service_name:
                raise EnvironmentError("Database connection environment variables not set")

            dsn_tns = oracledb.makedsn(db_host, db_port, service_name=db_service_name)
            connection = oracledb.connect(user=login, password=password, dsn=dsn_tns)
            logger.info("Database connection re-established successfully.")
        except Exception as e:
            logger.error(f"Error re-establishing database connection: {e}")
            connection = None  # Ensure connection is reset on failure
            raise
    return connection

def cache_table_names(connection):
    """Cache table names on startup."""
    global schema_cache
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT table_name FROM user_tables")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        logger.info(f"Initial list of tables: {table_names}")  # Log the list of tables
        for table_name in table_names:
            schema_cache[table_name] = None  # Initialize with None to indicate uncached details
        logger.info("Table names cached successfully.")
    except Exception as e:
        logger.error(f"Error caching table names: {e}")
        raise

def fetch_schema_from_cache(table_name):
    """Fetch schema details from cache or database."""
    global schema_cache
    logger.info(f"Querying schema for table: {table_name}")  # Log the table name being queried
    if table_name not in schema_cache:
        return "Table not found"

    if schema_cache[table_name] is None:  # If details are not cached, fetch them
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Update the query to include column descriptions
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

            # Fetch foreign key details
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
                "foreign_keys": foreign_keys  # Add foreign key details to the schema
            }
            logger.info(f"Schema details for table {table_name} cached successfully.")
        except Exception as e:
            logger.error(f"Error fetching schema details for table {table_name}: {e}")
            reset_db_connection()  # Reset the connection on failure
            return "Error fetching schema details"

    return schema_cache[table_name]

def format_oracle_error(e):
    """Format Oracle error details into a structured response."""
    try:
        if isinstance(e, oracledb.DatabaseError):
            error_obj = e.args[0]
            # Extract error code from the message if available
            error_msg = str(error_obj)
            error_code = None
            if error_msg.startswith('ORA-'):
                error_code = error_msg[4:9]  # Extract the 5 digits after 'ORA-'

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
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return {"success": True, "data": results}
    except oracledb.DatabaseError as e:
        error_details = format_oracle_error(e)
        logger.error(f"Oracle error executing query: {error_details}")
        return {"success": False, "error": error_details}
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return {"success": False, "error": {"error": "EXECUTION_ERROR", "message": str(e)}}

# ============================================================================
# Oracle MCP Streamable HTTP Transport Implementation
# ============================================================================

class OracleMCPStreamableHttp(StreamableHttpTransportBase):
    """
    Oracle MCP server implementation using Streamable HTTP transport.
    
    This class provides Oracle database tools using the Streamable HTTP
    transport with JSON-RPC framing.
    """
    
    def __init__(self):
        """Initialize the Oracle MCP Streamable HTTP server."""
        config = StreamableHttpConfig(
            endpoint="/mcp",
            framing_format="newline-delimited",
            request_timeout=60.0,  # Longer timeout for database operations
        )
        super().__init__("oraclemcp", config)
        logger.info("Oracle MCP Streamable HTTP transport initialized")
    
    async def _handle_initialize(self, params, session):
        """Handle initialize request - only tools are supported."""
        protocol_version = params.get("protocolVersion", "2024-11-05")
        # Support both old and new protocol versions
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            logger.warning(f"Client sent unsupported protocol version: {protocol_version}, using 2024-11-05")
            protocol_version = "2024-11-05"
        
        # Return server capabilities - only tools are supported (matching original oracleMCP)
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {
                    "tools": {},  # Tools are supported
                    # resources and prompts are not included, indicating they're not supported
                },
                "serverInfo": {
                    "name": self.server_name,
                    "version": "1.0.0",
                },
            },
        }
    
    async def _handle_tools_list(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = [
            {
                "name": "schemas",
                "description": "Get schema information for specified tables",
                "inputSchema": {
                    "type": "object",
                    "required": ["table_names"],
                    "properties": {
                        "table_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of table names to get schemas for"
                        }
                    }
                }
            },
            {
                "name": "get_valid_languages",
                "description": "Get valid language codes from LANGUES table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of languages to return (optional)"
                        }
                    }
                }
            },
            {
                "name": "query",
                "description": "Executes a SQL query and returns the results.",
                "inputSchema": {
                    "type": "object",
                    "required": ["sql_query"],
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The SQL query to execute."
                        }
                    }
                }
            },
            {
                "name": "execute_sql",
                "description": "Executes an SQL statement for INSERT or UPDATE operations.",
                "inputSchema": {
                    "type": "object",
                    "required": ["sql_statement"],
                    "properties": {
                        "sql_statement": {
                            "type": "string",
                            "description": "The SQL statement to execute."
                        }
                    }
                }
            },
            {
                "name": "list_user_tables_with_descriptions",
                "description": "Lists all user tables and their functional descriptions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_sql_optimization_rules",
                "description": "Returns the list of rules for optimization of SQL queries from optimization.json.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "explain_plan",
                "description": "Sends an EXPLAIN PLAN query to Oracle and returns the execution plan for the provided SQL query.",
                "inputSchema": {
                    "type": "object",
                    "required": ["sql_query"],
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The SQL query to explain."
                        }
                    }
                }
            },
            {
                "name": "optimize_sql_with_ai",
                "description": "Accepts a SQL query, references optimization rules from optimization.json, and calls an AI (gpt-4.1) to suggest or apply optimizations.",
                "inputSchema": {
                    "type": "object",
                    "required": ["sql_query"],
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The SQL query to optimize."
                        },
                        "add_comments": {
                            "type": "boolean",
                            "description": "If true, include optimization comments in the AI response. If false, return only the optimized SQL.",
                            "default": True
                        },
                        "table_descriptions": {
                            "type": "string",
                            "description": "Optional. JSON or text describing table structures and comments to provide context to the AI."
                        }
                    }
                }
            },
            {
                "name": "get_proc_rules",
                "description": "Returns the Pro*C coding rules from proc_rules.md.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools,
            },
        }
    
    async def _handle_tool_call(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Tool called: {tool_name} with arguments: {arguments}")
        
        try:
            if tool_name == "schemas":
                async for response in self._handle_schemas_tool(arguments, request_id):
                    yield response
            elif tool_name == "query":
                async for response in self._handle_query_tool(arguments, request_id):
                    yield response
            elif tool_name == "execute_sql":
                async for response in self._handle_execute_sql_tool(arguments, request_id):
                    yield response
            elif tool_name == "get_valid_languages":
                async for response in self._handle_get_valid_languages_tool(arguments, request_id):
                    yield response
            elif tool_name == "list_user_tables_with_descriptions":
                async for response in self._handle_list_user_tables_with_descriptions_tool(arguments, request_id):
                    yield response
            elif tool_name == "get_sql_optimization_rules":
                async for response in self._handle_get_sql_optimization_rules_tool(arguments, request_id):
                    yield response
            elif tool_name == "explain_plan":
                async for response in self._handle_explain_plan_tool(arguments, request_id):
                    yield response
            elif tool_name == "optimize_sql_with_ai":
                async for response in self._handle_optimize_sql_with_ai_tool(arguments, request_id):
                    yield response
            elif tool_name == "get_proc_rules":
                async for response in self._handle_get_pro_c_rules_tool(arguments, request_id):
                    yield response
            else:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        except ValueError as e:
            logger.error(f"Value error in tool call '{tool_name}': {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": str(e)
                }
            }
        
        except oracledb.DatabaseError as e:
            logger.error(f"Oracle database error in tool call '{tool_name}': {e}")
            error_details = format_oracle_error(e)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Database error",
                    "data": error_details
                }
            }
        
        except Exception as e:
            logger.error(f"Error in tool call '{tool_name}': {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
    
    async def _handle_schemas_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for the schemas tool."""
        logger.info(f"Processing schemas tool for tables: {arguments.get('table_names')}")
        table_names = arguments.get("table_names")
        if not table_names or not isinstance(table_names, list):
            logger.error("Invalid argument: table_names must be a list of table names")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": "Missing or invalid argument: table_names must be a list of table names"
                }
            }
            return

        logger.debug(f"Fetching schemas for tables: {table_names}")
        results = {}
        for table_name in table_names:
            schema = fetch_schema_from_cache(table_name.upper())
            if not schema or not isinstance(schema, dict):
                logger.warning(f"Schema not found or invalid for table: {table_name}")
                results[table_name] = f"Invalid schema for table '{table_name}'."
            else:
                results[table_name] = {
                    "columns": schema["columns"],
                    "constraints": schema["constraints"]
                }
        logger.debug(f"Schemas fetched: {results}")
        logger.info(f"Schemas fetched successfully for tables: {table_names}")
        yield {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": str(results)
                    }
                ]
            }
        }
    
    async def _handle_query_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for the query tool."""
        logger.debug(f"Processing query tool with arguments: {arguments}")
        sql_query = arguments.get("sql_query")
        if not sql_query:
            logger.error("Missing required parameter: sql_query")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": "Missing argument: sql_query"
                }
            }
            return

        logger.info(f"Executing SQL query: {sql_query}")
        result = execute_query(sql_query)

        if not result["success"]:
            error_msg = result["error"]
            if error_msg.get("code"):
                formatted_error = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
                if error_msg.get("offset"):
                    formatted_error += f"\nAt position: {error_msg['offset']}"
            else:
                formatted_error = f"Error: {error_msg['message']}"
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_error
                        }
                    ]
                }
            }
        else:
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result["data"])
                        }
                    ]
                }
            }
    
    async def _handle_execute_sql_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for the execute_sql tool."""
        logger.debug(f"Processing execute_sql tool with arguments: {arguments}")
        sql_statement = arguments.get("sql_statement")
        if not sql_statement:
            logger.error("Missing required parameter: sql_statement")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": "Missing argument: sql_statement"
                }
            }
            return

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            conn.commit()
            logger.info(f"SQL statement executed and committed: {sql_statement}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "SQL statement executed successfully."
                        }
                    ]
                }
            }
        except oracledb.DatabaseError as e:
            error_details = format_oracle_error(e)
            logger.error(f"Oracle error executing SQL statement: {error_details}")
            formatted_error = f"Oracle Error {error_details.get('code', 'Unknown')}: {error_details['message']}"
            if error_details.get('offset'):
                formatted_error += f"\nAt position: {error_details['offset']}"
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_error
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error executing SQL statement: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing SQL statement: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_list_user_tables_with_descriptions_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for listing user tables and their functional descriptions."""
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
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(table_list)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error fetching user tables with descriptions: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_optimize_sql_with_ai_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for optimizing a SQL query using AI and optimization rules."""
        sql_query = arguments.get("sql_query")
        add_comments = arguments.get("add_comments", True)
        table_descriptions = arguments.get("table_descriptions", None)
        if not sql_query:
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": "Missing required parameter: sql_query"
                }
            }
            return
        
        try:
            # Step 1: Extract table names from the SQL query (simple regex for FROM/JOIN/CTE)
            def extract_table_names(sql):
                # This is a basic regex and may not cover all SQL edge cases
                pattern = r"(?:from|join|into|update|with)\s+([a-zA-Z0-9_]+)"
                return list(set(re.findall(pattern, sql, re.IGNORECASE)))
            table_names = extract_table_names(sql_query)
            # Step 2: If table_descriptions not provided, get schemas for extracted tables
            if not table_descriptions and table_names:
                try:
                    schemas_result = await self._handle_schemas_tool({"table_names": table_names}, request_id)
                    # Extract the text content from the result
                    table_descriptions = None
                    async for response in schemas_result:
                        if "result" in response:
                            table_descriptions = response["result"]["content"][0]["text"]
                except Exception as e:
                    logger.error(f"Error fetching table schemas for AI optimization: {e}")
                    table_descriptions = None
            # Load optimization rules
            optimization_path = SCRIPT_DIR / "optimization.json"
            with open(optimization_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            # Prepare prompt for the AI
            prompt = "You are an expert SQL query optimizer."
            if table_descriptions:
                prompt += "\n\nHere are the relevant table structures and comments for context:\n"
                prompt += f"{table_descriptions}\n"
            prompt += (
                "\nGiven the following SQL query and a set of optimization rules, "
                "suggest improvements or rewrite the query to be as efficient as possible. "
                "You may add any relevant optimizations on top of the provided rules.\n\n"
                "Optimization Rules:\n"
                f"{json.dumps(rules, ensure_ascii=False, indent=2)}\n\n"
                "SQL Query:\n"
                f"{sql_query}\n\n"
            )
            if add_comments:
                prompt += "Optimized SQL and/or suggestions (include comments explaining optimizations):"
            else:
                prompt += "Please provide only the final optimized SQL query as output, with no explanations or comments."
            # Call OpenAI API (gpt-4.1)
            api_key = os.getenv('AI_API_KEY')
            if not api_key or api_key == "put_your_api_key_here":
                logger.error("AI_API_KEY environment variable not properly configured")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Error: The AI optimization feature requires a valid API key. Please update the AI_API_KEY value in $HOME/.bashrc_srai with your Azure OpenAI API key, then restart your terminal."
                            }
                        ]
                    }
                }
                return

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
            ai_reply = response.choices[0].message.content
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": ai_reply
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error optimizing SQL with AI: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_explain_plan_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for sending EXPLAIN PLAN to Oracle and returning the execution plan."""
        sql_query = arguments.get("sql_query")
        if not sql_query:
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": "Missing required parameter: sql_query"
                }
            }
            return
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Clean up previous plans for this session
            try:
                cursor.execute("DELETE FROM PLAN_TABLE")
            except Exception:
                pass  # PLAN_TABLE may not exist or be empty
            # Run EXPLAIN PLAN
            cursor.execute(f"EXPLAIN PLAN FOR {sql_query}")
            # Try to get the plan using DBMS_XPLAN if available
            try:
                cursor.execute("SELECT PLAN_TABLE_OUTPUT FROM TABLE(DBMS_XPLAN.DISPLAY())")
                plan_rows = cursor.fetchall()
                plan_text = "\n".join(row[0] for row in plan_rows)
            except Exception as e:
                # Fallback: select from PLAN_TABLE directly
                cursor.execute("SELECT * FROM PLAN_TABLE")
                plan_rows = cursor.fetchall()
                plan_text = str(plan_rows)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": plan_text
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error executing EXPLAIN PLAN: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_get_sql_optimization_rules_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for returning SQL optimization rules from optimization.json."""
        try:
            optimization_path = SCRIPT_DIR / "optimization.json"
            with open(optimization_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(rules, ensure_ascii=False, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error reading optimization.json: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_get_pro_c_rules_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for returning Pro*C coding rules from proc_rules.md."""
        try:
            proc_rules_path = SCRIPT_DIR / "proc_rules.md"
            with open(proc_rules_path, "r", encoding="utf-8") as f:
                rules = f.read()
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": rules
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error reading proc_rules.md: {e}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ]
                }
            }
    
    async def _handle_get_valid_languages_tool(
        self,
        arguments: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for get_valid_languages tool."""
        limit = arguments.get("limit", 10)
        sql = f"""
            SELECT LANCODE, LANLIBC, LANLIBL, LANUSED
            FROM LANGUES
            WHERE ROWNUM <= {limit}
            ORDER BY LANCODE
        """
        result = execute_query(sql)
        if not result["success"]:
            error_msg = result["error"]
            if error_msg.get("code"):
                formatted_error = f"Oracle Error {error_msg['code']}: {error_msg['message']}"
                if error_msg.get("offset"):
                    formatted_error += f"\nAt position: {error_msg['offset']}"
            else:
                formatted_error = f"Error: {error_msg['message']}"
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_error
                        }
                    ]
                }
            }
        else:
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result["data"])
                        }
                    ]
                }
            }


# ============================================================================
# FastAPI Application
# ============================================================================

# Create the transport instance
transport = OracleMCPStreamableHttp()

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    logger.info("Oracle MCP Streamable HTTP server starting up...")
    # Initialize database connection and cache table names
    try:
        global connection
        connection = get_db_connection()
        cache_table_names(connection)
        logger.info("Database connection and schema cache initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    yield
    # Shutdown
    logger.info("Oracle MCP Streamable HTTP server shutting down...")
    await transport.cleanup_sessions()

# Create FastAPI application with lifespan
app = FastAPI(
    title="Oracle MCP Streamable HTTP Server",
    description="Oracle MCP tools using Streamable HTTP transport",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "oraclemcp",
        "version": "1.0.0",
        "transport": "streamable-http",
        "endpoint": "/mcp",
        "tools": [
            "schemas",
            "query",
            "execute_sql",
            "get_valid_languages",
            "list_user_tables_with_descriptions",
            "get_sql_optimization_rules",
            "explain_plan",
            "optimize_sql_with_ai",
            "get_proc_rules"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": transport.get_session_count()
    }


@app.post("/mcp")
async def handle_mcp_request(request: Request):
    """
    Handle Streamable HTTP MCP requests.
    
    This endpoint accepts JSON-RPC requests with proper framing and returns
    responses with the same framing format.
    """
    # Read request body
    body = await request.body()
    logger.debug(f"Received request body: {body}")
    
    # Extract headers
    headers = dict(request.headers)
    session_id = headers.get("Mcp-Session-Id")
    logger.debug(f"Request headers: {headers}, session_id: {session_id}")
    
    # Parse request data (expecting newline-delimited JSON)
    try:
        import json
        request_data = json.loads(body.decode("utf-8").strip())
        logger.info(f"Processing JSON-RPC request: method={request_data.get('method')}, id={request_data.get('id')}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse request: {e}")
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e)
                }
            }),
            status_code=400,
            media_type="application/json"
        )
    
    # Process the request
    async def response_generator():
        async for response in transport.handle_request(request_data, headers, session_id):
            # Format response based on framing configuration
            logger.debug(f"Generating response: {response}")
            if transport.config.framing_format == "newline-delimited":
                yield (json.dumps(response) + "\n").encode("utf-8")
            else:
                yield json.dumps(response).encode("utf-8")
    
    return StreamingResponse(
        response_generator(),
        media_type="application/json",
        headers={
            "Content-Type": "application/json",
            "X-MCP-Transport": "streamable-http",
            "X-MCP-Framing": transport.config.framing_format,
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Oracle MCP Streamable HTTP Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger("oracleMCP_streamable").setLevel(log_level)
    
    logger.info(f"Starting Oracle MCP Streamable HTTP Server on http://{args.host}:{args.port}")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode Configuration Example (Streamable HTTP):

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
"""
