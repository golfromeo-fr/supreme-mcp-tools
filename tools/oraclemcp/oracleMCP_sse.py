#!/usr/bin/env python3

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import oracledb
import os
import logging
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import openai

# Configure logging with file and console output
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "oracleMCP.log"

# Configure root logger first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("oracleMCP")

# Log startup information
logger.info("="*80)
logger.info("OracleMCP Server Starting")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Log file can be monitored with: tail -f {LOG_FILE}")
logger.info("="*80)

# Create the server instance
server = None
sse_transport = None

def initialize_server():
    """Initialize the MCP server and transport"""
    global server, sse_transport
    try:
        logger.info("Creating MCP server instance")
        server = Server("oracleMCP")
        logger.info("Creating SSE transport")
        sse_transport = SseServerTransport("/messages/")
        return True
    except Exception as e:
        logger.error(f"Error initializing server: {e}")
        return False

# Initialize server components
if not initialize_server():
    raise RuntimeError("Failed to initialize MCP server")

# Define the SSE handler
logger.info("Setting up SSE handler...")
async def handle_sse(request):
    """Handle SSE connection and server initialization"""
    if not server or not sse_transport:
        logger.error("Server not properly initialized")
        raise RuntimeError("Server components not initialized")

    logger.debug("Setting up SSE connection")
    init_options = server.create_initialization_options()

    try:
        async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
            logger.debug("Starting server run with initialization")
            try:
                await server.run(streams[0], streams[1], init_options)
            except Exception as e:
                logger.error(f"Error during server run: {e}")
                raise
    except Exception as e:
        logger.error(f"Error in SSE connection: {e}")
        raise

# Cache for schema information
schema_cache = {}

# Modify schema cache to only cache table names on startup
logger.info("Caching table names on startup...")
# Add logging to display the initial list of tables
logger.info("Fetching initial list of tables...")
def cache_table_names(connection):
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

# Maintain a persistent database connection
logger.info("Setting up persistent database connection...")
connection = None

# Function to reset the connection
def reset_db_connection():
    global connection
    connection = None

# Modify the get_db_connection function to properly test and restart the connection if necessary
def get_db_connection():
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

# Update schema fetching to include foreign key details
# Add logging to display foreign key details

def fetch_schema_from_cache(table_name):
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
    """Format Oracle error details into a structured response"""
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
    """Execute a SQL query and return results or error details"""
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

# Define the list tools functionality
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="schemas",
            description="Get schema information for specified tables",
            inputSchema={
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
        ),
        types.Tool(
            name="get_valid_languages",
            description="Get valid language codes from LANGUES table",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of languages to return (optional)"
                    }
                }
            }
        ),
        types.Tool(
            name="query",
            description="Executes a SQL query and returns the results.",
            inputSchema={
                "type": "object",
                "required": ["sql_query"],
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL query to execute."
                    }
                }
            },
        ),
        types.Tool(
            name="execute_sql",
            description="Executes an SQL statement for INSERT or UPDATE operations.",
            inputSchema={
                "type": "object",
                "required": ["sql_statement"],
                "properties": {
                    "sql_statement": {
                        "type": "string",
                        "description": "The SQL statement to execute."
                    }
                }
            },
        ),
        types.Tool(
            name="list_user_tables_with_descriptions",
            description="Lists all user tables and their functional descriptions.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="get_sql_optimization_rules",
            description="Returns the list of rules for optimization of SQL queries from optimization.json.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="explain_plan",
            description="Sends an EXPLAIN PLAN query to Oracle and returns the execution plan for the provided SQL query.",
            inputSchema={
                "type": "object",
                "required": ["sql_query"],
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SQL query to explain."
                    }
                }
            }
        ),
        types.Tool(
            name="optimize_sql_with_ai",
            description="Accepts a SQL query, references optimization rules from optimization.json, and calls an AI (gpt-4.1) to suggest or apply optimizations.",
            inputSchema={
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
        ),
        types.Tool(
            name="get_proc_rules",
            description="Returns the Pro*C coding rules from proc_rules.md.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

# Main tool handler function - route to appropriate tool implementation
@server.call_tool()
async def tool_router(name: str, arguments: dict) -> list[types.TextContent]:
    """Central router function that directs tool calls to the appropriate handler"""
    logger.debug(f"tool_router received request for tool: '{name}' with arguments: {arguments}")

    handlers = {
        "schemas": handle_schemas_tool,
        "query": handle_query_tool,
        "execute_sql": handle_execute_sql_tool,
        "get_valid_languages": handle_get_valid_languages_tool,
        "list_user_tables_with_descriptions": handle_list_user_tables_with_descriptions_tool,
        "get_sql_optimization_rules": handle_get_sql_optimization_rules_tool,
        "explain_plan": handle_explain_plan_tool,
        "optimize_sql_with_ai": handle_optimize_sql_with_ai_tool,
        "get_proc_rules": handle_get_pro_c_rules_tool
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)

    logger.error(f"Unknown tool name: '{name}'")
    raise ValueError(f"Unknown tool: {name}")

# Handler for schemas tool
async def handle_schemas_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for the schemas tool"""
    logger.info(f"Processing schemas tool for tables: {arguments.get('table_names')}")
    table_names = arguments.get("table_names")
    if not table_names or not isinstance(table_names, list):
        logger.error("Invalid argument: table_names must be a list of table names")
        raise ValueError("Missing or invalid argument: table_names must be a list of table names")

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
    return [types.TextContent(type="text", text=str(results))]

# Handler for query tool
async def handle_query_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for the query tool"""
    logger.debug(f"Processing query tool with arguments: {arguments}")
    sql_query = arguments.get("sql_query")
    if not sql_query:
        logger.error("Missing required parameter: sql_query")
        raise ValueError("Missing argument: sql_query")

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
        return [types.TextContent(type="text", text=formatted_error)]

    return [types.TextContent(type="text", text=str(result["data"]))]

# Handler for execute_sql tool
async def handle_execute_sql_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for the execute_sql tool"""
    logger.debug(f"Processing execute_sql tool with arguments: {arguments}")
    sql_statement = arguments.get("sql_statement")
    if not sql_statement:
        logger.error("Missing required parameter: sql_statement")
        raise ValueError("Missing argument: sql_statement")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        conn.commit()
        logger.info(f"SQL statement executed and committed: {sql_statement}")
        return [types.TextContent(type="text", text="SQL statement executed successfully.")]
    except oracledb.DatabaseError as e:
        error_details = format_oracle_error(e)
        logger.error(f"Oracle error executing SQL statement: {error_details}")
        formatted_error = f"Oracle Error {error_details.get('code', 'Unknown')}: {error_details['message']}"
        if error_details.get('offset'):
            formatted_error += f"\nAt position: {error_details['offset']}"
        return [types.TextContent(type="text", text=formatted_error)]
    except Exception as e:
        logger.error(f"Error executing SQL statement: {e}")
        return [types.TextContent(type="text", text=f"Error executing SQL statement: {str(e)}")]

# Handler for list_user_tables_with_descriptions tool
async def handle_list_user_tables_with_descriptions_tool(arguments: dict) -> list[types.TextContent]:
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
        return [types.TextContent(type="text", text=str(table_list))]
    except Exception as e:
        logger.error(f"Error fetching user tables with descriptions: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_optimize_sql_with_ai_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for optimizing a SQL query using AI and optimization rules."""
    sql_query = arguments.get("sql_query")
    add_comments = arguments.get("add_comments", True)
    table_descriptions = arguments.get("table_descriptions", None)
    if not sql_query:
        return [types.TextContent(type="text", text="Missing required parameter: sql_query")]
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
                schemas_result = await handle_schemas_tool({"table_names": table_names})
                # schemas_result is a list of TextContent, take the first .text
                table_descriptions = schemas_result[0].text if schemas_result else None
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
            return [types.TextContent(type="text", text="Error: The AI optimization feature requires a valid API key. Please update the AI_API_KEY value in $HOME/.bashrc_srai with your Azure OpenAI API key, then restart your terminal.")]

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
        return [types.TextContent(type="text", text=ai_reply)]
    except Exception as e:
        logger.error(f"Error optimizing SQL with AI: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_explain_plan_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for sending EXPLAIN PLAN to Oracle and returning the execution plan."""
    sql_query = arguments.get("sql_query")
    if not sql_query:
        return [types.TextContent(type="text", text="Missing required parameter: sql_query")]
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
        return [types.TextContent(type="text", text=plan_text)]
    except Exception as e:
        logger.error(f"Error executing EXPLAIN PLAN: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_get_sql_optimization_rules_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for returning SQL optimization rules from optimization.json."""
    try:
        optimization_path = SCRIPT_DIR / "optimization.json"
        with open(optimization_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        return [types.TextContent(type="text", text=json.dumps(rules, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Error reading optimization.json: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_get_pro_c_rules_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for returning Pro*C coding rules from proc_rules.md."""
    try:
        proc_rules_path = SCRIPT_DIR / "proc_rules.md"
        with open(proc_rules_path, "r", encoding="utf-8") as f:
            rules = f.read()
        return [types.TextContent(type="text", text=rules)]
    except Exception as e:
        logger.error(f"Error reading proc_rules.md: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_get_valid_languages_tool(arguments: dict) -> list[types.TextContent]:
    """Handler for get_valid_languages tool"""
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
        return [types.TextContent(type="text", text=formatted_error)]
    return [types.TextContent(type="text", text=str(result["data"]))]

# Resource handlers
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """List available resources"""
    return []

@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """List available resource templates"""
    return []

# Ensure connection is initialized on startup
try:
    connection = get_db_connection()
    cache_table_names(connection)
except Exception as e:
    logger.error(f"Error during startup: {e}")
    raise

# Create the Starlette app
logger.info("Creating Starlette app...")
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)

# Start the server
logger.info("Starting Uvicorn server...")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
settings.json configuration:

"mcp": {
	"servers": {
	/*	"mcp-server-time": {
			"command": "python",
			"args": [
				"-m",
				"mcp_server_time",
				"--local-timezone=America/Los_Angeles"
			],
			"env": {}
		}, */
		"oracleMCP": {
			"type": "sse",
			"url": "http://0.0.0.0:8000/sse/",
			"headers": {
				"Content-Type": "application/json"
			}
		},
		"test10": {
			"type": "sse",
			"url": "http://0.0.0.0:8001/sse/",
			"headers": {
				"Content-Type": "application/json"
				}

				}
	}
},
"""
