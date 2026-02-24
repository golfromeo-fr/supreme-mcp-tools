#!/usr/bin/env python3
"""
RAG MCP Server - Streamable HTTP Transport
Provides tools for semantic code search, sparse search, and code indexing using Qdrant.
Migrated from SSE to Streamable HTTP transport.
"""
import sys
import os
import logging
import time
import subprocess
import psutil
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from contextlib import asynccontextmanager

# Check for required dependencies before importing
try:
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector
    from dotenv import load_dotenv
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure the virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Check for FastAPI and Streamable HTTP base
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import uvicorn
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure the virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# Add parent directories to path for importing StreamableHttpTransportBase
# The supreme-mcp-tools directory (parent of tools and launcher) needs to be in the path
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
    print("Try running from the supreme-mcp-tools directory: python tools/ragmcp/ragmcp_streamable.py", file=sys.stderr)
    sys.exit(1)

# Import optional dependencies
try:
    from indexer.sparse_vector_gen import generate_sparse_vector, get_global_generator
    SPARSE_VECTORS_AVAILABLE = True
except ImportError:
    SPARSE_VECTORS_AVAILABLE = False
    logging.warning("Sparse vector generator not available. Install sparse_vector_gen.py to enable sparse search.")

try:
    from copilot_context_injector import CopilotContextInjector, get_injector
    COPILOT_INJECTOR_AVAILABLE = True
except ImportError:
    COPILOT_INJECTOR_AVAILABLE = False
    logging.warning("Copilot context injector not available. Install copilot_context_injector.py for context injection.")

# Local embedding models
LOCAL_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
    logging.info("sentence-transformers available for local embeddings")
except ImportError:
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Configure logging
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR / "ragmcp.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ragmcp_streamable")

# Load configuration from .env file
env_path = SCRIPT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded configuration from {env_path}")

# Read embedding configuration
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'azure').lower()
LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'e5-large')

# Local embedding models configuration
LOCAL_EMBEDDING_MODELS = {
    'sbert-large': {
        'model_name': 'stsb-bert-large',
        'dimensions': 1024,
        'description': 'SBERT Large - English semantic similarity',
        'device': 'cpu'
    },
    'e5-large': {
        'model_name': 'intfloat/multilingual-e5-large',
        'dimensions': 1024,
        'description': 'Multilingual E5 Large - Multilingual semantic similarity',
        'device': 'cpu'
    },
    'small': {
        'model_name': 'BAAI/bge-small-en-v1.5',
        'dimensions': 384,
        'description': 'BGE Small - Fast English embeddings',
        'device': 'cpu'
    },
    'base': {
        'model_name': 'BAAI/bge-base-en-v1.5',
        'dimensions': 768,
        'description': 'BGE Base - Balanced English embeddings',
        'device': 'cpu'
    }
}

# Global model cache
_local_embedding_model = None
_local_embedding_model_name = None


def get_local_embedding_model(model_name: str = None):
    """Get or create a local embedding model instance."""
    global _local_embedding_model, _local_embedding_model_name
    
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        return None
    
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL
    
    if _local_embedding_model is not None and _local_embedding_model_name == model_name:
        logger.debug(f"Using cached local embedding model: {model_name}")
        return _local_embedding_model
    
    model_config = LOCAL_EMBEDDING_MODELS.get(model_name)
    if not model_config:
        logger.warning(f"Unknown local embedding model: {model_name}, falling back to e5-large")
        model_config = LOCAL_EMBEDDING_MODELS['e5-large']
    
    try:
        logger.info(f"Loading local embedding model: {model_config['model_name']}")
        _local_embedding_model = SentenceTransformer(model_config['model_name']).to(model_config['device'])
        _local_embedding_model_name = model_name
        logger.info(f"Local embedding model loaded: {model_config['description']} ({model_config['dimensions']}d)")
        return _local_embedding_model
    except Exception as e:
        logger.error(f"Failed to load local embedding model {model_config['model_name']}: {e}")
        return None


def generate_local_embeddings(texts: list, model_name: str = None):
    """Generate embeddings for a list of texts using local models."""
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        logger.error("Local embeddings not available - sentence-transformers not installed")
        return None
    
    model = get_local_embedding_model(model_name)
    if model is None:
        logger.error("Failed to get local embedding model")
        return None
    
    try:
        logger.debug(f"Generating local embeddings for {len(texts)} texts using model: {model_name or LOCAL_EMBEDDING_MODEL}")
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating local embeddings: {e}")
        return None


# Log startup information
logger.info("="*80)
logger.info("RAG MCP Server Starting (Streamable HTTP)")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Embedding provider: {EMBEDDING_PROVIDER}")
logger.info(f"Local embedding model: {LOCAL_EMBEDDING_MODEL}")
logger.info("="*80)

# Protocol version compatibility
SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-11-25"]


# ============================================================================
# RAGMCP Streamable HTTP Transport Implementation
# ============================================================================

class RAGMCPStreamableHttp(StreamableHttpTransportBase):
    """
    RAG MCP server implementation using Streamable HTTP transport.
    
    This class provides semantic code search, sparse search, and code indexing
    capabilities using Streamable HTTP transport with JSON-RPC framing.
    """
    
    def __init__(self):
        """Initialize the RAG MCP Streamable HTTP server."""
        config = StreamableHttpConfig(
            endpoint="/mcp",
            framing_format="newline-delimited",
            request_timeout=60.0,  # Longer timeout for Qdrant operations
        )
        super().__init__("ragmcp", config)
        logger.info("RAG MCP Streamable HTTP transport initialized")
        
        # Initialize Qdrant client
        self._init_qdrant_client()
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        logger.info("Initializing Qdrant client for semantic code search...")
        try:
            qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
            qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Qdrant client connected to {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.warning(f"Could not initialize Qdrant client: {e}")
            self.qdrant_client = None
    
    async def _handle_initialize(self, params, session):
        """Handle initialize request - tools are supported."""
        protocol_version = params.get("protocolVersion", "2024-11-05")
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            logger.warning(f"Client sent unsupported protocol version: {protocol_version}, using 2024-11-05")
            protocol_version = "2024-11-05"
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {
                    "tools": {},  # Tools are supported
                    # resources and prompts are not included
                },
                "serverInfo": {
                    "name": self.server_name,
                    "version": "1.0.0",
                },
            },
        }
    
    async def _handle_tools_list(self, params, session) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = [
            {
                "name": "search_code",
                "description": "Semantic search across indexed code (Pro*C, PL/SQL, Java, etc.) using natural language. Returns relevant code chunks with function names and file locations.",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'functions that process customer orders', 'EXEC SQL blocks that insert into STOMVT')"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Filter by file type: 'proc' (Pro*C), 'plsql' (PL/SQL), 'java', 'sql', etc. (optional)"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter by function/procedure name (optional)"
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Qdrant collection to search (default: folder.to.index-database-code)",
                            "default": "folder.to.index-database-code"
                        }
                    }
                }
            },
            {
                "name": "search_code_sparse",
                "description": "Lexical (BM25-style) code search using sparse vectors. Excellent for finding exact identifiers, table names, function names. Works offline without API costs. Use this for precise code lookups (e.g., 'STOMVT table', 'get_movement_type function').",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (identifiers, table names, function names work best)"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Filter by file type: 'proc' (Pro*C), 'plsql' (PL/SQL), 'java', 'sql', etc. (optional)"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter by function/procedure name (optional)"
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Qdrant collection to search (default: folder.to.index-database-code)",
                            "default": "folder.to.index-database-code"
                        }
                    }
                }
            },
            {
                "name": "get_copilot_context",
                "description": "Get formatted code context for GitHub Copilot injection. Retrieves relevant code using sparse vectors and formats it as inline comments or markdown. Perfect for making Copilot project-aware.",
                "inputSchema": {
                    "type": "object",
                    "required": ["current_context"],
                    "properties": {
                        "current_context": {
                            "type": "string",
                            "description": "Current code snippet or keywords from cursor position (5-10 lines recommended)"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["comment", "sidebar"],
                            "description": "Output format: 'comment' (C-style /* */ for inline injection), 'sidebar' (markdown for panel)",
                            "default": "comment"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["c", "python", "sql"],
                            "description": "Programming language for comment style (default: c)",
                            "default": "c"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of code chunks to retrieve (default: 3)",
                            "default": 3
                        },
                        "max_lines": {
                            "type": "number",
                            "description": "Maximum total lines of context (default: 50)",
                            "default": 50
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Qdrant collection to search (default: folder.to.index-database-code)",
                            "default": "folder.to.index-database-code"
                        }
                    }
                }
            },
            {
                "name": "start_indexing",
                "description": "Start background indexing of code files into Qdrant. Returns process ID (PID) for monitoring. Indexes Pro*C, PL/SQL, Java, and other files with smart function-level chunking. Supports sparse (BM25, $0), dense (embeddings, API cost), or hybrid modes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workspace_root": {
                            "type": "string",
                            "description": "Root workspace path to index (default: /path/to/your/workspace)",
                            "default": "/path/to/your/workspace"
                        },
                        "collection_name": {
                            "type": "string",
                            "description": "Qdrant collection name (default: folder.to.index-database-code)",
                            "default": "folder.to.index-database-code"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["sparse", "dense", "hybrid"],
                            "description": "Indexing mode: 'sparse' (BM25, offline, $0), 'dense' (embeddings, API cost), 'hybrid' (both, best quality). Defaults to config file setting if not specified."
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force reindex by deleting existing collection (default: false)",
                            "default": False
                        },
                        "directories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific directories to index (optional, default: all configured dirs)"
                        }
                    }
                }
            },
            {
                "name": "check_indexing_progress",
                "description": "Check the progress of background indexing process. Returns status, files processed, chunks indexed, and recent log entries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pid": {
                            "type": "number",
                            "description": "Process ID returned by start_indexing (optional, will check last known process if not provided)"
                        }
                    }
                }
            },
            {
                "name": "clear_index",
                "description": "Warning: Clear all indexed code from Qdrant vector database. Only use this when you are CERTAIN an index needs to be deleted. This is NOT needed for normal indexing operations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Qdrant collection to delete. Use 'ALL' to delete all collections (default: folder.to.index-application-code)",
                            "default": "folder.to.index-application-code"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag - must be true to delete (safety check)",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "list_collections",
                "description": "List all Qdrant collections with their stats (number of chunks, vector dimensions). Use this to see what's already indexed before starting new indexing.",
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
    
    async def _handle_tool_call(self, params, session, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Tool called: {tool_name} with arguments: {arguments}")
        
        try:
            if tool_name == "search_code":
                async for response in self._handle_search_code_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "search_code_sparse":
                async for response in self._handle_search_code_sparse_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "get_copilot_context":
                async for response in self._handle_get_copilot_context_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "start_indexing":
                async for response in self._handle_start_indexing_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "check_indexing_progress":
                async for response in self._handle_check_indexing_progress_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "clear_index":
                async for response in self._handle_clear_index_tool(arguments, request_id):
                    yield response
            
            elif tool_name == "list_collections":
                async for response in self._handle_list_collections_tool(arguments, request_id):
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
        
        except Exception as e:
            logger.error(f"Error in tool call '{tool_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e) if self.config.include_stack_traces else None
                }
            }

    # ========================================================================
    # Tool Handlers (adapted from original ragmcp.py)
    # ========================================================================

    async def _handle_search_code_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for semantic code search using Qdrant."""
        logger.debug(f"Processing search_code tool with arguments: {arguments}")

        if not self.qdrant_client:
            error_msg = "Qdrant client not initialized. Code search is unavailable."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        query = arguments.get("query")
        if not query:
            logger.error("Missing required parameter: query")
            raise ValueError("Missing argument: query")

        limit = arguments.get("limit", 5)
        file_type = arguments.get("file_type")
        function_name = arguments.get("function_name")
        collection_name = arguments.get("collection_name", "folder.to.index-database-code")

        try:
            logger.info(f"Searching code: '{query}' (limit={limit}, collection={collection_name})")

            # Generate embedding
            embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'azure')

            if embedding_provider == 'azure':
                azure_api_url = os.getenv('AZURE_EMBEDDING_API_URL',
                                         'https://put.your.API.gateway.ai/v1/embeddings')
                azure_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
                api_key = os.getenv('AI_API_KEY', '')

                if not api_key:
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "Error: AI_API_KEY not set. Cannot generate search embedding."}]}
                    }
                    return

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        azure_api_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}"
                        },
                        json={
                            "input": [query],
                            "model": azure_model
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    query_vector = data['data'][0]['embedding']
            else:
                if not LOCAL_EMBEDDINGS_AVAILABLE:
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": "Error: Local embeddings module not available. Install sentence-transformers."}]}
                    }
                    return

                try:
                    model_info = LOCAL_EMBEDDING_MODELS.get(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODELS['e5-large'])
                    logger.info(f"Using local embeddings for search query (model: {LOCAL_EMBEDDING_MODEL} - {model_info['description']})")
                    query_embeddings = generate_local_embeddings([query], model_name=LOCAL_EMBEDDING_MODEL)

                    if query_embeddings is None or len(query_embeddings) == 0:
                        yield {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {"content": [{"type": "text", "text": "Error: Failed to generate local embedding for query."}]}
                        }
                        return

                    query_vector = query_embeddings[0].tolist()
                    logger.info(f"Generated local embedding: dimension={len(query_vector)}")

                except Exception as e:
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error generating local embedding: {str(e)}"}]}
                    }
                    return

            # Build filter
            search_filter = None
            conditions = []

            if file_type:
                conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

            if function_name:
                conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

            if conditions:
                search_filter = Filter(must=conditions)

            # Perform search
            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                vectors_config = collection_info.config.params.vectors

                if isinstance(vectors_config, dict):
                    vector_names = list(vectors_config.keys())
                    vector_dim = len(query_vector)
                    vector_name = None

                    for vname in vector_names:
                        vconfig = vectors_config[vname]
                        if hasattr(vconfig, 'size') and vconfig.size == vector_dim:
                            vector_name = vname
                            logger.info(f"Using vector '{vector_name}' (matches dimension {vector_dim})")
                            break

                    if not vector_name:
                        vector_name = vector_names[0]
                        logger.warning(f"No matching vector dimension, using first: '{vector_name}'")

                    query_response = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        using=vector_name,
                        query_filter=search_filter,
                        limit=limit,
                        with_payload=True
                    )
                else:
                    logger.info(f"Collection has single unnamed vector")
                    query_response = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        query_filter=search_filter,
                        limit=limit,
                        with_payload=True
                    )
            except Exception as e:
                logger.error(f"Error detecting vector config: {e}, trying without 'using' parameter")
                query_response = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )

            search_results = query_response.points

            # Format results
            if not search_results:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "No results found."}]}
                }
                return

            formatted_results = []
            formatted_results.append(f"Found {len(search_results)} relevant code chunks:\n")
            formatted_results.append("=" * 80 + "\n")

            for i, hit in enumerate(search_results, 1):
                payload = hit.payload
                score = hit.score

                formatted_results.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
                formatted_results.append(f"File: {payload.get('filePath', 'Unknown')}\n")
                formatted_results.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
                formatted_results.append(f"Type: {payload.get('fileType', 'Unknown')}\n")

                if payload.get('functionName'):
                    formatted_results.append(f"Function: {payload['functionName']}\n")
                if payload.get('chunkType'):
                    formatted_results.append(f"Chunk Type: {payload['chunkType']}\n")

                formatted_results.append("\nCode:\n```\n")
                code_chunk = payload.get('codeChunk', '')
                lines = code_chunk.split('\n')
                if len(lines) > 50:
                    formatted_results.append('\n'.join(lines[:50]))
                    formatted_results.append(f"\n... ({len(lines) - 50} more lines)")
                else:
                    formatted_results.append(code_chunk)
                formatted_results.append("\n```\n")
                formatted_results.append("-" * 80 + "\n")

            result_text = ''.join(formatted_results)
            logger.info(f"Search completed: {len(search_results)} results")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": result_text}]}
            }

        except Exception as e:
            error_msg = f"Error searching code: {str(e)}"
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }

    async def _handle_search_code_sparse_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for sparse vector (lexical/BM25) code search."""
        logger.debug(f"Processing search_code_sparse tool with arguments: {arguments}")

        if not SPARSE_VECTORS_AVAILABLE:
            error_msg = "Sparse vector search not available. Missing sparse_vector_gen.py module."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        if not self.qdrant_client:
            error_msg = "Qdrant client not initialized. Code search is unavailable."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        query = arguments.get("query")
        if not query:
            logger.error("Missing required parameter: query")
            raise ValueError("Missing argument: query")

        limit = arguments.get("limit", 5)
        file_type = arguments.get("file_type")
        function_name = arguments.get("function_name")
        collection_name = arguments.get("collection_name", "folder.to.index-database-code")

        try:
            logger.info(f"Sparse searching code: '{query}' (limit={limit}, collection={collection_name})")

            query_metadata = {'language': file_type if file_type else 'unknown'}
            query_sparse_vec = generate_sparse_vector(query, query_metadata)

            if not query_sparse_vec:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "Error: Failed to generate sparse vector for query."}]}
                }
                return

            search_filter = None
            conditions = []

            if file_type:
                conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

            if function_name:
                conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

            if conditions:
                search_filter = Filter(must=conditions)

            query_response = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=SparseVector(
                    indices=list(query_sparse_vec.keys()),
                    values=list(query_sparse_vec.values())
                ),
                using="sparse",
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            search_results = query_response.points

            if not search_results:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "No results found."}]}
                }
                return

            formatted_results = []
            formatted_results.append(f"Found {len(search_results)} relevant code chunks (sparse/lexical search):\n")
            formatted_results.append("=" * 80 + "\n")

            for i, hit in enumerate(search_results, 1):
                payload = hit.payload
                score = hit.score

                formatted_results.append(f"\n**Result {i}** (relevance: {score:.3f})\n")
                formatted_results.append(f"File: {payload.get('filePath', 'Unknown')}\n")
                formatted_results.append(f"Lines: {payload.get('startLine', '?')}-{payload.get('endLine', '?')}\n")
                formatted_results.append(f"Type: {payload.get('fileType', 'Unknown')}\n")

                if payload.get('functionName'):
                    formatted_results.append(f"Function: {payload['functionName']}\n")
                if payload.get('chunkType'):
                    formatted_results.append(f"Chunk Type: {payload['chunkType']}\n")

                formatted_results.append("\nCode:\n```\n")
                code_chunk = payload.get('codeChunk', '')
                lines = code_chunk.split('\n')
                if len(lines) > 50:
                    formatted_results.append('\n'.join(lines[:50]))
                    formatted_results.append(f"\n... ({len(lines) - 50} more lines)")
                else:
                    formatted_results.append(code_chunk)
                formatted_results.append("\n```\n")
                formatted_results.append("-" * 80 + "\n")

            result_text = ''.join(formatted_results)
            logger.info(f"Sparse search completed: {len(search_results)} results")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": result_text}]}
            }

        except Exception as e:
            error_msg = f"Error in sparse code search: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }

    async def _handle_get_copilot_context_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler for getting formatted context for GitHub Copilot injection."""
        logger.debug(f"Processing get_copilot_context tool with arguments: {arguments}")

        if not COPILOT_INJECTOR_AVAILABLE:
            error_msg = "Copilot context injector not available. Missing copilot_context_injector.py module."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        if not SPARSE_VECTORS_AVAILABLE:
            error_msg = "Sparse vector search required for context injection. Missing sparse_vector_gen.py module."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        if not self.qdrant_client:
            error_msg = "Qdrant client not initialized. Context injection is unavailable."
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }
            return

        current_context = arguments.get("current_context")
        if not current_context:
            logger.error("Missing required parameter: current_context")
            raise ValueError("Missing argument: current_context")

        format_type = arguments.get("format", "comment")
        language = arguments.get("language", "c")
        limit = arguments.get("limit", 3)
        max_lines = arguments.get("max_lines", 50)
        collection_name = arguments.get("collection_name", "folder.to.index-database-code")

        try:
            logger.info(f"Getting Copilot context for: '{current_context[:50]}...' (format={format_type}, limit={limit})")

            injector = get_injector(max_context_lines=max_lines)
            keywords = injector.extract_keywords_from_context(current_context)

            if not keywords:
                logger.warning("No keywords extracted from context")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": injector._format_no_context(language)}]}
                }
                return

            search_query = ' '.join(keywords[:5])
            logger.debug(f"Extracted keywords for search: {keywords[:5]}")

            query_metadata = {'language': 'unknown'}
            query_sparse_vec = generate_sparse_vector(search_query, query_metadata)

            if not query_sparse_vec:
                logger.warning("Failed to generate sparse vector")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": injector._format_no_context(language)}]}
                }
                return

            query_response = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=SparseVector(
                    indices=list(query_sparse_vec.keys()),
                    values=list(query_sparse_vec.values())
                ),
                using="sparse",
                limit=limit,
                with_payload=True
            )
            search_results = query_response.points

            if not search_results:
                logger.warning("No code chunks found")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": injector._format_no_context(language)}]}
                }
                return

            chunks = [hit.payload for hit in search_results]
            logger.info(f"Found {len(chunks)} relevant chunks")

            if format_type == "comment":
                formatted_context = injector.format_context_comment(chunks, max_lines, language)
            elif format_type == "sidebar":
                formatted_context = injector.format_sidebar_context(chunks)
            else:
                formatted_context = injector.format_context_comment(chunks, max_lines, language)

            logger.info(f"Context formatted successfully ({len(formatted_context)} chars)")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": formatted_context}]}
            }

        except Exception as e:
            error_msg = f"Error generating Copilot context: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }

    async def _handle_start_indexing_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler to start background indexing process."""
        logger.debug(f"Processing start_indexing tool with arguments: {arguments}")

        workspace_root = arguments.get("workspace_root", "/path/to/your/workspace")
        collection_name = arguments.get("collection_name", "your-database-code")
        mode = arguments.get("mode")
        force = arguments.get("force", False)
        directories = arguments.get("directories")

        try:
            if not self.qdrant_client:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT environment variables."}]}
                }
                return

            collection_exists = False
            try:
                self.qdrant_client.get_collection(collection_name)
                collection_exists = True
                logger.info(f"Collection '{collection_name}' already exists")
            except Exception as e:
                if "Not found" in str(e) or "doesn't exist" in str(e):
                    logger.info(f"Collection '{collection_name}' does not exist, will be auto-created by indexer")
                    collection_exists = False
                else:
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"❌ Error checking collection: {str(e)}"}]}
                    }
                    return

            if force and collection_exists:
                message = """⚠️  **Collection Already Exists**

Collection `{collection_name}` already exists. The `force` parameter is no longer used to delete existing collections for safety.

**Options:**
1. **Continue indexing** (recommended): Remove `force=true` and rerun. The indexer will skip already-indexed files.
2. **Clear and reindex**: First use `clear_index(collection_name="{collection_name}", confirm=true)` to delete the collection, then start indexing again.

**Why this changed:** To prevent accidental data loss, the indexer now requires explicit confirmation before deleting existing indexes."""
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": message}]}
                }
                return

            indexer_script = SCRIPT_DIR.parent.parent / "oraclemcp" / "indexer" / "incremental_indexer.py"
            if not indexer_script.exists():
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": f"Error: Indexer script not found at {indexer_script}"}]}
                }
                return

            cmd = [
                "python3",
                str(indexer_script),
                workspace_root,
                "--collection", collection_name
            ]

            if mode:
                cmd.extend(["--mode", mode])

            if force:
                cmd.append("--force-reindex")

            if directories:
                cmd.extend(["--dirs"] + directories)

            logs_dir = SCRIPT_DIR / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"indexing_{collection_name}.log"

            logger.info(f"Starting indexing process: {' '.join(cmd)}")

            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=SCRIPT_DIR.parent.parent / "oraclemcp",
                    start_new_session=True
                )

            pid = process.pid

            pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
            with open(pid_file, 'w') as f:
                json.dump({
                    "pid": pid,
                    "workspace_root": workspace_root,
                    "collection_name": collection_name,
                    "log_file": str(log_file),
                    "started_at": time.time()
                }, f)

            result = f"""✅ Indexing started successfully!

**Process Information:**
- PID: {pid}
- Workspace: {workspace_root}
- Collection: {collection_name}
- Force reindex: {force}
- Log file: {log_file}

**Next Steps:**
1. Monitor progress: Use check_indexing_progress tool
2. View live logs: tail -f {log_file}
3. Check collection: Query Qdrant at qdrant:6333

The indexing process is running in the background. Use check_indexing_progress to monitor status.
"""

            logger.info(f"Indexing started with PID {pid}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        except Exception as e:
            error_msg = f"Error starting indexing: {str(e)}"
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }

    async def _handle_check_indexing_progress_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Handler to check indexing progress."""
        logger.debug(f"Processing check_indexing_progress tool with arguments: {arguments}")

        pid = arguments.get("pid")

        try:
            pid_file = SCRIPT_DIR / "logs" / "indexing.pid"
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    pid_info = json.load(f)
                    if not pid:
                        pid = pid_info.get("pid")
                    log_file = Path(pid_info.get("log_file", SCRIPT_DIR / "logs" / "indexing.log"))
                    collection_name = pid_info.get("collection_name", "folder.to.index-application-code")
                    workspace_root = pid_info.get("workspace_root", "Unknown")
                    started_at = pid_info.get("started_at", 0)
            else:
                log_file = SCRIPT_DIR / "logs" / "indexing.log"
                collection_name = "folder.to.index-application-code"
                workspace_root = "Unknown"
                started_at = 0

            is_running = False
            if pid:
                try:
                    process = psutil.Process(pid)
                    is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    is_running = False

            if not log_file.exists():
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "No indexing log found. Start indexing first with start_indexing tool."}]}
                }
                return

            with open(log_file, 'r') as f:
                log_lines = f.readlines()

            files_processed = sum(1 for line in log_lines if "Processing [" in line)
            chunks_indexed = sum(1 for line in log_lines if "✓ Indexed batch" in line)
            errors = sum(1 for line in log_lines if "ERROR" in line)

            total_files = "Unknown"
            for line in reversed(log_lines):
                if "Found " in line and " files to index" in line:
                    try:
                        total_files = line.split("Found ")[1].split(" files")[0]
                        break
                    except:
                        pass

            recent_logs = ''.join(log_lines[-15:]) if log_lines else "No logs yet"

            collection_stats = "Unknown"
            if self.qdrant_client:
                try:
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    points_count = collection_info.points_count
                    collection_stats = f"{points_count:,} chunks indexed"
                except Exception as e:
                    collection_stats = f"Error: {str(e)}"

            runtime = "Unknown"
            if started_at > 0:
                elapsed = int(time.time() - started_at)
                minutes = elapsed // 60
                seconds = elapsed % 60
                runtime = f"{minutes}m {seconds}s"

            status_icon = "✓ RUNNING" if is_running else "✗ COMPLETED/STOPPED"

            result = f"""📊 Indexing Progress Report

**Status:** {status_icon}
**Process ID:** {pid if pid else 'Unknown'}
**Workspace:** {workspace_root}
**Collection:** {collection_name}
**Runtime:** {runtime}

**Progress:**
- Files processed: {files_processed} / {total_files}
- Chunks indexed: {chunks_indexed} batches
- Errors: {errors}
- Collection size: {collection_stats}

**Recent Log Entries (last 15 lines):**
```
{recent_logs}```

**Log File:** {log_file}

{'Process is still running. Check back later for updates.' if is_running else 'Process has completed. Review logs for final status.'}
"""

            logger.info(f"Progress check: {files_processed} files, {chunks_indexed} batches, running={is_running}")
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        except Exception as e:
            error_msg = f"Error checking indexing progress: {str(e)}"
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"Error: {error_msg}"}]}
            }

    async def _handle_list_collections_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """List all Qdrant collections with their stats"""
        try:
            if not self.qdrant_client:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."}]}
                }
                return

            collections = self.qdrant_client.get_collections().collections

            if not collections:
                message = """📊 **No Collections Found**

The Qdrant vector database is empty. No code has been indexed yet.

**To start indexing:**
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="your-database-code")
```
"""
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": message}]}
                }
                return

            result = ["📊 **Qdrant Collections**\n"]
            result.append("=" * 80 + "\n\n")

            for collection in collections:
                try:
                    collection_info = self.qdrant_client.get_collection(collection.name)
                    points_count = collection_info.points_count
                    vector_size = collection_info.config.params.vectors.size
                    distance = collection_info.config.params.vectors.distance

                    result.append(f"**{collection.name}**\n")
                    result.append(f"  - Chunks indexed: {points_count:,}\n")
                    result.append(f"  - Vector dimensions: {vector_size}\n")
                    result.append(f"  - Distance metric: {distance}\n")
                    result.append("\n")
                except Exception as e:
                    result.append(f"**{collection.name}**\n")
                    result.append(f"  - Error: {str(e)}\n\n")

            result.append("-" * 80 + "\n")
            result.append(f"\n**Total collections:** {len(collections)}\n")
            result.append("\n**Note:** The standard embedding dimension is 3072 for text-embedding-3-large model.")

            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": ''.join(result)}]}
            }

        except Exception as e:
            error_msg = f"Error listing collections: {str(e)}"
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
            }

    async def _handle_clear_index_tool(self, arguments: dict, request_id) -> AsyncGenerator[Dict[str, Any], None]:
        """Clear all indexed code from Qdrant vector database"""
        try:
            collection_name = arguments.get("collection_name", "folder.to.index-application-code")
            confirm = arguments.get("confirm", False)

            if not self.qdrant_client:
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT."}]}
                }
                return

            delete_all = collection_name.upper() == "ALL"

            if not confirm:
                if delete_all:
                    collections = self.qdrant_client.get_collections().collections
                    collection_list = "\n".join([f"  - {c.name}" for c in collections])

                    message = f"""⚠️  **CONFIRMATION REQUIRED - DELETE ALL COLLECTIONS**

You are about to **DELETE EVERYTHING** from Qdrant, including:
{collection_list}

This action is **IRREVERSIBLE** and will:
- Delete ALL indexed code chunks from ALL collections
- Remove ALL embeddings (including Roo Code indexing)
- Completely wipe the vector database

To confirm, call this tool again with `confirm: true`

Example:
```
clear_index(collection_name="ALL", confirm=true)
```
"""
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": message}]}
                    }
                else:
                    message = f"""⚠️  **CONFIRMATION REQUIRED**

You are about to **DELETE ALL DATA** from collection: `{collection_name}`

This action is **IRREVERSIBLE** and will:
- Delete all indexed code chunks
- Remove all embeddings
- Clear the vector database

To confirm, call this tool again with `confirm: true`

Example:
```
clear_index(collection_name="{collection_name}", confirm=true)
```
"""
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": message}]}
                    }
                return

            if delete_all:
                collections = self.qdrant_client.get_collections().collections
                total_deleted = 0
                deleted_names = []

                for collection in collections:
                    try:
                        collection_info = self.qdrant_client.get_collection(collection.name)
                        points_count = collection_info.points_count
                        self.qdrant_client.delete_collection(collection.name)
                        total_deleted += points_count
                        deleted_names.append(f"  - {collection.name} ({points_count:,} chunks)")
                        logger.info(f"Deleted collection '{collection.name}' with {points_count} chunks")
                    except Exception as e:
                        deleted_names.append(f"  - {collection.name} (error: {str(e)})")

                deleted_list = "\n".join(deleted_names)
                result = f"""✅ **All Collections Cleared Successfully**

**Deleted {len(collections)} collections:**
{deleted_list}

**Total chunks deleted:** {total_deleted:,}
**Status:** Qdrant completely wiped clean

You can now start fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="your-application-code")
```
"""
                logger.info(f"Cleared ALL collections - total {total_deleted} chunks deleted")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": result}]}
                }
                return

            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                points_count = collection_info.points_count

                self.qdrant_client.delete_collection(collection_name)

                result = f"""✅ **Index Cleared Successfully**

**Collection:** `{collection_name}`
**Deleted:** {points_count:,} code chunks
**Status:** Collection removed from Qdrant

The vector database has been wiped clean. You can now start a fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="{collection_name}")
```
"""
                logger.info(f"Cleared collection '{collection_name}' - deleted {points_count} chunks")
                yield {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": result}]}
                }

            except Exception as e:
                if "Not found" in str(e) or "doesn't exist" in str(e):
                    message = f"""ℹ️  **Collection Not Found**

Collection `{collection_name}` does not exist in Qdrant.

**Available actions:**
- Start indexing: `start_indexing(workspace_root="/path/to/your/workspace")`
- Check progress: `check_indexing_progress()`
"""
                    yield {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": message}]}
                    }
                else:
                    raise e

        except Exception as e:
            error_msg = f"Error clearing index: {str(e)}"
            logger.error(error_msg)
            yield {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": f"❌ Error: {error_msg}"}]}
            }


# ============================================================================
# FastAPI Application
# ============================================================================

# Create transport instance
transport = RAGMCPStreamableHttp()

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    logger.info("RAG MCP Streamable HTTP server starting up...")
    
    # Log Qdrant client status (initialized in transport.__init__)
    if transport.qdrant_client:
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = os.getenv('QDRANT_PORT', '6333')
        logger.info(f"Qdrant client connected to {qdrant_host}:{qdrant_port}")
    else:
        logger.warning("Qdrant client not initialized. Code search features will be unavailable.")
    
    yield
    logger.info("RAG MCP Streamable HTTP server shutting down...")
    await transport.cleanup_sessions()

# Create FastAPI application with lifespan
app = FastAPI(
    title="RAG MCP Streamable HTTP Server",
    description="RAG and Code Indexing MCP server using Streamable HTTP transport",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "ragmcp",
        "version": "1.0.0",
        "transport": "streamable-http",
        "endpoint": "/mcp",
        "tools": [
            "search_code",
            "search_code_sparse",
            "get_copilot_context",
            "start_indexing",
            "check_indexing_progress",
            "clear_index",
            "list_collections"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": transport.get_session_count(),
        "qdrant_connected": transport.qdrant_client is not None
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
    
    # Process request
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
    
    parser = argparse.ArgumentParser(description="RAG MCP Streamable HTTP Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8004,
        help="Port to bind to (default: 8004)"
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
    logging.getLogger("ragmcp_streamable").setLevel(log_level)
    
    logger.info(f"Starting RAG MCP Streamable HTTP Server on http://{args.host}:{args.port}")
    
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
    "ragmcp": {
      "type": "streamable-http",
      "url": "http://localhost:8004/mcp",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}
"""