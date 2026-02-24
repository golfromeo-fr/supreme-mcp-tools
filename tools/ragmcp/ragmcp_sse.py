#!/usr/bin/env python3
"""
RAG MCP Server - RAG and Code Indexing Tools
Provides tools for semantic code search, sparse search, and code indexing using Qdrant.
"""
import sys
import os
import logging
import time
import subprocess
import psutil
import json
from pathlib import Path
from typing import Optional

# Check for required dependencies before importing
try:
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector
    from dotenv import load_dotenv
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}", file=sys.stderr)
    print("Please make sure the virtual environment is activated and all dependencies are installed.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
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
logger = logging.getLogger("ragmcp")

# Load configuration from .env file
env_path = SCRIPT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded configuration from {env_path}")

# Read embedding configuration
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'azure').lower()
LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'e5-large')  # 'sbert-large', 'e5-large', 'small', 'base'

# ============================================================================
# Local Embedding Model Definitions
# ============================================================================

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
    """
    Get or create a local embedding model instance.
    
    Args:
        model_name: Model identifier (sbert-large, e5-large, small, base)
                  Defaults to LOCAL_EMBEDDING_MODEL env var
    
    Returns:
        SentenceTransformer instance or None if not available
    """
    global _local_embedding_model, _local_embedding_model_name
    
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        return None
    
    # Use provided model name or default from environment
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL
    
    # Return cached model if same
    if _local_embedding_model is not None and _local_embedding_model_name == model_name:
        logger.debug(f"Using cached local embedding model: {model_name}")
        return _local_embedding_model
    
    # Get model configuration
    model_config = LOCAL_EMBEDDING_MODELS.get(model_name)
    if not model_config:
        logger.warning(f"Unknown local embedding model: {model_name}, falling back to e5-large")
        model_config = LOCAL_EMBEDDING_MODELS['e5-large']
    
    # Load model
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
    """
    Generate embeddings for a list of texts using local models.
    
    Args:
        texts: List of text strings to embed
        model_name: Model identifier (sbert-large, e5-large, small, base)
    
    Returns:
        numpy array of embeddings or None if failed
    """
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
logger.info("RAG MCP Server Starting")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Embedding provider: {EMBEDDING_PROVIDER}")
logger.info(f"Local embedding model: {LOCAL_EMBEDDING_MODEL}")
logger.info("="*80)

# ============================================================================
# Server Initialization
# ============================================================================

# Verify server components
logger.info("Initializing RAG MCP Server...")

try:
    server = Server("ragmcp")
    sse_transport = SseServerTransport("/messages/")
    logger.info("Server components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server components: {e}")
    sys.exit(1)

# ============================================================================
# Qdrant Client Initialization
# ============================================================================

logger.info("Initializing Qdrant client for semantic code search...")
try:
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Qdrant client connected to {qdrant_host}:{qdrant_port}")
except Exception as e:
    logger.warning(f"Could not initialize Qdrant client: {e}")
    qdrant_client = None

# ============================================================================
# SSE Handler
# ============================================================================

async def handle_sse(request):
    """Handle SSE connection and server initialization"""
    from starlette.responses import Response
    
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
    return Response()

# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_search_code_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for semantic code search using Qdrant.
    Searches indexed code (Pro*C, PL/SQL, Java, etc.) using natural language.
    """
    logger.debug(f"Processing search_code tool with arguments: {arguments}")

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Code search is unavailable."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    # Extract parameters
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

        # Detect collection's vector configuration
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' config: {collection_info.config}")
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")

        # First, we need to generate an embedding for the search query
        # We'll use the Azure API or local model (same as indexer)
        embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'azure')

        if embedding_provider == 'azure':
            # Generate embedding using Azure API
            azure_api_url = os.getenv('AZURE_EMBEDDING_API_URL',
                                     'https://put.your.API.gateway.ai/v1/embeddings')
            azure_model = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
            api_key = os.getenv('AI_API_KEY', '')

            if not api_key:
                return [types.TextContent(type="text",
                    text="Error: AI_API_KEY not set. Cannot generate search embedding.")]

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
            # Use local model
            if not LOCAL_EMBEDDINGS_AVAILABLE:
                return [types.TextContent(type="text",
                    text="Error: Local embeddings module not available. Install sentence-transformers.")]

            try:
                model_info = LOCAL_EMBEDDING_MODELS.get(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODELS['e5-large'])
                logger.info(f"Using local embeddings for search query (model: {LOCAL_EMBEDDING_MODEL} - {model_info['description']})")
                query_embeddings = generate_local_embeddings([query], model_name=LOCAL_EMBEDDING_MODEL)

                if query_embeddings is None or len(query_embeddings) == 0:
                    return [types.TextContent(type="text",
                        text="Error: Failed to generate local embedding for query.")]

                query_vector = query_embeddings[0].tolist()  # Convert numpy array to list
                logger.info(f"Generated local embedding: dimension={len(query_vector)}")

            except Exception as e:
                return [types.TextContent(type="text",
                    text=f"Error generating local embedding: {str(e)}")]

        # Build filter if file_type or function_name specified
        search_filter = None
        conditions = []

        if file_type:
            conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

        if function_name:
            conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

        if conditions:
            search_filter = Filter(must=conditions)

        # Perform search using query_points
        # Detect if collection uses named vectors or single vector
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            # Check if it's a dict (named vectors) or VectorParams (single vector)
            if isinstance(vectors_config, dict):
                # Named vectors - need to specify which one
                vector_names = list(vectors_config.keys())
                logger.info(f"Collection has named vectors: {vector_names}")

                # Try to find appropriate vector name
                vector_dim = len(query_vector)
                vector_name = None

                for vname in vector_names:
                    vconfig = vectors_config[vname]
                    if hasattr(vconfig, 'size') and vconfig.size == vector_dim:
                        vector_name = vname
                        logger.info(f"Using vector '{vector_name}' (matches dimension {vector_dim})")
                        break

                if not vector_name:
                    # Default to first vector name
                    vector_name = vector_names[0]
                    logger.warning(f"No matching vector dimension, using first: '{vector_name}'")

                query_response = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
            else:
                # Single unnamed vector
                logger.info(f"Collection has single unnamed vector")
                query_response = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
        except Exception as e:
            logger.error(f"Error detecting vector config: {e}, trying without 'using' parameter")
            query_response = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

        search_results = query_response.points

        # Format results
        if not search_results:
            return [types.TextContent(type="text", text="No results found.")]

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
            # Limit code display to first 50 lines to avoid overwhelming output
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
        return [types.TextContent(type="text", text=result_text)]

    except Exception as e:
        error_msg = f"Error searching code: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_search_code_sparse_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for sparse vector (lexical/BM25) code search.
    No API costs, works offline, excellent for exact identifiers.
    """
    logger.debug(f"Processing search_code_sparse tool with arguments: {arguments}")

    if not SPARSE_VECTORS_AVAILABLE:
        error_msg = "Sparse vector search not available. Missing sparse_vector_gen.py module."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Code search is unavailable."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    # Extract parameters
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

        # Generate sparse vector for the query
        query_metadata = {
            'language': file_type if file_type else 'unknown'
        }
        query_sparse_vec = generate_sparse_vector(query, query_metadata)

        if not query_sparse_vec:
            return [types.TextContent(type="text", text="Error: Failed to generate sparse vector for query.")]

        # Build filter if file_type or function_name specified
        search_filter = None
        conditions = []

        if file_type:
            conditions.append(FieldCondition(key="fileType", match=MatchValue(value=file_type)))

        if function_name:
            conditions.append(FieldCondition(key="functionName", match=MatchValue(value=function_name)))

        if conditions:
            search_filter = Filter(must=conditions)

        # Perform sparse search using query_points
        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=list(query_sparse_vec.keys()),
                values=list(query_sparse_vec.values())
            ),
            using="sparse",  # Use sparse vectors
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        search_results = query_response.points

        # Format results (same format as semantic search for consistency)
        if not search_results:
            return [types.TextContent(type="text", text="No results found.")]

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
            # Limit code display to first 50 lines
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
        return [types.TextContent(type="text", text=result_text)]

    except Exception as e:
        error_msg = f"Error in sparse code search: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_get_copilot_context_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler for getting formatted context for GitHub Copilot injection.
    Retrieves relevant code using sparse search and formats it for Copilot consumption.
    """
    logger.debug(f"Processing get_copilot_context tool with arguments: {arguments}")

    if not COPILOT_INJECTOR_AVAILABLE:
        error_msg = "Copilot context injector not available. Missing copilot_context_injector.py module."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    if not SPARSE_VECTORS_AVAILABLE:
        error_msg = "Sparse vector search required for context injection. Missing sparse_vector_gen.py module."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    if not qdrant_client:
        error_msg = "Qdrant client not initialized. Context injection is unavailable."
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]

    # Extract parameters
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

        # Get context injector instance
        injector = get_injector(max_context_lines=max_lines)

        # Extract keywords from current context
        keywords = injector.extract_keywords_from_context(current_context)

        if not keywords:
            logger.warning("No keywords extracted from context")
            return [types.TextContent(type="text", text=injector._format_no_context(language))]

        # Build search query from keywords (top 5 most relevant)
        search_query = ' '.join(keywords[:5])
        logger.debug(f"Extracted keywords for search: {keywords[:5]}")

        # Generate sparse vector for the query
        query_metadata = {'language': 'unknown'}
        query_sparse_vec = generate_sparse_vector(search_query, query_metadata)

        if not query_sparse_vec:
            logger.warning("Failed to generate sparse vector")
            return [types.TextContent(type="text", text=injector._format_no_context(language))]

        # Perform sparse search to get structured results
        query_response = qdrant_client.query_points(
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
            return [types.TextContent(type="text", text=injector._format_no_context(language))]

        # Extract chunk payloads
        chunks = [hit.payload for hit in search_results]

        logger.info(f"Found {len(chunks)} relevant chunks")

        # Format based on requested format
        if format_type == "comment":
            formatted_context = injector.format_context_comment(chunks, max_lines, language)
        elif format_type == "sidebar":
            formatted_context = injector.format_sidebar_context(chunks)
        else:
            formatted_context = injector.format_context_comment(chunks, max_lines, language)

        logger.info(f"Context formatted successfully ({len(formatted_context)} chars)")
        return [types.TextContent(type="text", text=formatted_context)]

    except Exception as e:
        error_msg = f"Error generating Copilot context: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_start_indexing_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler to start background indexing process.
    Auto-creates collection if it doesn't exist with correct dimensions (3072).
    Returns PID and status information.
    """
    logger.debug(f"Processing start_indexing tool with arguments: {arguments}")

    workspace_root = arguments.get("workspace_root", "/path/to/your/workspace")
    collection_name = arguments.get("collection_name", "your-database-code")
    mode = arguments.get("mode")  # Optional: sparse, dense, or hybrid
    force = arguments.get("force", False)
    directories = arguments.get("directories")

    try:
        # Check if Qdrant client is available
        if not qdrant_client:
            return [types.TextContent(type="text",
                text="❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT environment variables.")]

        # Auto-create collection if it doesn't exist (with correct dimensions)
        collection_exists = False
        try:
            qdrant_client.get_collection(collection_name)
            collection_exists = True
            logger.info(f"Collection '{collection_name}' already exists")
        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                logger.info(f"Collection '{collection_name}' does not exist, will be auto-created by indexer")
                collection_exists = False
            else:
                # Unknown error
                return [types.TextContent(type="text",
                    text=f"❌ Error checking collection: {str(e)}")]

        # If force=True and collection exists, warn but don't delete automatically
        if force and collection_exists:
            return [types.TextContent(type="text", text=f"""
⚠️  **Collection Already Exists**

Collection `{collection_name}` already exists. The `force` parameter is no longer used to delete existing collections for safety.

**Options:**
1. **Continue indexing** (recommended): Remove `force=true` and rerun. The indexer will skip already-indexed files.
2. **Clear and reindex**: First use `clear_index(collection_name="{collection_name}", confirm=true)` to delete the collection, then start indexing again.

**Why this changed:** To prevent accidental data loss, the indexer now requires explicit confirmation before deleting existing indexes.
""")]

        # Build command - using incremental indexer
        indexer_script = SCRIPT_DIR.parent.parent / "oraclemcp" / "indexer" / "incremental_indexer.py"
        if not indexer_script.exists():
            return [types.TextContent(type="text",
                text=f"Error: Indexer script not found at {indexer_script}")]

        cmd = [
            "python3",
            str(indexer_script),
            workspace_root,
            "--collection", collection_name
        ]

        # Add mode parameter if specified
        if mode:
            cmd.extend(["--mode", mode])

        if force:
            cmd.append("--force-reindex")

        if directories:
            cmd.extend(["--dirs"] + directories)

        # Create logs/ directory if it doesn't exist
        logs_dir = SCRIPT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file path in logs/ subfolder
        log_file = logs_dir / f"indexing_{collection_name}.log"

        # Start process in background
        logger.info(f"Starting indexing process: {' '.join(cmd)}")

        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=SCRIPT_DIR.parent.parent / "oraclemcp",
                start_new_session=True  # Detach from parent
            )

        pid = process.pid

        # Save PID to file in logs/ subfolder for later reference
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
        return [types.TextContent(type="text", text=result)]

    except Exception as e:
        error_msg = f"Error starting indexing: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_check_indexing_progress_tool(arguments: dict) -> list[types.TextContent]:
    """
    Handler to check indexing progress.
    Returns status, files processed, chunks indexed, errors, and recent logs.
    """
    logger.debug(f"Processing check_indexing_progress tool with arguments: {arguments}")

    pid = arguments.get("pid")

    try:
        # Load PID info from file if not provided (in logs/ subfolder)
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

        # Check if process is running
        is_running = False
        if pid:
            try:
                process = psutil.Process(pid)
                is_running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                is_running = False

        # Read log file
        if not log_file.exists():
            return [types.TextContent(type="text",
                text="No indexing log found. Start indexing first with start_indexing tool.")]

        with open(log_file, 'r') as f:
            log_lines = f.readlines()

        # Parse progress from logs
        files_processed = sum(1 for line in log_lines if "Processing [" in line)
        chunks_indexed = sum(1 for line in log_lines if "✓ Indexed batch" in line)
        errors = sum(1 for line in log_lines if "ERROR" in line)

        # Get total files if available
        total_files = "Unknown"
        for line in reversed(log_lines):
            if "Found " in line and " files to index" in line:
                try:
                    total_files = line.split("Found ")[1].split(" files")[0]
                    break
                except:
                    pass

        # Get last 15 lines of log
        recent_logs = ''.join(log_lines[-15:]) if log_lines else "No logs yet"

        # Get Qdrant collection stats
        collection_stats = "Unknown"
        if qdrant_client:
            try:
                collection_info = qdrant_client.get_collection(collection_name)
                points_count = collection_info.points_count
                collection_stats = f"{points_count:,} chunks indexed"
            except Exception as e:
                collection_stats = f"Error: {str(e)}"

        # Calculate runtime
        runtime = "Unknown"
        if started_at > 0:
            elapsed = int(time.time() - started_at)
            minutes = elapsed // 60
            seconds = elapsed % 60
            runtime = f"{minutes}m {seconds}s"

        # Build status report
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
        return [types.TextContent(type="text", text=result)]

    except Exception as e:
        error_msg = f"Error checking indexing progress: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_list_collections_tool(arguments: dict) -> list[types.TextContent]:
    """List all Qdrant collections with their stats"""
    try:
        if not qdrant_client:
            return [types.TextContent(type="text", text="❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT.")]

        collections = qdrant_client.get_collections().collections

        if not collections:
            return [types.TextContent(type="text", text="""
📊 **No Collections Found**

The Qdrant vector database is empty. No code has been indexed yet.

**To start indexing:**
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="your-database-code")
```
""")]

        # Build formatted output
        result = ["📊 **Qdrant Collections**\n"]
        result.append("=" * 80 + "\n\n")

        for collection in collections:
            try:
                collection_info = qdrant_client.get_collection(collection.name)
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

        return [types.TextContent(type="text", text=''.join(result))]

    except Exception as e:
        error_msg = f"Error listing collections: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"❌ Error: {error_msg}")]


async def handle_clear_index_tool(arguments: dict) -> list[types.TextContent]:
    """Clear all indexed code from Qdrant vector database"""
    try:
        collection_name = arguments.get("collection_name", "folder.to.index-application-code")
        confirm = arguments.get("confirm", False)

        if not qdrant_client:
            return [types.TextContent(type="text", text="❌ Error: Qdrant client not initialized. Check QDRANT_HOST and QDRANT_PORT.")]

        # Check if user wants to delete ALL collections
        delete_all = collection_name.upper() == "ALL"

        # Safety check - require explicit confirmation
        if not confirm:
            if delete_all:
                # Get all collections
                collections = qdrant_client.get_collections().collections
                collection_list = "\n".join([f"  - {c.name}" for c in collections])

                return [types.TextContent(type="text", text=f"""
⚠️  **CONFIRMATION REQUIRED - DELETE ALL COLLECTIONS**

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
""")]
            else:
                return [types.TextContent(type="text", text=f"""
⚠️  **CONFIRMATION REQUIRED**

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
""")]

        # Delete ALL collections
        if delete_all:
            collections = qdrant_client.get_collections().collections
            total_deleted = 0
            deleted_names = []

            for collection in collections:
                try:
                    collection_info = qdrant_client.get_collection(collection.name)
                    points_count = collection_info.points_count
                    qdrant_client.delete_collection(collection.name)
                    total_deleted += points_count
                    deleted_names.append(f"  - {collection.name} ({points_count:,} chunks)")
                    logger.info(f"Deleted collection '{collection.name}' with {points_count} chunks")
                except Exception as e:
                    deleted_names.append(f"  - {collection.name} (error: {str(e)})")

            deleted_list = "\n".join(deleted_names)
            result = f"""
✅ **All Collections Cleared Successfully**

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
            return [types.TextContent(type="text", text=result)]

        # Delete single collection
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            points_count = collection_info.points_count

            # Delete the collection
            qdrant_client.delete_collection(collection_name)

            result = f"""
✅ **Index Cleared Successfully**

**Collection:** `{collection_name}`
**Deleted:** {points_count:,} code chunks
**Status:** Collection removed from Qdrant

The vector database has been wiped clean. You can now start a fresh indexing with:
```
start_indexing(workspace_root="/path/to/your/workspace", collection_name="{collection_name}")
```
"""
            logger.info(f"Cleared collection '{collection_name}' - deleted {points_count} chunks")
            return [types.TextContent(type="text", text=result)]

        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                return [types.TextContent(type="text", text=f"""
ℹ️  **Collection Not Found**

Collection `{collection_name}` does not exist in Qdrant.

**Available actions:**
- Start indexing: `start_indexing(workspace_root="/path/to/your/workspace")`
- Check progress: `check_indexing_progress()`
""")]
            else:
                raise e

    except Exception as e:
        error_msg = f"Error clearing index: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"❌ Error: {error_msg}")]


# ============================================================================
# Tool Registration
# ============================================================================

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="search_code",
            description="Semantic search across indexed code (Pro*C, PL/SQL, Java, etc.) using natural language. Returns relevant code chunks with function names and file locations.",
            inputSchema={
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
        ),
        types.Tool(
            name="search_code_sparse",
            description="Lexical (BM25-style) code search using sparse vectors. Excellent for finding exact identifiers, table names, function names. Works offline without API costs. Use this for precise code lookups (e.g., 'STOMVT table', 'get_movement_type function').",
            inputSchema={
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
        ),
        types.Tool(
            name="get_copilot_context",
            description="Get formatted code context for GitHub Copilot injection. Retrieves relevant code using sparse vectors and formats it as inline comments or markdown. Perfect for making Copilot project-aware.",
            inputSchema={
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
        ),
        types.Tool(
            name="start_indexing",
            description="Start background indexing of code files into Qdrant. Returns process ID (PID) for monitoring. Indexes Pro*C, PL/SQL, Java, and other files with smart function-level chunking. Supports sparse (BM25, $0), dense (embeddings, API cost), or hybrid modes.",
            inputSchema={
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
        ),
        types.Tool(
            name="check_indexing_progress",
            description="Check the progress of background indexing process. Returns status, files processed, chunks indexed, and recent log entries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "number",
                        "description": "Process ID returned by start_indexing (optional, will check last known process if not provided)"
                    }
                }
            }
        ),
        types.Tool(
            name="clear_index",
            description="Warning: Clear all indexed code from Qdrant vector database. Only use this when you are CERTAIN an index needs to be deleted. This is NOT needed for normal indexing operations.",
            inputSchema={
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
        ),
        types.Tool(
            name="list_collections",
            description="List all Qdrant collections with their stats (number of chunks, vector dimensions). Use this to see what's already indexed before starting new indexing.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


# ============================================================================
# Tool Router
# ============================================================================

@server.call_tool()
async def tool_router(name: str, arguments: dict) -> list[types.TextContent]:
    """Route tool calls to appropriate handlers"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    handlers = {
        "search_code": handle_search_code_tool,
        "search_code_sparse": handle_search_code_sparse_tool,
        "get_copilot_context": handle_get_copilot_context_tool,
        "start_indexing": handle_start_indexing_tool,
        "check_indexing_progress": handle_check_indexing_progress_tool,
        "clear_index": handle_clear_index_tool,
        "list_collections": handle_list_collections_tool
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)

    logger.error(f"Unknown tool: {name}")
    raise ValueError(f"Unknown tool: {name}")


# ============================================================================
# Resource Handlers
# ============================================================================

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return []


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    return []


# ============================================================================
# Create Starlette App
# ============================================================================

app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG MCP Server on http://0.0.0.0:8004")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8004)
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


"""
VSCode settings.json configuration:

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
"""
