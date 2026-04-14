#!/usr/bin/env python3
"""
Incremental code indexer with dual filtering and change detection.

This script:
1. Scans workspace with dual filtering (whitelist + blacklist)
2. Checks Qdrant to find already-indexed files
3. Detects which files are new or modified
4. Only indexes/reindexes changed files
5. Removes stale files from Qdrant
6. Stores metadata for future incremental runs
"""

import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Any
import logging
import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import httpx
import tiktoken

# Handle both direct execution and module import
if __name__ == "__main__":
    # When run directly, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from indexer.file_filters import EXCLUSION_PATTERNS, VALID_PATTERNS, should_keep_file
    from indexer.metadata_config import get_metadata_path
    from indexer.smart_chunkers import get_smart_chunker
    from indexer.sparse_vector_gen import generate_sparse_vector, get_global_generator
    from indexer.local_embeddings import generate_local_embeddings, get_local_model
else:
    # When imported as module, use relative imports
    from .file_filters import EXCLUSION_PATTERNS, VALID_PATTERNS, should_keep_file
    from .metadata_config import get_metadata_path
    from .smart_chunkers import get_smart_chunker
    from .sparse_vector_gen import generate_sparse_vector, get_global_generator
    from .local_embeddings import generate_local_embeddings, get_local_model

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, SparseVector, SparseVectorParams, SparseIndexParams

# Load configuration
script_dir = Path(__file__).parent
config_file = script_dir / 'db_indexer_config.env'
if config_file.exists():
    load_dotenv(config_file)

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level),
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("incremental_indexer")

# Configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))

# Embedding provider configuration
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local').lower()  # 'azure' or 'local'
LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'gte-qwen')

# Azure OpenAI configuration (only used when EMBEDDING_PROVIDER=azure)
AZURE_EMBEDDING_API_URL = os.getenv('AZURE_EMBEDDING_API_URL',
                                    'https://genai-gateway.azure-api.net/v1/embeddings')
AZURE_EMBEDDING_MODEL = os.getenv('AZURE_EMBEDDING_MODEL', 'text-embedding-3-large')
AZURE_EMBEDDING_DIMENSIONS = int(os.getenv('AZURE_EMBEDDING_DIMENSIONS', '3072'))
AZURE_API_KEY = os.getenv('AI_API_KEY', '')

# Determine embedding dimensions based on provider
if EMBEDDING_PROVIDER == 'local':
    if LOCAL_EMBEDDING_MODEL == 'small':
        EMBEDDING_DIMENSIONS = 384
    elif LOCAL_EMBEDDING_MODEL == 'e5-large':
        EMBEDDING_DIMENSIONS = 1024
    elif LOCAL_EMBEDDING_MODEL == 'gte-qwen':
        EMBEDDING_DIMENSIONS = 1536
    else:
        EMBEDDING_DIMENSIONS = 768
    logger.info(f"Using local embeddings: {LOCAL_EMBEDDING_MODEL} model ({EMBEDDING_DIMENSIONS} dims)")
else:
    EMBEDDING_DIMENSIONS = AZURE_EMBEDDING_DIMENSIONS
    logger.info(f"Using Azure embeddings: {AZURE_EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dims)")

# Chunking configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '300'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
USE_SMART_CHUNKING = os.getenv('USE_SMART_CHUNKING', 'true').lower() in ('true', '1', 'yes')

# Sparse vector configuration
USE_SPARSE_VECTORS = os.getenv('USE_SPARSE_VECTORS', 'false').lower() in ('true', '1', 'yes')
SPARSE_ONLY_MODE = os.getenv('SPARSE_ONLY_MODE', 'false').lower() in ('true', '1', 'yes')
logger.info(f"Sparse vectors: {'ENABLED' if USE_SPARSE_VECTORS else 'DISABLED'}")
if USE_SPARSE_VECTORS:
    logger.info(f"Mode: {'SPARSE ONLY' if SPARSE_ONLY_MODE else 'HYBRID (dense + sparse)'}")

# Token estimation for Azure
MAX_EMBEDDING_TOKENS = 8192
token_encoding = None
if AZURE_API_KEY:
    try:
        token_encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(f"Loaded tiktoken encoding for token estimation (max: {MAX_EMBEDDING_TOKENS} tokens)")
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding: {e}, will use character-based estimation")


class MetadataStore:
    """Stores and manages file metadata for incremental indexing."""

    def __init__(self, metadata_file: Path):
        self.metadata_file = metadata_file
        self.data = self._load()

    def _load(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {'files': {}, 'last_index': None}

    def save(self):
        """Save metadata to file."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_file_metadata(self, rel_path: str) -> Optional[Dict]:
        """Get metadata for a file."""
        return self.data['files'].get(rel_path)

    def update_file_metadata(self, rel_path: str, mtime: float, size: int, hash_val: str, chunks: int):
        """Update metadata for a file."""
        # Ensure 'files' key exists
        if 'files' not in self.data:
            self.data['files'] = {}

        self.data['files'][rel_path] = {
            'mtime': mtime,
            'size': size,
            'hash': hash_val,
            'chunks': chunks,
            'indexed_at': datetime.utcnow().isoformat()
        }

    def remove_file_metadata(self, rel_path: str):
        """Remove metadata for a file."""
        if 'files' in self.data:
            self.data['files'].pop(rel_path, None)

    def mark_index_complete(self):
        """Mark indexing as complete."""
        self.data['last_index'] = datetime.utcnow().isoformat()


def should_index_file(file_path: Path, workspace_root: Path) -> Tuple[bool, str]:
    """Dual filtering: blacklist + whitelist. Wrapper around should_keep_file."""
    try:
        rel_path = str(file_path.relative_to(workspace_root))
    except ValueError:
        return False, "outside workspace"

    return should_keep_file(rel_path)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""


def get_qdrant_files(collection_name: str, workspace_root: Path) -> Set[str]:
    """Get set of files already in Qdrant."""
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            return set()

        files = set()
        offset = None

        while True:
            records, offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            if not records:
                break

            for record in records:
                file_path = record.payload.get('filePath')
                if file_path:
                    file_path_obj = Path(file_path)
                    for possible_root in ['/workspaces/spfr', '/workspaces/spfr.orig', str(workspace_root.resolve())]:
                        try:
                            rel_path = str(file_path_obj.relative_to(possible_root))
                            files.add(rel_path)
                            break
                        except ValueError:
                            continue

            if offset is None:
                break

        return files

    except Exception as e:
        logger.warning(f"Could not query Qdrant: {e}")
        return set()


def categorize_files(workspace_root: Path, directories: List[str], metadata_store: MetadataStore,
                     qdrant_files: Set[str]) -> Dict[str, List]:
    """Categorize files as new, modified, or unchanged."""
    extensions = {'.pc', '.sql', '.sh', '.pkg', '.pkb', '.h', '.c', '.cpp', '.hpp',
                  '.py', '.pyx', '.pyi', '.js', '.jsx', '.ts', '.tsx', '.mjs',
                  '.java', '.kt', '.scala', '.go', '.rs', '.rb', '.php', '.swift',
                  '.vue', '.svelte', '.toml', '.yaml', '.yml', '.json', '.xml',
                  '.md', '.rst', '.cfg', '.ini', '.conf'}

    result = {
        'new': [],
        'modified': [],
        'unchanged': [],
        'excluded': []
    }

    scanned_files = set()

    for dir_name in directories:
        dir_path = workspace_root / dir_name
        if not dir_path.exists():
            continue

        for file_path in dir_path.rglob('*'):
            if not file_path.is_file() or file_path.suffix not in extensions:
                continue

            # Apply dual filtering
            should_index, reason = should_index_file(file_path, workspace_root)

            if not should_index:
                result['excluded'].append((file_path, reason))
                continue

            try:
                rel_path = str(file_path.relative_to(workspace_root))
                scanned_files.add(rel_path)

                # Check if in Qdrant
                if rel_path not in qdrant_files:
                    result['new'].append(file_path)
                    continue

                # Check if modified
                current_mtime = file_path.stat().st_mtime
                current_size = file_path.stat().st_size

                metadata = metadata_store.get_file_metadata(rel_path)

                if not metadata:
                    # In Qdrant but no metadata - treat as modified
                    result['modified'].append(file_path)
                    continue

                # Quick check: size changed
                if metadata.get('size') != current_size:
                    result['modified'].append(file_path)
                    continue

                # mtime check
                if metadata.get('mtime') and current_mtime > metadata['mtime']:
                    # Verify with hash
                    if metadata.get('hash'):
                        current_hash = compute_file_hash(file_path)
                        if current_hash != metadata['hash']:
                            result['modified'].append(file_path)
                        else:
                            result['unchanged'].append(file_path)
                    else:
                        result['modified'].append(file_path)
                else:
                    result['unchanged'].append(file_path)

            except Exception as e:
                logger.debug(f"Error categorizing {file_path}: {e}")

    # Find stale files (in Qdrant but not on disk)
    result['stale'] = [rel_path for rel_path in qdrant_files if rel_path not in scanned_files]

    return result


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using tiktoken.
    Falls back to character-based estimation if tiktoken not available.
    """
    if token_encoding is not None:
        try:
            return len(token_encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token encoding failed: {e}, using character estimation")

    # Fallback: rough estimation (1 token ≈ 4 characters for code)
    return len(text) // 4


def should_pre_split_chunk(chunk_text: str) -> bool:
    """
    Check if a chunk should be pre-split before sending to Azure.
    Returns True if estimated tokens exceed the context window.
    """
    estimated_tokens = estimate_tokens(chunk_text)
    if estimated_tokens > MAX_EMBEDDING_TOKENS:
        logger.info(f"Pre-split check: {estimated_tokens} tokens > {MAX_EMBEDDING_TOKENS} limit")
        return True
    return False


def chunk_file_simple(file_path: Path) -> List[Dict[str, Any]]:
    """
    Simple line-based chunking strategy.
    Returns list of chunks with metadata.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []

    if not lines:
        return []

    chunks = []
    total_lines = len(lines)
    chunk_index = 0

    start_line = 0
    while start_line < total_lines:
        end_line = min(start_line + CHUNK_SIZE, total_lines)

        # Extract chunk content
        chunk_lines = lines[start_line:end_line]
        chunk_text = ''.join(chunk_lines)

        # Skip empty chunks
        if not chunk_text.strip():
            start_line += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        # Determine file type
        file_type = 'unknown'
        if file_path.suffix == '.pc':
            file_type = 'proc'
        elif file_path.suffix == '.sql':
            if '/pkg/' in str(file_path):
                file_type = 'plsql'
            elif '/ddl/' in str(file_path):
                file_type = 'ddl'
            elif '/dml/' in str(file_path):
                file_type = 'dml'
            else:
                file_type = 'sql'
        elif file_path.suffix == '.sh':
            file_type = 'shell'
        elif file_path.suffix in ('.py', '.pyx', '.pyi'):
            file_type = 'python'
        elif file_path.suffix in ('.js', '.jsx', '.mjs', '.cjs'):
            file_type = 'javascript'
        elif file_path.suffix in ('.ts', '.tsx'):
            file_type = 'typescript'
        elif file_path.suffix in ('.java', '.kt'):
            file_type = 'java'
        elif file_path.suffix in ('.go',):
            file_type = 'go'
        elif file_path.suffix in ('.rs',):
            file_type = 'rust'
        elif file_path.suffix in ('.c', '.h'):
            file_type = 'c'
        elif file_path.suffix in ('.cpp', '.hpp', '.cc', '.cxx'):
            file_type = 'cpp'
        elif file_path.suffix in ('.vue',):
            file_type = 'vue'
        elif file_path.suffix in ('.toml', '.yaml', '.yml', '.json', '.xml', '.cfg', '.ini', '.conf'):
            file_type = 'config'
        elif file_path.suffix in ('.md', '.rst', '.txt'):
            file_type = 'docs'

        chunks.append({
            'file_path': str(file_path),
            'code_chunk': chunk_text,
            'start_line': start_line + 1,
            'end_line': end_line,
            'file_type': file_type,
            'chunk_index': chunk_index,
            'total_chunks': -1
        })

        chunk_index += 1
        start_line += CHUNK_SIZE - CHUNK_OVERLAP

    # Update total_chunks
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)

    return chunks


async def generate_embeddings_azure(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Azure OpenAI API."""
    if not AZURE_API_KEY:
        raise ValueError("AI_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                AZURE_EMBEDDING_API_URL,
                headers={
                    "Authorization": f"Bearer {AZURE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": AZURE_EMBEDDING_MODEL,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            embeddings = [item['embedding'] for item in data['data']]
            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Azure API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


def generate_embeddings_local_sync(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local BGE model (synchronous)."""
    try:
        embeddings_np = generate_local_embeddings(
            texts,
            model_name=LOCAL_EMBEDDING_MODEL,
            normalize=True,
            batch_size=32,
            show_progress=False
        )
        # Convert numpy array to list of lists
        return embeddings_np.tolist()
    except Exception as e:
        logger.error(f"Error generating local embeddings: {e}")
        raise


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using configured provider."""
    if EMBEDDING_PROVIDER == 'local':
        # Run local embeddings in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_embeddings_local_sync, texts)
    else:
        return await generate_embeddings_azure(texts)


async def index_file(file_path: Path, workspace_root: Path, collection_name: str,
                    qdrant_client: QdrantClient, metadata_store: MetadataStore) -> int:
    """Index a single file. Returns number of chunks indexed."""
    try:
        rel_path = str(file_path.relative_to(workspace_root))
        logger.info(f"Indexing: {rel_path}")

        # Chunk file using smart chunker if available
        smart_chunker = get_smart_chunker(file_path) if USE_SMART_CHUNKING else None

        if smart_chunker:
            logger.debug(f"Using smart chunker for {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                chunks = smart_chunker(content, str(file_path))
            except Exception as e:
                logger.error(f"Smart chunker failed for {file_path}: {e}, falling back")
                chunks = chunk_file_simple(file_path)
        else:
            chunks = chunk_file_simple(file_path)

        if not chunks:
            logger.warning(f"No chunks created for {file_path}")
            return 0

        # Pre-process chunks to handle oversized ones
        processed_chunks = []
        for chunk in chunks:
            chunk_text = chunk['code_chunk']

            if should_pre_split_chunk(chunk_text):
                estimated_tokens = estimate_tokens(chunk_text)
                logger.warning(f"Chunk too large ({estimated_tokens} tokens), splitting")

                # Split by lines
                lines = chunk_text.split('\n')
                for max_chars in [5000, 3000, 2000]:
                    pieces = []
                    current_piece = []
                    current_size = 0

                    for line in lines:
                        line_size = len(line) + 1
                        if current_size + line_size > max_chars and current_piece:
                            piece_text = '\n'.join(current_piece)
                            if not should_pre_split_chunk(piece_text):
                                piece_chunk = chunk.copy()
                                piece_chunk['code_chunk'] = piece_text
                                pieces.append(piece_chunk)
                                current_piece = [line]
                                current_size = line_size
                            else:
                                break
                        else:
                            current_piece.append(line)
                            current_size += line_size

                    if current_piece and pieces:
                        piece_text = '\n'.join(current_piece)
                        if not should_pre_split_chunk(piece_text):
                            piece_chunk = chunk.copy()
                            piece_chunk['code_chunk'] = piece_text
                            pieces.append(piece_chunk)

                    if pieces and all(not should_pre_split_chunk(p['code_chunk']) for p in pieces):
                        processed_chunks.extend(pieces)
                        logger.info(f"Split into {len(pieces)} pieces")
                        break
                else:
                    logger.error(f"Failed to split chunk, skipping")
            else:
                processed_chunks.append(chunk)

        # Get embeddings (dense and/or sparse)
        chunk_texts = [c['code_chunk'] for c in processed_chunks]

        # Generate dense embeddings (if not sparse-only mode)
        embeddings = []
        if not SPARSE_ONLY_MODE:
            try:
                embeddings = await get_embeddings(chunk_texts)
                if not embeddings:
                    logger.error(f"No dense embeddings generated for {file_path}")
                    return 0
            except Exception as e:
                logger.error(f"Failed to generate dense embeddings: {e}")
                if EMBEDDING_PROVIDER == 'azure' and not AZURE_API_KEY:
                    logger.error("Azure provider selected but AI_API_KEY not set")
                return 0

        # Generate sparse vectors (if enabled)
        sparse_vectors = []
        if USE_SPARSE_VECTORS:
            logger.debug(f"Generating sparse vectors for {len(processed_chunks)} chunks")
            for chunk in processed_chunks:
                metadata = {
                    'language': chunk.get('file_type', 'unknown'),
                    'file_type': 'table_definition' if '/ddl/' in str(file_path) else 'source'
                }
                sparse_vec = generate_sparse_vector(chunk['code_chunk'], metadata)
                sparse_vectors.append(sparse_vec)

            if not sparse_vectors:
                logger.error(f"No sparse vectors generated for {file_path}")
                return 0

        # Ensure we have at least one type of vector
        if not embeddings and not sparse_vectors:
            logger.error(f"No vectors generated for {file_path}")
            return 0

        # Compute file hash
        file_hash = compute_file_hash(file_path)
        file_mtime = file_path.stat().st_mtime
        file_size = file_path.stat().st_size

        # Create points with appropriate vector types
        points = []
        for i, chunk in enumerate(processed_chunks):
            point_id = hashlib.md5(f"{rel_path}:{i}".encode()).hexdigest()

            # Prepare vector data based on mode
            if USE_SPARSE_VECTORS and SPARSE_ONLY_MODE:
                # Sparse-only mode: use named sparse vectors
                vector_data = {
                    "sparse": SparseVector(
                        indices=list(sparse_vectors[i].keys()),
                        values=list(sparse_vectors[i].values())
                    )
                }
            elif USE_SPARSE_VECTORS and embeddings:
                # Hybrid mode: both dense and sparse
                vector_data = {
                    "dense": embeddings[i],
                    "sparse": SparseVector(
                        indices=list(sparse_vectors[i].keys()),
                        values=list(sparse_vectors[i].values())
                    )
                }
            else:
                # Dense-only mode (legacy)
                vector_data = embeddings[i]

            points.append(PointStruct(
                id=point_id,
                vector=vector_data,
                payload={
                    'filePath': str(file_path),
                    'codeChunk': chunk['code_chunk'],
                    'contentHash': file_hash,
                    'indexedAt': datetime.utcnow().isoformat(),
                    'fileMtime': file_mtime,
                    'fileSize': file_size,
                    'chunkIndex': i,
                    'startLine': chunk.get('start_line', 1),
                    'endLine': chunk.get('end_line', 1),
                    'fileType': chunk.get('file_type', 'unknown'),
                    'functionName': chunk.get('function_name', '')
                }
            ))

        # Upsert to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        # Update metadata
        metadata_store.update_file_metadata(rel_path, file_mtime, file_size, file_hash, len(points))

        logger.info(f"✓ Indexed {rel_path}: {len(points)} chunks")
        return len(points)

    except Exception as e:
        logger.error(f"Error indexing {file_path}: {e}")
        return 0


def remove_stale_files(stale_files: List[str], collection_name: str,
                      qdrant_client: QdrantClient, metadata_store: MetadataStore):
    """Remove stale files from Qdrant."""
    if not stale_files:
        return

    logger.info(f"Removing {len(stale_files)} stale files from Qdrant...")

    for rel_path in stale_files:
        try:
            # Delete all points for this file
            from qdrant_client.models import Filter, FieldCondition, MatchText

            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key='filePath',
                            match=MatchText(text=rel_path)
                        )
                    ]
                )
            )

            # Remove from metadata
            metadata_store.remove_file_metadata(rel_path)

        except Exception as e:
            logger.error(f"Error removing {rel_path}: {e}")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Incremental code indexer')
    parser.add_argument('workspace_root', type=str, help='Workspace root directory')
    parser.add_argument('--collection', type=str, default='spfr-application-code')
    parser.add_argument('--dirs', type=str, nargs='+',
                       default=['pc', 'pkg', 'ddl', 'dml', 'batch', 'trg', 'typ', 'inc', 'cinc', 'cpc'])
    parser.add_argument('--mode', type=str, choices=['sparse', 'dense', 'hybrid'],
                       help='Indexing mode: sparse (BM25, $0), dense (embeddings, API cost), hybrid (both)')
    parser.add_argument('--force', action='store_true', help='Force reindex all files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--metadata-file', type=str, help='Override default metadata file location')

    args = parser.parse_args()

    # Override sparse vector configuration if --mode is specified
    global USE_SPARSE_VECTORS, SPARSE_ONLY_MODE
    if args.mode:
        if args.mode == 'sparse':
            USE_SPARSE_VECTORS = True
            SPARSE_ONLY_MODE = True
            logger.info(f"🎯 Mode override: SPARSE ONLY (from --mode argument)")
        elif args.mode == 'dense':
            USE_SPARSE_VECTORS = False
            SPARSE_ONLY_MODE = False
            logger.info(f"🎯 Mode override: DENSE ONLY (from --mode argument)")
        elif args.mode == 'hybrid':
            USE_SPARSE_VECTORS = True
            SPARSE_ONLY_MODE = False
            logger.info(f"🎯 Mode override: HYBRID (from --mode argument)")

    workspace_path = Path(args.workspace_root).resolve()
    # Use centralized metadata path unless explicitly overridden
    metadata_file = Path(args.metadata_file) if args.metadata_file else get_metadata_path(args.workspace_root)

    logger.info("=" * 80)
    logger.info("🚀 INCREMENTAL CODE INDEXER")
    logger.info("=" * 80)
    logger.info(f"Workspace: {workspace_path}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Metadata: {metadata_file}")
    logger.info(f"Execution: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"Indexing Mode: {'SPARSE ONLY ($0)' if SPARSE_ONLY_MODE else 'HYBRID (dense+sparse)' if USE_SPARSE_VECTORS else 'DENSE ONLY (embeddings)'}")
    logger.info("")

    # Initialize
    logger.info("🔧 Initializing metadata store...")
    metadata_store = MetadataStore(metadata_file)
    logger.info("   ✓ Metadata store ready")

    logger.info("🔧 Connecting to Qdrant...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info(f"   ✓ Connected to {QDRANT_HOST}:{QDRANT_PORT}")

    # Check/create collection with correct vector configuration
    logger.info("📊 Checking Qdrant collection...")
    try:
        qdrant_client.get_collection(args.collection)
        logger.info(f"   ✓ Collection '{args.collection}' exists")
    except Exception as e:
        if "Not found" in str(e) or "doesn't exist" in str(e):
            # Determine vector configuration based on mode
            if USE_SPARSE_VECTORS and SPARSE_ONLY_MODE:
                logger.info(f"   Creating SPARSE-ONLY collection '{args.collection}'...")
                qdrant_client.create_collection(
                    collection_name=args.collection,
                    vectors_config={},  # Empty for sparse-only
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams()
                        )
                    }
                )
            elif USE_SPARSE_VECTORS:
                logger.info(f"   Creating HYBRID collection '{args.collection}' (dense + sparse)...")
                logger.info(f"   Using {EMBEDDING_PROVIDER} embeddings with {EMBEDDING_DIMENSIONS} dimensions")
                qdrant_client.create_collection(
                    collection_name=args.collection,
                    vectors_config={
                        "dense": VectorParams(
                            size=EMBEDDING_DIMENSIONS,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams()
                        )
                    }
                )
            else:
                logger.info(f"   Creating DENSE-ONLY collection '{args.collection}' with {EMBEDDING_DIMENSIONS} dimensions...")
                logger.info(f"   Using {EMBEDDING_PROVIDER} embeddings")
                qdrant_client.create_collection(
                    collection_name=args.collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
            logger.info(f"   ✓ Collection '{args.collection}' created successfully")
        else:
            logger.error(f"   ✗ Error checking collection: {e}")
            raise

    # Get Qdrant files
    qdrant_files = get_qdrant_files(args.collection, workspace_path)
    logger.info(f"   Files in Qdrant: {len(qdrant_files)}")
    logger.info("")

    # Categorize files
    logger.info("🔎 Analyzing files...")
    categories = categorize_files(workspace_path, args.dirs, metadata_store, qdrant_files)

    logger.info(f"   New files: {len(categories['new'])}")
    logger.info(f"   Modified files: {len(categories['modified'])}")
    logger.info(f"   Unchanged files: {len(categories['unchanged'])}")
    logger.info(f"   Stale in Qdrant: {len(categories['stale'])}")
    logger.info(f"   Excluded files: {len(categories['excluded'])}")
    logger.info("")

    if args.dry_run:
        logger.info("🔍 DRY RUN - No changes will be made")
        return

    # Index new and modified files
    to_index = categories['new'] + categories['modified']

    if to_index:
        logger.info(f"📝 Indexing {len(to_index)} files...")

        total_chunks = 0
        for i, file_path in enumerate(to_index, 1):
            rel_path = file_path.relative_to(workspace_path)
            logger.info(f"   [{i}/{len(to_index)}] {rel_path}")

            chunks = asyncio.run(index_file(file_path, workspace_path, args.collection, qdrant_client, metadata_store))
            total_chunks += chunks

        logger.info(f"   ✅ Indexed {total_chunks} chunks")
        logger.info("")

    # Remove stale files
    if categories['stale']:
        remove_stale_files(categories['stale'], args.collection, qdrant_client, metadata_store)
        logger.info("")

    # Save metadata
    metadata_store.mark_index_complete()
    metadata_store.save()

    logger.info("=" * 80)
    logger.info("✅ INDEXING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
