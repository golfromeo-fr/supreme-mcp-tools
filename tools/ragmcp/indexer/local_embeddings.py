#!/usr/bin/env python3
"""
Local CPU-friendly embeddings using BGE models.
No GPU required, works offline, zero API cost.

This module provides semantic search capabilities without relying on external APIs,
making it ideal for:
- Offline/air-gapped environments
- Cost-sensitive deployments
- Survival when cloud API access is lost

Models used: BAAI/bge-m3 (default), BAAI/bge-base-en-v1.5 (fast testing)
- Downloads automatically on first use
- Cached locally for future use
- CPU-optimized (no GPU required)
"""

from sentence_transformers import SentenceTransformer
from typing import Any
import logging
import os
import numpy as np

logger = logging.getLogger("local_embeddings")

# Global model cache to avoid reloading
_model_cache: dict[str, SentenceTransformer] = {}

# Recommended models (ranked by quality vs speed tradeoff)
MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "base": "BAAI/bge-base-en",
}

DEFAULT_MODEL = "bge-m3"


def get_local_model(
    model_name: str | None = None,
    cache_folder: str | None = None
) -> SentenceTransformer:
    """
    Get or load local embedding model (cached in memory).

    Args:
        model_name: Model identifier ('base', 'small', or full HuggingFace name)
        cache_folder: Directory to cache downloaded models (default: ~/.cache/torch)

    Returns:
        Loaded SentenceTransformer model

    Example:
        >>> model = get_local_model('base')
        >>> embeddings = model.encode(['hello world'])
    """
    # Resolve model name from alias or use directly
    if model_name is None:
        model_name = DEFAULT_MODEL

    resolved_name = MODELS.get(model_name, model_name)

    # Check cache first
    if resolved_name in _model_cache:
        logger.debug(f"Using cached model: {resolved_name}")
        return _model_cache[resolved_name]

    # Load model
    logger.info(f"Loading local embedding model: {resolved_name}")
    logger.info("First-time download may take a few minutes...")

    try:
        trust_code = False
        
        try:
            model = SentenceTransformer(
                resolved_name,
                cache_folder=cache_folder,
                device='cpu',
                local_files_only=True,
                trust_remote_code=trust_code,
            )
            logger.info(f"✓ Loaded from cache: {resolved_name}")
        except Exception as cache_error:
            logger.info(f"Model not in cache, downloading from HuggingFace...")
            old_offline = os.environ.pop('HF_HUB_OFFLINE', None)
            old_transformers = os.environ.pop('TRANSFORMERS_OFFLINE', None)
            try:
                model = SentenceTransformer(
                    resolved_name,
                    cache_folder=cache_folder,
                    device='cpu',
                    local_files_only=False,
                    trust_remote_code=trust_code,
                )
            finally:
                if old_offline is not None:
                    os.environ['HF_HUB_OFFLINE'] = old_offline
                if old_transformers is not None:
                    os.environ['TRANSFORMERS_OFFLINE'] = old_transformers
            logger.info(f"✓ Downloaded and loaded: {resolved_name}")

        _model_cache[resolved_name] = model
        logger.info(f"  Dimensions: {model.get_sentence_embedding_dimension()}")
        logger.info(f"  Max sequence length: {model.max_seq_length}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model {resolved_name}: {e}")
        raise


def generate_local_embeddings(
    texts: list[str],
    model_name: str | None = None,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings using local CPU model.

    Args:
        texts: List of text chunks to embed
        model_name: Model to use ('base', 'small', or full name)
        normalize: Whether to L2-normalize embeddings (recommended for similarity)
        batch_size: Number of texts to process at once
        show_progress: Show progress bar for large batches

    Returns:
        numpy array of embeddings (shape: [len(texts), dimensions])

    Example:
        >>> texts = ["function calculate_price", "table STOMVT"]
        >>> embeddings = generate_local_embeddings(texts)
        >>> embeddings.shape
        (2, 768)
    """
    if not texts:
        return np.array([])

    model = get_local_model(model_name)

    logger.debug(f"Generating embeddings for {len(texts)} texts")

    try:
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        logger.debug(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


def generate_local_embedding(
    text: str,
    model_name: str | None = None,
    normalize: bool = True
) -> list[float]:
    """
    Generate a single embedding (convenience wrapper).

    Args:
        text: Single text to embed
        model_name: Model to use
        normalize: Whether to L2-normalize

    Returns:
        List of floats representing the embedding

    Example:
        >>> embedding = generate_local_embedding("calculate sales price")
        >>> len(embedding)
        768
    """
    embeddings = generate_local_embeddings([text], model_name, normalize)
    return embeddings[0].tolist()


def get_model_info(model_name: str | None = None) -> dict[str, Any]:
    """
    Get information about a model without loading it fully.

    Args:
        model_name: Model identifier

    Returns:
        Dictionary with model metadata
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    resolved_name = MODELS.get(model_name, model_name)

    # Try to get from cache first
    if resolved_name in _model_cache:
        model = _model_cache[resolved_name]
        return {
            "name": resolved_name,
            "dimensions": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
            "cached": True
        }

    # Return known info without loading
    info = {
        "name": resolved_name,
        "cached": False
    }

    if model_name == "bge-m3":
        info.update({
            "dimensions": 1024,
            "size_mb": 1200,
            "speed": "medium",
            "quality": "excellent",
            "multilingual": True,
            "hybrid": True
        })
    elif model_name == "base":
        info.update({
            "dimensions": 768,
            "size_mb": 300,
            "speed": "fast",
            "quality": "good"
        })

    return info


def clear_model_cache():
    """Clear all cached models from memory."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")


# Convenience function for testing
def test_local_embeddings():
    """Quick test to verify local embeddings work."""
    print("Testing local embeddings...")

    # Test with small model for speed
    test_texts = [
        "calculate sales price with discount",
        "STOMVT table operations",
        "function RecupClasse from PARPOSTES"
    ]

    print(f"\nGenerating embeddings for {len(test_texts)} test texts...")
    embeddings = generate_local_embeddings(test_texts, model_name="small")

    print(f"✓ Success! Generated embeddings shape: {embeddings.shape}")
    print(f"  Model: {MODELS['small']}")
    print(f"  Dimensions: {embeddings.shape[1]}")
    print(f"  Example embedding (first 5 dims): {embeddings[0][:5]}")

    # Test similarity
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\nSimilarity test:")
    print(f"  Text 0 vs Text 1: {sim_01:.3f}")
    print(f"  Text 0 vs Text 2: {sim_02:.3f}")
    print("\n✓ Local embeddings working correctly!")

    return True


if __name__ == "__main__":
    # Run test when executed directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_local_embeddings()

