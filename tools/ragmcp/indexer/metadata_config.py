"""
Centralized metadata configuration for incremental indexing.
All scripts should import metadata paths from here.
"""

from pathlib import Path

# Metadata storage location relative to workspace root
METADATA_DIR = ".indexing_metadata"
METADATA_FILENAME = "indexing_metadata.json"

def get_metadata_path(workspace_root: str, lane_slug: str | None = None) -> Path:
    """
    Get the full path to the metadata file.

    Args:
        workspace_root: Path to workspace root directory
        lane_slug: Embedding-lane slug (provider-model-dim) for per-lane
            sidecars (P2 lanes). None → the legacy filename, used by
            sparse-only indexing (model-independent) and older runs.

    Returns:
        Path object pointing to metadata file
    """
    workspace_path = Path(workspace_root).resolve()
    metadata_dir = workspace_path / METADATA_DIR

    # Ensure directory exists
    metadata_dir.mkdir(parents=True, exist_ok=True)

    filename = f"indexing_metadata.{lane_slug}.json" if lane_slug else METADATA_FILENAME
    return metadata_dir / filename
