"""
Centralized metadata configuration for incremental indexing.
All scripts should import metadata paths from here.
"""

from pathlib import Path

# Metadata storage location relative to workspace root
METADATA_DIR = ".indexing_metadata"
METADATA_FILENAME = "indexing_metadata.json"

def get_metadata_path(workspace_root: str) -> Path:
    """
    Get the full path to the metadata file.

    Args:
        workspace_root: Path to workspace root directory

    Returns:
        Path object pointing to metadata file
    """
    workspace_path = Path(workspace_root).resolve()
    metadata_dir = workspace_path / METADATA_DIR

    # Ensure directory exists
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return metadata_dir / METADATA_FILENAME
