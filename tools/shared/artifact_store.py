"""
Artifact Store - S3/MinIO client for storing large memory artifacts.

Provides object storage for raw artifacts like file snapshots, diffs, etc.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRef:
    """Reference to a stored artifact."""
    key: str
    size_bytes: int
    stored_at: str
    content_type: str | None = None
    metadata: dict[str, str] | None = None


class ArtifactStoreError(Exception):
    """
    Raised when an artifact operation fails for a reason other than the
    key not existing (network error, auth failure, throttling, ...).

    "Not found" is NOT an error: exists() returns False and load()/
    get_metadata() return None in that case. This exception distinguishes
    "artifact store unreachable" from "artifact gone".
    """


def _is_s3_not_found(e: Exception) -> bool:
    """True if the exception is an S3 ClientError with HTTP status 404."""
    import botocore.exceptions
    return (
        isinstance(e, botocore.exceptions.ClientError)
        and e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
    )


class ArtifactStore:
    """
    S3/MinIO artifact store for large memory blobs.

    Falls back to local filesystem if S3 is not configured.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str = "memory-artifacts",
        local_fallback: bool = True,
        local_dir: str | None = None,
    ):
        self.endpoint = endpoint or os.getenv("S3_ENDPOINT")
        self.access_key = access_key or os.getenv("S3_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("S3_SECRET_KEY")
        self.bucket = bucket
        self.local_fallback = local_fallback
        self.local_dir = local_dir or os.getenv(
            "ARTIFACT_LOCAL_DIR",
            str(Path.home() / ".memorymcp" / "artifacts")
        )
        self._client = None
        self._initialized = False

    def _init_client(self) -> bool:
        """Initialize S3 client if possible."""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if self.endpoint and self.access_key and self.secret_key:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                )
                logger.info(f"S3 artifact store initialized: {self.endpoint}")
                return True
            except ImportError:
                logger.warning("boto3 not installed, falling back to local storage")
            except Exception as e:
                logger.warning(f"S3 init failed, falling back to local: {e}")

        if self.local_fallback:
            Path(self.local_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local artifact storage: {self.local_dir}")
            return False

        return False

    def _local_path(self, key: str) -> Path:
        """Get local filesystem path for a key."""
        path = (Path(self.local_dir) / key).resolve()
        base = Path(self.local_dir).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Key escapes artifact directory: {key}")
        return path

    def _meta_path(self, key: str) -> Path:
        """Get local filesystem path for a key's metadata file."""
        path = self._local_path(key)
        return path.with_suffix(path.suffix + ".meta")

    async def save(
        self,
        data: bytes | str,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ArtifactRef:
        """
        Save artifact to store.

        Args:
            data: Binary or text data
            key: Unique key/path for the artifact
            content_type: MIME type
            metadata: Custom metadata

        Returns:
            ArtifactRef with storage info
        """
        self._init_client()

        if isinstance(data, str):
            data = data.encode("utf-8")

        size = len(data)
        now = datetime.now(timezone.utc).isoformat()

        if self._client:
            try:
                extra_args = {}
                if content_type:
                    extra_args["ContentType"] = content_type
                if metadata:
                    extra_args["Metadata"] = metadata

                self._client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    **extra_args
                )
                logger.debug(f"Saved artifact to S3: {key} ({size} bytes)")
            except Exception as e:
                logger.error(f"S3 save failed: {e}")
                raise
        else:
            # Local fallback
            path = self._local_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Store metadata as .meta file
            meta_path = self._meta_path(key)
            meta = {
                "stored_at": now,
                "size": size,
                "content_type": content_type,
                "metadata": metadata or {},
            }
            meta_path.write_text(json.dumps(meta))

            path.write_bytes(data)
            logger.debug(f"Saved artifact locally: {key} ({size} bytes)")

        return ArtifactRef(
            key=key,
            size_bytes=size,
            stored_at=now,
            content_type=content_type,
            metadata=metadata,
        )

    async def load(self, key: str) -> bytes | None:
        """
        Load artifact from store.

        Args:
            key: Artifact key

        Returns:
            Binary data or None if not found

        Raises:
            ArtifactStoreError: On network/storage failures (not for missing keys)
        """
        self._init_client()

        if self._client:
            try:
                response = self._client.get_object(Bucket=self.bucket, Key=key)
                return response["Body"].read()
            except Exception as e:
                if _is_s3_not_found(e):
                    return None
                logger.error(f"S3 load failed for {key}: {e}")
                raise ArtifactStoreError(f"S3 load failed for {key}: {e}") from e
        else:
            path = self._local_path(key)
            if path.exists():
                return path.read_bytes()
            return None

    async def delete(self, key: str) -> bool:
        """
        Delete artifact from store.

        Args:
            key: Artifact key

        Returns:
            True if deleted, False if not found
        """
        self._init_client()

        if self._client:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
            except Exception as e:
                if _is_s3_not_found(e):
                    return False
                logger.error(f"S3 head_object failed for {key}: {e}")
                return False
            try:
                self._client.delete_object(Bucket=self.bucket, Key=key)
                logger.debug(f"Deleted artifact from S3: {key}")
                return True
            except Exception as e:
                logger.error(f"S3 delete failed for {key}: {e}")
                return False
        else:
            path = self._local_path(key)
            meta_path = self._meta_path(key)
            deleted = False
            if path.exists():
                path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            if deleted:
                logger.debug(f"Deleted artifact locally: {key}")
            return deleted

    async def exists(self, key: str) -> bool:
        """
        Check if artifact exists.

        Raises:
            ArtifactStoreError: On network/storage failures (not for missing keys)
        """
        self._init_client()

        if self._client:
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception as e:
                if _is_s3_not_found(e):
                    return False
                logger.error(f"S3 exists check failed for {key}: {e}")
                raise ArtifactStoreError(f"S3 exists check failed for {key}: {e}") from e
        else:
            return self._local_path(key).exists()

    async def get_metadata(self, key: str) -> dict[str, Any] | None:
        """
        Get artifact metadata.

        Returns:
            Metadata dict or None if not found

        Raises:
            ArtifactStoreError: On network/storage failures (not for missing keys)
        """
        self._init_client()

        if self._client:
            try:
                response = self._client.head_object(Bucket=self.bucket, Key=key)
                return {
                    "size": response.get("ContentLength"),
                    "content_type": response.get("ContentType"),
                    "metadata": response.get("Metadata", {}),
                    "stored_at": response.get("LastModified"),
                }
            except Exception as e:
                if _is_s3_not_found(e):
                    return None
                logger.error(f"S3 get_metadata failed for {key}: {e}")
                raise ArtifactStoreError(f"S3 get_metadata failed for {key}: {e}") from e
        else:
            meta_path = self._meta_path(key)
            if meta_path.exists():
                return json.loads(meta_path.read_text())
            return None


def generate_artifact_key(
    memory_id: str,
    artifact_type: str,
    extension: str = "",
) -> str:
    """
    Generate a deterministic artifact key.

    Args:
        memory_id: Memory UUID
        artifact_type: Type like "snapshot", "diff", "output"
        extension: File extension like ".txt", ".json"

    Returns:
        Key path like "memories/abc123/snapshot.txt"
    """
    return f"memories/{memory_id}/{artifact_type}{extension}"


# Global instance
_artifact_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store