"""
Tests for ArtifactStore (S3/MinIO + local fallback).

Focus: LOW-8 — S3 exists/load/get_metadata must distinguish "key not found"
(404 -> False/None, unchanged) from network/storage errors (ArtifactStoreError).
All S3 failures are simulated with mocked clients; no real S3 is touched.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError, EndpointConnectionError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.shared.artifact_store import ArtifactStore, ArtifactStoreError


def _client_error(status: int, code: str = "404", op: str = "HeadObject") -> ClientError:
    """Build a botocore ClientError with the given HTTP status code."""
    return ClientError(
        {
            "Error": {"Code": code, "Message": "simulated failure"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        op,
    )


def _s3_store(client: MagicMock) -> ArtifactStore:
    """ArtifactStore wired to a fake S3 client (no boto3 init, no local dir use)."""
    store = ArtifactStore(local_fallback=True, local_dir=tempfile.mkdtemp())
    store._client = client
    store._initialized = True
    return store


def _run(coro):
    """Run a coroutine on a fresh loop (see tests/test_bug_fixes_low.py)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestExistsS3(unittest.TestCase):
    """LOW-8: exists() must separate 404 from network/storage errors."""

    def test_true_when_present(self):
        client = MagicMock()
        store = _s3_store(client)
        self.assertTrue(_run(store.exists("memories/abc/snapshot.txt")))
        client.head_object.assert_called_once()

    def test_false_on_404(self):
        client = MagicMock()
        client.head_object.side_effect = _client_error(404)
        store = _s3_store(client)
        self.assertFalse(_run(store.exists("memories/abc/snapshot.txt")))

    def test_raises_on_non404_client_error(self):
        # 500 (throttle/server) — must not be reported as "artifact gone"
        client = MagicMock()
        client.head_object.side_effect = _client_error(500, code="InternalError")
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.exists("memories/abc/snapshot.txt"))

    def test_raises_on_403_auth_error(self):
        client = MagicMock()
        client.head_object.side_effect = _client_error(403, code="AccessDenied")
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.exists("memories/abc/snapshot.txt"))

    def test_raises_on_network_error(self):
        # Non-ClientError transport failure (connection refused/timeout)
        client = MagicMock()
        client.head_object.side_effect = EndpointConnectionError(
            endpoint_url="http://127.0.0.1:9"
        )
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.exists("memories/abc/snapshot.txt"))


class TestLoadS3(unittest.TestCase):
    """LOW-8: load() must separate 404 (None) from network/storage errors."""

    def _client_with_body(self, body: bytes) -> MagicMock:
        client = MagicMock()
        fake_body = MagicMock()
        fake_body.read.return_value = body
        client.get_object.return_value = {"Body": fake_body}
        return client

    def test_returns_data_when_present(self):
        store = _s3_store(self._client_with_body(b"payload"))
        self.assertEqual(_run(store.load("memories/abc/snapshot.txt")), b"payload")

    def test_none_on_404(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error(404, code="NoSuchKey", op="GetObject")
        store = _s3_store(client)
        self.assertIsNone(_run(store.load("memories/abc/snapshot.txt")))

    def test_raises_on_non404_client_error(self):
        client = MagicMock()
        client.get_object.side_effect = _client_error(500, code="InternalError", op="GetObject")
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.load("memories/abc/snapshot.txt"))

    def test_raises_on_network_error(self):
        client = MagicMock()
        client.get_object.side_effect = EndpointConnectionError(
            endpoint_url="http://127.0.0.1:9"
        )
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.load("memories/abc/snapshot.txt"))


class TestGetMetadataS3(unittest.TestCase):
    """LOW-8: get_metadata() must separate 404 (None) from network/storage errors."""

    def test_returns_metadata_when_present(self):
        client = MagicMock()
        client.head_object.return_value = {
            "ContentLength": 7,
            "ContentType": "text/plain",
            "Metadata": {"owner": "test"},
        }
        store = _s3_store(client)
        meta = _run(store.get_metadata("memories/abc/snapshot.txt"))
        self.assertEqual(meta["size"], 7)
        self.assertEqual(meta["metadata"], {"owner": "test"})

    def test_none_on_404(self):
        client = MagicMock()
        client.head_object.side_effect = _client_error(404)
        store = _s3_store(client)
        self.assertIsNone(_run(store.get_metadata("memories/abc/snapshot.txt")))

    def test_raises_on_network_error(self):
        client = MagicMock()
        client.head_object.side_effect = EndpointConnectionError(
            endpoint_url="http://127.0.0.1:9"
        )
        store = _s3_store(client)
        with self.assertRaises(ArtifactStoreError):
            _run(store.get_metadata("memories/abc/snapshot.txt"))


class TestCallerHandlesStoreError(unittest.TestCase):
    """
    LOW-8: a caller can now tell "artifact gone" (False) from
    "artifact store unreachable" (ArtifactStoreError) and degrade gracefully.
    """

    @staticmethod
    async def artifact_visible(store: ArtifactStore, key: str):
        """Representative caller: True=present, False=confirmed gone,
        None=store unreachable (caller logs and degrades instead of crashing)."""
        try:
            return await store.exists(key)
        except ArtifactStoreError:
            return None

    def test_caller_distinguishes_gone_from_unreachable(self):
        present = MagicMock()
        gone = MagicMock()
        gone.head_object.side_effect = _client_error(404)
        unreachable = MagicMock()
        unreachable.head_object.side_effect = EndpointConnectionError(
            endpoint_url="http://127.0.0.1:9"
        )

        self.assertIs(True, _run(self.artifact_visible(_s3_store(present), "k")))
        self.assertIs(False, _run(self.artifact_visible(_s3_store(gone), "k")))
        # Network failure no longer masquerades as "artifact gone"
        self.assertIsNone(_run(self.artifact_visible(_s3_store(unreachable), "k")))


class TestDeleteS3Unchanged(unittest.TestCase):
    """LOW-7 fix must keep working after the LOW-8 refactor."""

    def test_delete_false_on_404(self):
        client = MagicMock()
        client.head_object.side_effect = _client_error(404)
        store = _s3_store(client)
        self.assertFalse(_run(store.delete("memories/abc/snapshot.txt")))
        client.delete_object.assert_not_called()


class TestLocalFallbackUnchanged(unittest.TestCase):
    """Local fallback behavior is untouched by the S3 error split."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = ArtifactStore(local_fallback=True, local_dir=self.tmpdir)

    def test_exists_false_then_true_after_save(self):
        self.assertFalse(_run(self.store.exists("roundtrip/key.txt")))
        _run(self.store.save(b"data", "roundtrip/key.txt"))
        self.assertTrue(_run(self.store.exists("roundtrip/key.txt")))

    def test_load_none_for_missing(self):
        self.assertIsNone(_run(self.store.load("roundtrip/missing.txt")))


if __name__ == "__main__":
    unittest.main()
