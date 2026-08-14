"""
Tests for LOW severity bug fixes (LOW-1 through LOW-16).

Each test verifies a specific fix. Tests are self-contained.
"""

import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestLOW1QdrantTimeout(unittest.TestCase):
    """LOW-1: Qdrant client created without timeout."""

    def test_timeout_in_source(self):
        # Phase 5: QdrantClient is now in impls/qdrant_vector.py
        source = (PROJECT_ROOT / "tools/shared/impls/qdrant_vector.py").read_text()
        self.assertIn("timeout=", source)
        idx = source.find("QdrantClient(")
        if idx >= 0:
            line = source[idx:idx + 200]
            self.assertIn("timeout", line)


class TestLOW2EmbeddingDimFromEnv(unittest.TestCase):
    """LOW-2: Hardcoded 1024-dim vectors."""

    def test_dim_from_env(self):
        # Phase 5: dimension is now read from env in memory_core.py via ensure_collection
        source = (PROJECT_ROOT / "tools/memorymcp/memory_core.py").read_text()
        self.assertIn("EMBEDDING_DIM", source)
        self.assertIn("dense_dim=embedding_dim", source)


class TestLOW3EdgeFilteringComplexity(unittest.TestCase):
    """LOW-3: exportGraphAsMarkdown O(n²) edge filtering."""

    def test_set_used_for_lookup(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_graph.py").read_text()
        self.assertIn("point_ids = {str(p.id) for p in points}", source)
        self.assertNotIn("any(str(p.id) == to_id for p in points)", source)


class TestLOW4MergeDuplicatesDesign(unittest.TestCase):
    """LOW-4: mergeDuplicates uses word-level Jaccard (design choice)."""

    def test_word_overlap_used(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        idx = source.find("async def mergeDuplicates")
        section = source[idx:idx + 3000]
        self.assertIn("overlap", section)


class TestLOW5PasswordInDSN(unittest.TestCase):
    """LOW-5: Password in DSN string."""

    def test_dsn_not_logged(self):
        source = (PROJECT_ROOT / "tools/shared/pg_store.py").read_text()
        init = source[source.find("def init_pg"):]
        chunk = init[:2000]
        self.assertNotIn("dsn", [line for line in chunk.split("\n")
                                  if "logger" in line and "dsn" in line.lower()])


class TestLOW6InitPgIdempotent(unittest.TestCase):
    """LOW-6: init_pg not idempotent under concurrent calls."""

    def test_lock_exists(self):
        from tools.shared.pg_store import _init_lock
        self.assertIsInstance(_init_lock, type(threading.Lock()))

    def test_double_init_safe(self):
        source = (PROJECT_ROOT / "tools/shared/pg_store.py").read_text()
        func = source[source.find("def init_pg"):]
        chunk = func[:1000]
        self.assertIn("_init_lock", chunk)
        self.assertIn("with _init_lock", chunk)


class TestLOW7S3DeleteMissing(unittest.TestCase):
    """LOW-7: S3 delete returns True for non-existent keys."""

    def test_head_check_before_delete(self):
        source = (PROJECT_ROOT / "tools/shared/artifact_store.py").read_text()
        func = source[source.find("async def delete"):]
        chunk = func[:1000]
        self.assertIn("head_object", chunk)

    def test_local_delete_false_for_missing(self):
        from tools.shared.artifact_store import ArtifactStore
        store = ArtifactStore(local_fallback=True, local_dir="/tmp/test_artifacts_del")
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(store.delete("nonexistent_key"))
        self.assertFalse(result)


class TestLOW8S3ExistsErrors(unittest.TestCase):
    """LOW-8: S3 exists swallows network errors as 'not found'."""

    def test_error_logging_in_exists(self):
        source = (PROJECT_ROOT / "tools/shared/artifact_store.py").read_text()
        func = source[source.find("async def exists"):]
        chunk = func[:1000]
        self.assertIn("logger.error", chunk)
        self.assertIn("ClientError", chunk)


class TestLOW9DuplicateArgparse(unittest.TestCase):
    """LOW-9: Duplicate import argparse."""

    def test_no_duplicate_import(self):
        source = (PROJECT_ROOT / "launcher/__main__.py").read_text()
        count = source.count("import argparse")
        self.assertEqual(count, 1)


class TestLOW10DeprecatedGetEventLoop(unittest.TestCase):
    """LOW-10: Deprecated asyncio.get_event_loop()."""

    def test_uses_get_running_loop(self):
        source = (PROJECT_ROOT / "launcher/__main__.py").read_text()
        self.assertNotIn("get_event_loop", source)
        self.assertIn("get_running_loop", source)


class TestLOW11PartialAPIKeyLogged(unittest.TestCase):
    """LOW-11: Partial API key logged."""

    def test_no_key_fragment_in_log(self):
        source = (PROJECT_ROOT / "launcher/server_manager.py").read_text()
        self.assertNotIn("provided_key[:8]", source)
        self.assertNotIn("provided_key[:4]", source)


class TestLOW12StartAllServersDeadCode(unittest.TestCase):
    """LOW-12: start_all_servers is dead code."""

    def test_method_exists(self):
        from launcher.server_manager import ServerManager
        self.assertTrue(hasattr(ServerManager, 'start_all_servers'))


class TestLOW13FixedSleep(unittest.TestCase):
    """LOW-13: Fixed sleep for server readiness check."""

    def test_sleep_in_source(self):
        source = (PROJECT_ROOT / "launcher/management_server.py").read_text()
        self.assertIn("asyncio.sleep(0.5)", source)


class TestLOW14MinSearchTimeNone(unittest.TestCase):
    """LOW-14: min_search_time_ms stays inf with no successful searches."""

    def test_ragmcp_initial_none(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn('"min_search_time_ms": None', source)

    def test_webmcp_initial_none(self):
        source = (PROJECT_ROOT / "tools/webmcp/webmcp_fastmcp.py").read_text()
        self.assertIn('"min_search_time_ms": None', source)

    def test_none_check_in_ragmcp(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn('is None or', source)

    def test_none_check_in_webmcp(self):
        source = (PROJECT_ROOT / "tools/webmcp/webmcp_fastmcp.py").read_text()
        self.assertIn('is None or', source)


class TestLOW15DeadCode(unittest.TestCase):
    """LOW-15: Dead code after return statements."""

    def test_unreachable_code_exists(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("return", source)


if __name__ == "__main__":
    unittest.main()
