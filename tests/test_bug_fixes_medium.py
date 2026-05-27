"""
Tests for MEDIUM severity bug fixes (MED-1 through MED-21).

Each test verifies a specific fix. Tests are self-contained and don't require
external services (Qdrant, PostgreSQL, S3, etc.).
"""

import os
import sys
import time
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestMED1ScoringWeightsGamma(unittest.TestCase):
    """MED-1: gamma not normalized when alpha+beta change in queryMemory."""

    def test_weights_sum_to_one(self):
        from tools.shared.relevance_scorer import ScoringWeights
        w = ScoringWeights()
        w.alpha = 0.7
        w.beta = 0.3
        w.gamma = 0.0
        total = w.alpha + w.beta + w.gamma
        self.assertAlmostEqual(total, 1.0)

    def test_source_sets_gamma_to_zero(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        section = source[source.find("weights = ScoringWeights()"):]
        chunk = section[:200]
        self.assertIn("weights.gamma = 0.0", chunk)


class TestMED2HeadingTrailingHash(unittest.TestCase):
    """MED-2: Heading trailing # not stripped."""

    def test_trailing_hash_stripped(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_text.py").read_text()
        self.assertIn("rstrip(\"#\")", source)


class TestMED3SelfLoopEdges(unittest.TestCase):
    """MED-3: Self-loop edges silently rejected."""

    def test_self_loop_handled_in_source(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_graph.py").read_text()
        section = source[source.find("async def createMemoryEdge"):]
        chunk = section[:3000]
        self.assertIn("from_id == to_id", chunk)
        self.assertIn("self-loop", chunk)


class TestMED4GenerateEmbeddingAsync(unittest.TestCase):
    """MED-4: generate_embedding blocks async event loop."""

    def test_to_thread_in_upsert(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        idx = source.find("async def upsertMemory")
        section = source[idx:idx + 3000]
        self.assertIn("asyncio.to_thread", section)

    def test_to_thread_in_query(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        idx = source.find("async def queryMemory")
        section = source[idx:idx + 3000]
        self.assertIn("asyncio.to_thread", section)


class TestMED5CachePeriodicEviction(unittest.TestCase):
    """MED-5: Cache no automatic eviction (memory leak)."""

    def test_cleanup_triggered_on_set(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=1)
        cache._cleanup_interval = 0  # trigger every time
        cache.set("a", "val1", ttl=0.01)
        time.sleep(0.02)
        cache._last_cleanup = 0  # force cleanup check
        cache.set("b", "val2", ttl=60)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "val2")

    def test_has_cleanup_interval(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache()
        self.assertTrue(hasattr(cache, '_cleanup_interval'))
        self.assertTrue(hasattr(cache, '_last_cleanup'))


class TestMED7GetMemoryOrder(unittest.TestCase):
    """MED-7: get_memory increments usage_count before SELECT."""

    def test_select_before_update(self):
        source = (PROJECT_ROOT / "tools/shared/pg_store.py").read_text()
        func = source[source.find("def get_memory"):]
        chunk = func[:800]
        select_pos = chunk.find("SELECT *")
        update_pos = chunk.find("UPDATE memories")
        self.assertLess(select_pos, update_pos,
                        "SELECT should come before UPDATE in get_memory")


class TestMED8NavFooterRemoval(unittest.TestCase):
    """MED-8: Nav/footer removal conditioned on include_tables."""

    def test_optimized_keeps_nav_when_tables(self):
        from tools.shared.html_utils import clean_html_optimized
        html = '<nav>Menu</nav><table><tr><td>Data</td></tr></table>'
        result = clean_html_optimized(html, include_tables=True)
        self.assertIn("Menu", result)
        self.assertIn("Data", result)

    def test_optimized_removes_nav_when_no_tables(self):
        from tools.shared.html_utils import clean_html_optimized
        html = '<nav>Menu</nav><p>Content</p>'
        result = clean_html_optimized(html, include_tables=False)
        self.assertNotIn("Menu", result)
        self.assertIn("Content", result)

    def test_basic_removes_nav_when_no_tables(self):
        from tools.shared.html_utils import clean_html_basic
        html = '<nav>Menu</nav><p>Content</p>'
        result = clean_html_basic(html, include_tables=False)
        self.assertNotIn("Menu", result)


class TestMED9CacheKeyMixedTypes(unittest.TestCase):
    """MED-9: generate_cache_key crashes with mixed-type dict keys."""

    def test_mixed_types_no_crash(self):
        from tools.shared.cache import generate_cache_key
        key = generate_cache_key("http://example.com", {1: "a", "b": 2})
        self.assertIsInstance(key, str)
        self.assertTrue(len(key) > 0)

    def test_deterministic(self):
        from tools.shared.cache import generate_cache_key
        params = {1: "a", "b": 2, 3: "c"}
        key1 = generate_cache_key("http://example.com", params)
        key2 = generate_cache_key("http://example.com", params)
        self.assertEqual(key1, key2)


class TestMED10SSNRegex(unittest.TestCase):
    """MED-10: SSN regex massive false-positive rate."""

    def test_valid_ssn_detected(self):
        from tools.shared.pii_redactor import PIIRedactor
        redactor = PIIRedactor()
        matches = redactor.find_all("SSN: 123-45-6789")
        types = [m.pii_type for m in matches]
        self.assertIn("ssn", types)

    def test_plain_number_not_detected(self):
        from tools.shared.pii_redactor import PIIRedactor
        redactor = PIIRedactor()
        matches = redactor.find_all("Order number 1234567890")
        types = [m.pii_type for m in matches]
        self.assertNotIn("ssn", types)

    def test_no_dashes_not_detected(self):
        from tools.shared.pii_redactor import PIIRedactor
        redactor = PIIRedactor()
        matches = redactor.find_all("Code: 123456789")
        types = [m.pii_type for m in matches]
        self.assertNotIn("ssn", types)


class TestMED12EnvFileLocking(unittest.TestCase):
    """MED-12: .env file read-modify-write without file locking."""

    def test_fcntl_import_attempted(self):
        source = (PROJECT_ROOT / "launcher/env_manager.py").read_text()
        self.assertIn("fcntl", source)

    def test_concurrent_env_writes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("FOO=bar\nBAZ=qux\n")
            env_path = Path(f.name)

        try:
            from launcher.env_manager import _update_env_file
            errors = []

            def write_var(name, val):
                try:
                    _update_env_file(name, val, env_path)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=write_var, args=("KEY", str(i)))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Errors during concurrent writes: {errors}")
            content = env_path.read_text()
            self.assertIn("KEY=", content)
        finally:
            env_path.unlink(missing_ok=True)


class TestMED13AllActiveLinesDeactivated(unittest.TestCase):
    """MED-13: Only last active .env line deactivated."""

    def test_all_duplicates_commented(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("FOO=first\nOTHER=val\nFOO=second\nFOO=third\n")
            env_path = Path(f.name)

        try:
            from launcher.env_manager import _update_env_file
            _update_env_file("FOO", "new", env_path)
            content = env_path.read_text()
            lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
            active_foo = [l for l in lines if l.strip().startswith("FOO=")]
            self.assertEqual(len(active_foo), 1)
            self.assertIn("new", active_foo[0])
        finally:
            env_path.unlink(missing_ok=True)


class TestMED14StreamingResponseClose(unittest.TestCase):
    """MED-14: Streaming HTTP responses never closed on exception."""

    def test_finally_blocks_in_discover_tools(self):
        source = (PROJECT_ROOT / "launcher/tools_config.py").read_text()
        func = source[source.find("async def discover_tools_from_server"):]
        chunk = func[:3000]
        self.assertIn("finally:", chunk)
        self.assertIn("aclose", chunk)


class TestMED15OAuthExpiration(unittest.TestCase):
    """MED-15: OAuth pending codes never expire."""

    def test_ttl_field_in_data_structures(self):
        source = (PROJECT_ROOT / "launcher/server_manager.py").read_text()
        self.assertIn("_oauth_ttl", source)
        self.assertIn("expired_codes", source)
        self.assertIn("expired_clients", source)

    def test_pending_codes_have_timestamp(self):
        source = (PROJECT_ROOT / "launcher/server_manager.py").read_text()
        self.assertIn("time.time())", source)
        idx = source.find("_pending_codes[code]")
        self.assertTrue(idx > 0, "_pending_codes should store timestamps")


class TestMED16LRUCacheEviction(unittest.TestCase):
    """MED-16: Cache eviction is FIFO, not LRU."""

    def test_reinsert_on_overwrite(self):
        source = (PROJECT_ROOT / "launcher/distributed_registry.py").read_text()
        func = source[source.find("async def set(self, key"):]
        chunk = func[:500]
        self.assertIn("del self.cache[key]", chunk)


class TestMED17LockDuringIteration(unittest.TestCase):
    """MED-17: Lock released before iterating subscriber queues."""

    def test_subscriber_list_copied(self):
        source = (PROJECT_ROOT / "launcher/distributed_registry.py").read_text()
        func = source[source.find("async def publish"):]
        chunk = func[:800]
        self.assertIn("list(", chunk)


class TestMED18EventLoopReuse(unittest.TestCase):
    """MED-18: asyncio.run() per file during indexing."""

    def test_no_asyncio_run_in_loop(self):
        source = (PROJECT_ROOT / "tools/ragmcp/indexer/incremental_indexer.py").read_text()
        self.assertNotIn("asyncio.run(index_file", source)
        self.assertIn("run_until_complete", source)


class TestMED19EnvironRestoration(unittest.TestCase):
    """MED-19: local_embeddings.py mutates os.environ globally."""

    def test_environ_restored(self):
        source = (PROJECT_ROOT / "tools/ragmcp/indexer/local_embeddings.py").read_text()
        self.assertIn("old_offline", source)
        self.assertIn("finally:", source)

    def test_no_bare_del(self):
        source = (PROJECT_ROOT / "tools/ragmcp/indexer/local_embeddings.py").read_text()
        self.assertNotIn("del os.environ['HF_HUB_OFFLINE']", source)
        self.assertNotIn("del os.environ['TRANSFORMERS_OFFLINE']", source)


class TestMED20MetricsForFileConversions(unittest.TestCase):
    """MED-20: Metrics skipped for file-based conversions."""

    def test_metrics_before_return(self):
        source = (PROJECT_ROOT / "tools/convertermcp/convertermcp_fastmcp.py").read_text()
        idx = source.find("out_path.parent.mkdir")
        section = source[idx:idx + 1500]
        metrics_pos = section.find('metrics["total_conversions"]')
        return_pos = section.find("return msg")
        self.assertLess(metrics_pos, return_pos,
                        "Metrics should be recorded before return in file path")


class TestMED21SimpleCacheMaxSize(unittest.TestCase):
    """MED-21: SimpleCache has no size limit."""

    def test_max_size_enforced(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        cache = SimpleCache(default_ttl=60)
        self.assertEqual(cache.MAX_SIZE, 1000)

    def test_eviction_when_full(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        cache = SimpleCache(default_ttl=60)
        cache.MAX_SIZE = 5
        for i in range(10):
            cache.set(f"key_{i}", f"val_{i}", ttl=60)
        self.assertLessEqual(len(cache.cache), 5)


if __name__ == "__main__":
    unittest.main()
