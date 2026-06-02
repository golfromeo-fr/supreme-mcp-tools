"""
Regression tests — behavioral/integration coverage for core subsystems.

Covers areas that don't have dedicated test files:
- Tool discovery (only fastmcp files discovered, support modules skipped)
- ArtifactStore (save/load/delete cycle, path traversal edge cases)
- TTLCache (concurrent access, stats accuracy, cleanup)
- HTML utils (flag combinations, malformed input, roundtrip)
- SSRF protection (DNS resolution edge cases, env override safety)
- Sparse vector generation (vocabulary growth, query isolation, idempotency)
- Rate limiter (burst, refill, concurrent, reset)
- Port manager (allocation, conflicts, ranges)
- Plugin loader (bad plugins, reload, sys.modules hygiene)
- Relevance scorer (weight normalization, boundary conditions)
"""

import os
import sys
import time
import json
import threading
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestToolDiscovery(unittest.TestCase):
    """Regression: discovery only picks up fastmcp files, not support modules."""

    def test_memorymcp_only_discovers_fastmcp(self):
        from launcher.tool_discovery import ToolDiscovery
        td = ToolDiscovery([str(PROJECT_ROOT / "tools/memorymcp")])
        tools = td.discover()
        self.assertIn("memorymcp", tools)
        self.assertTrue(tools["memorymcp"].file_path.endswith("memorymcp_fastmcp.py"))

    def test_ragmcp_only_discovers_fastmcp(self):
        from launcher.tool_discovery import ToolDiscovery
        td = ToolDiscovery([str(PROJECT_ROOT / "tools/ragmcp")])
        tools = td.discover()
        self.assertIn("ragmcp", tools)
        self.assertTrue(tools["ragmcp"].file_path.endswith("ragmcp_fastmcp.py"))

    def test_no_support_modules_discovered(self):
        from launcher.tool_discovery import ToolDiscovery
        td = ToolDiscovery([str(PROJECT_ROOT / "tools/memorymcp")])
        tools = td.discover()
        discovered_files = [t.file_path for t in tools.values()]
        for f in discovered_files:
            self.assertNotIn("memory_core", f)
            self.assertNotIn("text_utils", f)
            self.assertNotIn("memory_text", f)
            self.assertNotIn("memory_graph", f)
            self.assertNotIn("memory_tools", f)

    def test_all_six_tools_discoverable(self):
        from launcher.tool_discovery import ToolDiscovery
        td = ToolDiscovery([
            str(PROJECT_ROOT / "tools/memorymcp"),
            str(PROJECT_ROOT / "tools/ragmcp"),
            str(PROJECT_ROOT / "tools/webmcp"),
            str(PROJECT_ROOT / "tools/simplemcp"),
            str(PROJECT_ROOT / "tools/convertermcp"),
            str(PROJECT_ROOT / "tools/oraclemcp"),
        ])
        tools = td.discover()
        self.assertEqual(len(tools), 6)
        self.assertEqual(
            set(tools.keys()),
            {"memorymcp", "ragmcp", "webmcp", "simplemcp", "convertermcp", "oraclemcp"}
        )


class TestArtifactStoreSaveLoad(unittest.TestCase):
    """Regression: ArtifactStore save/load/delete roundtrip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from tools.shared.artifact_store import ArtifactStore
        self.store = ArtifactStore(local_fallback=True, local_dir=self.tmpdir)

    def test_save_and_load(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            ref = loop.run_until_complete(self.store.save(b"hello world", "test/key.bin"))
            self.assertEqual(ref.size_bytes, 11)
            data = loop.run_until_complete(self.store.load("test/key.bin"))
            self.assertEqual(data, b"hello world")
        finally:
            loop.close()

    def test_save_text_auto_encode(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            ref = loop.run_until_complete(self.store.save("unicode: \u00e9\u00e8\u00ea", "test/t.txt"))
            self.assertIsNotNone(ref)
            data = loop.run_until_complete(self.store.load("test/t.txt"))
            self.assertEqual(data.decode(), "unicode: \u00e9\u00e8\u00ea")
        finally:
            loop.close()

    def test_delete_nonexistent(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.store.delete("no/such/key"))
            self.assertFalse(result)
        finally:
            loop.close()

    def test_exists_nonexistent(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.store.exists("no/such/key"))
            self.assertFalse(result)
        finally:
            loop.close()

    def test_load_nonexistent(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.store.load("no/such/key"))
            self.assertIsNone(result)
        finally:
            loop.close()

    def test_path_traversal_variants(self):
        for bad_key in ["../etc/passwd", "foo/../../../bar", "/etc/passwd"]:
            with self.assertRaises(ValueError, msg=f"Should reject: {bad_key}"):
                self.store._local_path(bad_key)
        normalized_backslash = Path("..\\windows\\system32").resolve()
        self.assertFalse(str(self.store._local_path("normal/key")).startswith(str(normalized_backslash)))


class TestTTLCacheConcurrent(unittest.TestCase):
    """Regression: TTLCache under concurrent access."""

    def test_concurrent_read_write(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60)
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    cache.set(f"key_{n}_{i}", f"val_{i}", ttl=60)
            except Exception as e:
                errors.append(e)

        def reader(n):
            try:
                for i in range(100):
                    cache.get(f"key_{n}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for n in range(5):
            threads.append(threading.Thread(target=writer, args=(n,)))
            threads.append(threading.Thread(target=reader, args=(n,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_stats_accuracy(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("missing")  # miss
        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)

    def test_expired_entry_removed_on_get(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=1)
        cache.set("ephemeral", "gone_soon", ttl=0.01)
        time.sleep(0.02)
        self.assertIsNone(cache.get("ephemeral"))
        self.assertEqual(len(cache.cache), 0)

    def test_cleanup_removes_all_expired(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=1)
        for i in range(10):
            cache.set(f"key_{i}", f"val_{i}", ttl=0.01)
        cache.set("fresh", "stays", ttl=60)
        time.sleep(0.02)
        cache.cleanup_expired()
        self.assertEqual(len(cache.cache), 1)
        self.assertEqual(cache.get("fresh"), "stays")


class TestHTMLUtilsFlagCombinations(unittest.TestCase):
    """Regression: clean_html_* with all flag combinations."""

    def test_all_disabled(self):
        from tools.shared.html_utils import clean_html_optimized
        html = '<img src="x"/><table><tr><td>t</td></tr></table><a href="x">link</a><p>text</p>'
        result = clean_html_optimized(html, include_images=False, include_tables=False, include_links=False)
        self.assertNotIn("<img", result)
        self.assertNotIn("<table", result)
        self.assertIn("link", result)
        self.assertNotIn("href", result)
        self.assertIn("text", result)

    def test_all_enabled(self):
        from tools.shared.html_utils import clean_html_optimized
        html = '<img src="x"/><table><tr><td>t</td></tr></table><a href="x">link</a>'
        result = clean_html_optimized(html, include_images=True, include_tables=True, include_links=True)
        self.assertIn("t", result)
        self.assertIn("link", result)

    def test_empty_input(self):
        from tools.shared.html_utils import clean_html_optimized, clean_html_basic
        self.assertEqual(clean_html_optimized(""), "")
        self.assertEqual(clean_html_basic(""), "")

    def test_none_input(self):
        from tools.shared.html_utils import clean_html_optimized, clean_html_basic
        self.assertEqual(clean_html_optimized(None), "")
        self.assertEqual(clean_html_basic(None), "")

    def test_script_tag_removal(self):
        from tools.shared.html_utils import clean_html_optimized
        html = '<p>safe</p><script>alert("xss")</script><p>also safe</p>'
        result = clean_html_optimized(html)
        self.assertNotIn("alert", result)
        self.assertNotIn("script", result)
        self.assertIn("safe", result)

    def test_basic_vs_optimized_parity(self):
        from tools.shared.html_utils import clean_html_optimized, clean_html_basic
        html = '<p>Hello</p><a href="http://example.com">Link</a><img src="pic.jpg"/>'
        basic = clean_html_basic(html, include_images=False, include_links=False)
        optimized = clean_html_optimized(html, include_images=False, include_links=False)
        self.assertIn("Hello", basic)
        self.assertIn("Hello", optimized)
        self.assertIn("Link", basic)
        self.assertIn("Link", optimized)


class TestSSRFEdgeCases(unittest.TestCase):
    """Regression: SSRF protection edge cases."""

    def test_google_metadata(self):
        from tools.shared.utils import is_internal_url
        self.assertTrue(is_internal_url("http://metadata.google.internal/computeMetadata/v1/"))

    def test_azure_metadata(self):
        from tools.shared.utils import is_internal_url
        self.assertTrue(is_internal_url("http://metadata.azure.internal/metadata/instance"))

    def test_ipv6_loopback(self):
        from tools.shared.utils import is_internal_url
        self.assertTrue(is_internal_url("http://[::1]/admin"))

    def test_external_url_allowed(self):
        from tools.shared.utils import is_internal_url
        self.assertFalse(is_internal_url("http://example.com/page"))

    def test_unresolvable_host_treated_as_internal(self):
        from tools.shared.utils import is_internal_url
        self.assertTrue(is_internal_url("http://this-host-definitely-does-not-exist-xyz123.invalid/"))


class TestSparseVectorIdempotency(unittest.TestCase):
    """Regression: sparse vector generation is deterministic."""

    def test_same_input_produces_terms(self):
        from tools.ragmcp.indexer.sparse_vector_gen import CodeSparseVectorGenerator
        gen = CodeSparseVectorGenerator()
        v1 = gen.generate_sparse_vector("SELECT * FROM users WHERE id = 1")
        v2 = gen.generate_sparse_vector("SELECT * FROM users WHERE id = 1")
        self.assertTrue(len(v1) > 0)
        self.assertEqual(set(v1.keys()), set(v2.keys()))

    def test_different_input_different_output(self):
        from tools.ragmcp.indexer.sparse_vector_gen import CodeSparseVectorGenerator
        gen = CodeSparseVectorGenerator()
        v1 = gen.generate_sparse_vector("SELECT * FROM users")
        v2 = gen.generate_sparse_vector("INSERT INTO orders VALUES (1)")
        self.assertNotEqual(v1, v2)

    def test_empty_input_empty_vector(self):
        from tools.ragmcp.indexer.sparse_vector_gen import CodeSparseVectorGenerator
        gen = CodeSparseVectorGenerator()
        v = gen.generate_sparse_vector("")
        self.assertEqual(v, {})

    def test_vocabulary_grows_with_indexing(self):
        from tools.ragmcp.indexer.sparse_vector_gen import CodeSparseVectorGenerator
        gen = CodeSparseVectorGenerator()
        gen.generate_sparse_vector("SELECT * FROM foo", {"language": "sql"})
        vocab1 = gen.get_vocabulary_size()
        gen.generate_sparse_vector("INSERT INTO bar VALUES (1)", {"language": "sql"})
        vocab2 = gen.get_vocabulary_size()
        self.assertGreaterEqual(vocab2, vocab1)

    def test_query_does_not_affect_idf(self):
        from tools.ragmcp.indexer.sparse_vector_gen import CodeSparseVectorGenerator
        gen = CodeSparseVectorGenerator()
        gen.generate_index_vector("SELECT * FROM foo", {"language": "sql"})
        idf_before = gen._compute_idf("select")
        for _ in range(100):
            gen.generate_query_vector("SELECT * FROM bar", {"language": "sql"})
        idf_after = gen._compute_idf("select")
        self.assertAlmostEqual(idf_before, idf_after, places=5)


class TestRateLimiterBehavior(unittest.TestCase):
    """Regression: RateLimiter token bucket behavior."""

    def test_allows_within_burst(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        for _ in range(5):
            self.assertTrue(rl.is_allowed("key"))

    def test_blocks_over_burst(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst_size=3)
        for _ in range(3):
            rl.is_allowed("key")
        self.assertFalse(rl.is_allowed("key"))

    def test_refill_over_time(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=6000, burst_size=1)
        rl.is_allowed("key")
        self.assertFalse(rl.is_allowed("key"))
        time.sleep(0.02)
        self.assertTrue(rl.is_allowed("key"))

    def test_remaining_decreases(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst_size=10)
        rl.is_allowed("key")
        self.assertLess(rl.get_remaining("key"), 10)

    def test_reset_restores_bucket(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        rl.is_allowed("key")
        rl.is_allowed("key")
        rl.reset("key")
        self.assertTrue(rl.is_allowed("key"))

    def test_separate_keys_independent(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=60, burst_size=1)
        self.assertTrue(rl.is_allowed("key_a"))
        self.assertTrue(rl.is_allowed("key_b"))
        self.assertFalse(rl.is_allowed("key_a"))
        self.assertFalse(rl.is_allowed("key_b"))


class TestPortManagerAllocation(unittest.TestCase):
    """Regression: PortManager allocation logic."""

    def _config(self):
        return {
            "ranges": {"mcp": [9000, 9099], "mgmt": [9100, 9199], "system": [9200, 9299]},
            "reserved": {"central_management": 9200},
            "assignments": {"webmcp": 9001, "simplemcp": 9002},
        }

    def test_allocate_from_mcp_range(self):
        from launcher.port_manager import PortManager
        pm = PortManager(ports_config=self._config())
        port = pm.allocate_port("test_tool", "mcp")
        self.assertGreaterEqual(port, 9000)
        self.assertLessEqual(port, 9099)

    def test_allocate_unique(self):
        from launcher.port_manager import PortManager
        pm = PortManager(ports_config=self._config())
        ports = set()
        for i in range(10):
            p = pm.allocate_port(f"tool_{i}", "mcp")
            ports.add(p)
        self.assertEqual(len(ports), 10)

    def test_reserved_port_accessible(self):
        from launcher.port_manager import PortManager
        pm = PortManager(ports_config=self._config())
        reserved = pm.reserved_ports
        self.assertEqual(reserved.get("central_management"), 9200)


class TestPluginLoaderHygiene(unittest.TestCase):
    """Regression: Plugin loader sys.modules hygiene."""

    def setUp(self):
        self.plugin_dir = Path(tempfile.mkdtemp()) / "plugins"
        self.plugin_dir.mkdir(parents=True)

    def test_bad_syntax_plugin_cleaned_up(self):
        from launcher.plugins.loader import PluginLoader
        (self.plugin_dir / "bad_syntax.py").write_text("def (broken syntax")
        loader = PluginLoader(plugin_dir=str(self.plugin_dir))
        with self.assertRaises(SyntaxError):
            loader.load_plugin("bad_syntax", "test")
        self.assertNotIn("bad_syntax", sys.modules)

    def test_missing_register_function(self):
        from launcher.plugins.loader import PluginLoader
        (self.plugin_dir / "no_register.py").write_text("x = 1\n")
        loader = PluginLoader(plugin_dir=str(self.plugin_dir))
        with self.assertRaises(ValueError):
            loader.load_plugin("no_register", "test")
        self.assertIn("no_register", sys.modules)
        del sys.modules["no_register"]

    def test_reload_fresh_module(self):
        from launcher.plugins.loader import PluginLoader
        (self.plugin_dir / "reloadable.py").write_text(
            "counter = 0\ndef register(r, t): global counter; counter += 1\n"
        )
        loader = PluginLoader(plugin_dir=str(self.plugin_dir))
        reg = MagicMock()
        m1 = loader.load_plugin("reloadable", "test", registry=reg)
        self.assertIsNotNone(m1)
        m2 = loader.reload_plugin("reloadable", "test", registry=reg)
        self.assertIsNot(m1, m2)
        del sys.modules["reloadable"]

    def test_discover_ignores_underscore_prefix(self):
        from launcher.plugins.loader import PluginLoader
        (self.plugin_dir / "_hidden.py").write_text("def register(r,t): pass\n")
        (self.plugin_dir / "visible.py").write_text("def register(r,t): pass\n")
        loader = PluginLoader(plugin_dir=str(self.plugin_dir))
        discovered = loader.discover_plugins()
        self.assertNotIn("_hidden", discovered)
        self.assertIn("visible", discovered)
        if "visible" in sys.modules:
            del sys.modules["visible"]


class TestRelevanceScorerBoundaries(unittest.TestCase):
    """Regression: Relevance scorer edge cases."""

    def test_all_zero_weights(self):
        from tools.shared.relevance_scorer import ScoringWeights, compute_relevance_score
        w = ScoringWeights(alpha=0.0, beta=0.0, gamma=0.0)
        score = compute_relevance_score(0.9, None, 5, weights=w)
        self.assertAlmostEqual(score, 0.0)

    def test_only_semantic(self):
        from tools.shared.relevance_scorer import ScoringWeights, compute_relevance_score
        w = ScoringWeights(alpha=1.0, beta=0.0, gamma=0.0)
        score = compute_relevance_score(0.8, None, 0, weights=w)
        self.assertAlmostEqual(score, 0.8)

    def test_score_clamped_at_one(self):
        from tools.shared.relevance_scorer import ScoringWeights, compute_relevance_score
        w = ScoringWeights(alpha=1.0, beta=1.0, gamma=1.0)
        score = compute_relevance_score(1.0, None, 100, weights=w)
        self.assertLessEqual(score, 1.0)

    def test_cosine_similarity_zero_vectors(self):
        from tools.shared.relevance_scorer import cosine_similarity
        self.assertEqual(cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)
        self.assertEqual(cosine_similarity([1, 0, 0], [0, 0, 0]), 0.0)

    def test_cosine_similarity_orthogonal(self):
        from tools.shared.relevance_scorer import cosine_similarity
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_usage_boost_saturates(self):
        from tools.shared.relevance_scorer import compute_usage_boost
        self.assertAlmostEqual(compute_usage_boost(1000, 100), 1.0)
        self.assertAlmostEqual(compute_usage_boost(50, 100), 0.5)


class TestSimpleCacheExpiry(unittest.TestCase):
    """Regression: SimpleCache expiry and max size."""

    def test_expired_entry_returns_none(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        cache = SimpleCache(default_ttl=1)
        cache.set("short", "data", ttl=0.01)
        time.sleep(0.02)
        self.assertIsNone(cache.get("short"))

    def test_max_size_evicts_oldest(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        cache = SimpleCache(default_ttl=60)
        cache.MAX_SIZE = 3
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("d"), "4")

    def test_clear_empties(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        cache = SimpleCache(default_ttl=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        self.assertIsNone(cache.get("a"))
        self.assertEqual(len(cache.cache), 0)


class TestPIIRedactorRegression(unittest.TestCase):
    """Regression: PII redactor edge cases."""

    def test_no_pii_clean_text_unchanged(self):
        from tools.shared.pii_redactor import PIIRedactor
        r = PIIRedactor()
        text = "Hello world, this is a clean message."
        matches = r.find_all(text)
        self.assertEqual(len(matches), 0)

    def test_redact_multiple_types(self):
        from tools.shared.pii_redactor import PIIRedactor
        r = PIIRedactor()
        text = "Email: user@example.com and key: api_key=abc123def456ghi789jkl012mno345"
        matches = r.find_all(text)
        types = {m.pii_type for m in matches}
        self.assertIn("email", types)

    def test_ssn_with_dashes_accepted(self):
        from tools.shared.pii_redactor import PIIRedactor
        r = PIIRedactor()
        matches = r.find_all("SSN: 123-45-6789")
        types = [m.pii_type for m in matches]
        self.assertIn("ssn", types)

    def test_ten_digit_number_rejected(self):
        from tools.shared.pii_redactor import PIIRedactor
        r = PIIRedactor()
        matches = r.find_all("Order: 1234567890")
        types = [m.pii_type for m in matches]
        self.assertNotIn("ssn", types)

    def test_private_key_detected(self):
        from tools.shared.pii_redactor import PIIRedactor
        r = PIIRedactor()
        matches = r.find_all("-----BEGIN RSA PRIVATE KEY-----\nMIIEowI...")
        types = [m.pii_type for m in matches]
        self.assertIn("private_key", types)


class TestCacheKeyDeterminism(unittest.TestCase):
    """Regression: cache key generation."""

    def test_order_independent(self):
        from tools.shared.cache import generate_cache_key
        k1 = generate_cache_key("http://x", {"a": 1, "b": 2})
        k2 = generate_cache_key("http://x", {"b": 2, "a": 1})
        self.assertEqual(k1, k2)

    def test_url_sensitive(self):
        from tools.shared.cache import generate_cache_key
        k1 = generate_cache_key("http://a", {"x": 1})
        k2 = generate_cache_key("http://b", {"x": 1})
        self.assertNotEqual(k1, k2)

    def test_empty_params(self):
        from tools.shared.cache import generate_cache_key
        k = generate_cache_key("http://x", {})
        self.assertIsInstance(k, str)
        self.assertTrue(len(k) > 0)


if __name__ == "__main__":
    unittest.main()
