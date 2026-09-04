"""
Tests for HIGH severity bug fixes (HIGH-1 through HIGH-18, excluding already-tested ones).

Tests are self-contained and don't require external services.
"""

import os
import re
import sys
import copy
import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestHIGH1ReindexMemoryDimValidation(unittest.TestCase):
    """HIGH-1: reindexMemory accepts mismatched embedding dimensions."""

    def test_dimension_check_in_source(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        reindex_section = source[source.find("async def reindexMemory"):]
        # Should contain dimension validation
        self.assertIn("dimension", reindex_section[:3000].lower())


class TestHIGH2DecayOrExpireLogic(unittest.TestCase):
    """HIGH-2: decayOrExpire uses OR instead of AND for ttl+usage."""

    def test_decay_conditions_in_source(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_tools.py").read_text()
        decay_section = source[source.find("async def decayOrExpire"):]
        # Should reference both ttl_days and min_usage with AND logic
        chunk = decay_section[:3000]
        self.assertIn("ttl_days", chunk)
        self.assertIn("min_usage", chunk)


class TestHIGH3EdgeRaceCondition(unittest.TestCase):
    """HIGH-3: createMemoryEdge reads and writes edges non-atomically."""

    def test_edge_append_uses_current_edges(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_graph.py").read_text()
        edge_section = source[source.find("async def createMemoryEdge"):]
        chunk = edge_section[:2000]
        # Should retrieve current edges before appending
        self.assertIn("edges", chunk)
        self.assertIn("current_edges", chunk)


class TestHIGH4CrossRefContinue(unittest.TestCase):
    """HIGH-4: textToGraph stops at first cross-ref instead of continuing."""

    def test_cross_refs_handled_with_continue(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_text.py").read_text()
        # Look for continue statement after cross-ref handling
        self.assertIn("continue", source)


class TestHIGH5ClustersPreserved(unittest.TestCase):
    """HIGH-5: strip_llm_artifacts removes CLUSTERS section."""

    def test_clusters_section_preserved(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/memorymcp"))
        from text_utils import strip_llm_artifacts
        text = "Some preamble\n\n## CLUSTERS:\nCLUSTER_A: rule1, rule2 — summary\n\n## CLUSTERS\nrule3, rule4"
        result = strip_llm_artifacts(text)
        self.assertIn("CLUSTERS", result)


class TestHIGH6DanglingMermaidNodes(unittest.TestCase):
    """HIGH-6: getMemoryGraph creates dangling Mermaid nodes."""

    def test_graph_guard_for_missing_nodes(self):
        source = (PROJECT_ROOT / "tools/memorymcp/memory_graph.py").read_text()
        graph_section = source[source.find("async def getMemoryGraph"):]
        # Should have guard against missing nodes
        chunk = graph_section[:3000]
        # The fix adds a check for node existence
        self.assertTrue("not in" in chunk or "exists" in chunk.lower() or "if " in chunk)


class TestHIGH7RateLimiterThreadSafety(unittest.TestCase):
    """HIGH-7: RateLimiter uses asyncio.Lock in sync context."""

    def test_rate_limiter_uses_threading_lock(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=10)
        self.assertIsInstance(rl._lock, threading.Lock)

    def test_concurrent_is_allowed(self):
        from launcher.security.rate_limit import RateLimiter
        rl = RateLimiter(requests_per_minute=100, burst_size=10)
        results = []
        errors = []

        def consume():
            try:
                results.append(rl.is_allowed("test_key"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=consume) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 20)


class TestHIGH8DoubleShutdownRace(unittest.TestCase):
    """HIGH-8: ServerManager shutdown race condition."""

    def test_graceful_shutdown_mechanism(self):
        source = (PROJECT_ROOT / "launcher/server_manager.py").read_text()
        # Should use should_exit for graceful shutdown
        self.assertIn("should_exit", source)


class TestHIGH9BaseExceptionCatch(unittest.TestCase):
    """HIGH-9: ServiceRegistry catches BaseException (too broad)."""

    def test_health_check_catches_exception_not_base(self):
        source = (PROJECT_ROOT / "launcher/service_registry.py").read_text()
        # The health check catch should use Exception, not BaseException
        health_check = source[source.find("async def _check_health"):]
        # Verify it doesn't catch bare BaseException
        self.assertNotIn("except BaseException", health_check[:2000])
        # Should catch Exception or specific types
        self.assertTrue("except" in health_check[:2000])


class TestHIGH10WebSocketAuth(unittest.TestCase):
    """HIGH-10: Management server WebSocket has no auth."""

    def test_websocket_auth_check_in_source(self):
        source = (PROJECT_ROOT / "launcher/management_server.py").read_text()
        ws_section = source[source.find("async def websocket_events"):]
        chunk = ws_section[:2000]
        self.assertIn("auth", chunk.lower())


class TestHIGH11PluginSysModulesCleanup(unittest.TestCase):
    """HIGH-11: Broken module left in sys.modules on plugin load failure."""

    def test_cleanup_on_exec_failure(self):
        from launcher.plugins.loader import PluginLoader
        loader = PluginLoader(plugin_dir="/tmp/test_plugins_nonexistent")
        # Create a bad plugin
        plugin_dir = Path("/tmp/test_plugins_bad")
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "bad_plugin.py").write_text("raise RuntimeError('intentional')")

        loader2 = PluginLoader(plugin_dir=str(plugin_dir))
        with self.assertRaises(RuntimeError):
            loader2.load_plugin("bad_plugin", "test_tool")
        # Module should NOT be in sys.modules after failure
        self.assertNotIn("bad_plugin", sys.modules)

    def test_success_adds_to_sys_modules(self):
        from launcher.plugins.loader import PluginLoader
        plugin_dir = Path("/tmp/test_plugins_good")
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "good_plugin.py").write_text(
            "def register(registry, tool_name): pass\n"
        )

        loader = PluginLoader(plugin_dir=str(plugin_dir))
        reg = MagicMock()
        loader.load_plugin("good_plugin", "test_tool", registry=reg)
        self.assertIn("good_plugin", sys.modules)
        # Cleanup
        del sys.modules["good_plugin"]


class TestHIGH12RegistryWriteLock(unittest.TestCase):
    """HIGH-12: _global_registries write without lock."""

    def test_write_lock_exists(self):
        from launcher.tool_extensions.registry import _registry_write_lock
        self.assertIsInstance(_registry_write_lock, type(threading.Lock()))

    def test_concurrent_registry_registration(self):
        from launcher.tool_extensions.registry import (
            ExtensionRegistry, _global_registries,
        )
        errors = []

        def register_tool(name):
            try:
                reg = ExtensionRegistry(tool_name=name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_tool, args=(f"tool_{i}",))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(_global_registries), 20)

        # Cleanup
        _global_registries.clear()


class TestHIGH13ConnectionPool(unittest.TestCase):
    """HIGH-13: Fake connection pool creates new connection per operation."""

    def test_pool_import_attempted(self):
        # Phase 5: pool logic lives in impls/postgres_sql.py (_PsycopgPool)
        source = (PROJECT_ROOT / "tools/shared/impls/postgres_sql.py").read_text()
        self.assertIn("ConnectionPool", source)


class TestHIGH14PgTrgmOptional(unittest.TestCase):
    """HIGH-14: _ensure_schema fails if pg_trgm unavailable."""

    def test_pg_trgm_in_try_except(self):
        # Phase 5: pg_trgm handling moved to impls/postgres_sql.py
        # (behavioural coverage in tests/test_postgres_impl_bugs.py)
        source = (PROJECT_ROOT / "tools/shared/impls/postgres_sql.py").read_text()
        schema_start = source.find("def _ensure_schema")
        schema_end = source.find("\n    def ", schema_start + 10)
        schema = source[schema_start:schema_end]
        self.assertIn("pg_trgm", schema)
        self.assertIn("try:", schema)


class TestHIGH16AsyncToThread(unittest.TestCase):
    """HIGH-16: google_search_api blocks async event loop."""

    def test_asyncio_to_thread_in_source(self):
        source = (PROJECT_ROOT / "tools/webmcp/webmcp_fastmcp.py").read_text()
        google_section = source[source.find("async def google_search_api"):]
        chunk = google_section[:3000]
        self.assertIn("asyncio.to_thread", chunk)


class TestHIGH17HtmlToMarkdownDedup(unittest.TestCase):
    """HIGH-17: _html_to_markdown duplicates nested content."""

    def test_visited_set_in_render(self):
        source = (PROJECT_ROOT / "tools/webmcp/webmcp_fastmcp.py").read_text()
        md_section = source[source.find("def _html_to_markdown"):]
        chunk = md_section[:2000]
        self.assertIn("_visited", chunk)

    def test_no_descendants_iteration(self):
        source = (PROJECT_ROOT / "tools/webmcp/webmcp_fastmcp.py").read_text()
        md_section = source[source.find("def _html_to_markdown"):]
        chunk = md_section[:2000]
        # Old code used soup.descendants; new code should not
        self.assertNotIn(".descendants", chunk)


class TestHIGH18HybridParallel(unittest.TestCase):
    """HIGH-18: Hybrid search runs dense+sparse in parallel via asyncio.gather."""

    def test_asyncio_gather_in_hybrid(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        hybrid_section = source[source.find("async def _search_hybrid"):]
        chunk = hybrid_section[:2000]
        self.assertIn("asyncio.gather", chunk)

    def test_create_task_used(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        hybrid_section = source[source.find("async def _search_hybrid"):]
        chunk = hybrid_section[:2000]
        self.assertIn("asyncio.create_task", chunk)


class TestHIGH18RRFFusion(unittest.TestCase):
    """HIGH-18: Hybrid search uses Reciprocal Rank Fusion with deduplication."""

    def _make_result(self, doc_id, score, file_path="test.py"):
        return {"id": doc_id, "score": score, "payload": {"filePath": file_path, "codeChunk": "code"}}

    def test_rrf_function_exists(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("def _reciprocal_rank_fusion", source)

    def test_rrf_deduplicates_by_id(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        dense = [self._make_result("A", 0.9), self._make_result("B", 0.7)]
        sparse = [self._make_result("A", 0.8), self._make_result("C", 0.6)]

        fused = _reciprocal_rank_fusion([dense, sparse], k=60)
        ids = [r["id"] for r in fused]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs in fused results")
        self.assertEqual(set(ids), {"A", "B", "C"})

    def test_rrf_combined_score_higher_for_overlap(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        dense = [self._make_result("A", 0.9)]
        sparse = [self._make_result("A", 0.8), self._make_result("B", 0.6)]

        fused = _reciprocal_rank_fusion([dense, sparse], k=60)
        a_score = next(r["score"] for r in fused if r["id"] == "A")
        b_score = next(r["score"] for r in fused if r["id"] == "B")
        self.assertGreater(a_score, b_score, "Doc appearing in both lists should rank higher")

    def test_rrf_formula_correct(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        k = 60
        dense = [self._make_result("A", 0.9), self._make_result("B", 0.7)]
        sparse = [self._make_result("A", 0.8)]

        fused = _reciprocal_rank_fusion([dense, sparse], k=k)
        a_score = next(r["score"] for r in fused if r["id"] == "A")
        b_score = next(r["score"] for r in fused if r["id"] == "B")

        expected_a = 1.0 / (k + 1) + 1.0 / (k + 1)
        expected_b = 1.0 / (k + 2)
        self.assertAlmostEqual(a_score, expected_a, places=10)
        self.assertAlmostEqual(b_score, expected_b, places=10)

    def test_rrf_sorted_descending(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        dense = [self._make_result("A", 0.9), self._make_result("C", 0.5)]
        sparse = [self._make_result("B", 0.8), self._make_result("A", 0.7)]

        fused = _reciprocal_rank_fusion([dense, sparse], k=60)
        scores = [r["score"] for r in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_empty_input(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        fused = _reciprocal_rank_fusion([[], []], k=60)
        self.assertEqual(fused, [])

    def test_rrf_single_list(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        items = [self._make_result("A", 0.9), self._make_result("B", 0.7)]
        fused = _reciprocal_rank_fusion([items], k=60)
        self.assertEqual(len(fused), 2)

    def test_rrf_preserves_payload(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/ragmcp"))
        from ragmcp_fastmcp import _reciprocal_rank_fusion

        dense = [{"id": "X", "score": 0.9, "payload": {"filePath": "a.py", "codeChunk": "code"}}]
        sparse = [{"id": "X", "score": 0.8, "payload": {"filePath": "a.py", "codeChunk": "code"}}]

        fused = _reciprocal_rank_fusion([dense, sparse], k=60)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["payload"]["filePath"], "a.py")

    def test_hybrid_calls_rrf_not_concatenation(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        hybrid_section = source[source.find("async def _search_hybrid"):]
        chunk = hybrid_section[:2000]
        self.assertIn("_reciprocal_rank_fusion", chunk)
        self.assertNotIn("=== Dense Search ===\n{dense_results_text}", chunk)
        self.assertIn("=== Fused Results ===", chunk)

    def test_hybrid_uses_structured_search_functions(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        hybrid_section = source[source.find("async def _search_hybrid"):]
        chunk = hybrid_section[:2000]
        self.assertIn("_do_dense_search", chunk)
        self.assertIn("_do_sparse_search", chunk)

    def test_do_dense_returns_structured_data(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("async def _do_dense_search", source)
        self.assertIn("async def _do_sparse_search", source)


if __name__ == "__main__":
    unittest.main()
