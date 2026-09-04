"""
Tests for the unified shared cache core (plan C1: ONE cache implementation).

The LRU/TTL/size logic lives exactly once, in tools/shared/cache.py:TTLCache:
- launcher.distributed_registry.CacheManager is a thin asyncio facade over it,
- webmcp_fastmcp.SimpleCache is a thin subclass pinning MAX_SIZE=1000.

Complements the source-pinned bug-fix tests (MED-5, MED-9, MED-16, MED-21 in
tests/test_bug_fixes_medium.py) by testing the SHARED core's behavior directly.
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(coro):
    """Run a coroutine on a LOCAL event loop, never touching the global state.

    (Prior art: tests/test_bug_fixes_medium.py::_run — asyncio.run would call
    asyncio.set_event_loop(None) on exit and break later get_event_loop calls.)
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSharedCoreLRU(unittest.TestCase):
    """True-LRU semantics of the shared core (MED-16 semantics, sync side)."""

    def _cache(self):
        from tools.shared.cache import TTLCache
        return TTLCache(default_ttl=60, max_size=3)

    def test_move_to_end_on_fresh_get(self):
        cache = self._cache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        self.assertEqual(cache.get("k1"), "v1")  # k1 becomes most-recently-used
        cache.set("k4", "v4")                    # overflow
        self.assertEqual(len(cache.cache), 3)
        self.assertIsNone(cache.get("k2"))       # least-recently-used evicted...
        self.assertEqual(cache.get("k1"), "v1")  # ...not the oldest-inserted key
        self.assertEqual(cache.get("k3"), "v3")
        self.assertEqual(cache.get("k4"), "v4")

    def test_overwrite_reinsert_marks_recent(self):
        cache = self._cache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("a", 10)  # overwrite: a becomes most-recently-used, no eviction
        self.assertEqual(len(cache.cache), 3)
        cache.set("d", 4)   # evicts b, not a
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 10)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.get("d"), 4)

    def test_unbounded_by_default(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60)  # max_size defaults to None
        for i in range(50):
            cache.set(f"k{i}", i)
        self.assertEqual(len(cache.cache), 50)


class TestSharedCoreSizeBound(unittest.TestCase):
    """Size-bound behavior of the shared core."""

    def test_expired_purged_before_live_lru(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60, max_size=2)
        cache.set("old", "o", ttl=0.01)
        cache.set("recent", "r")
        time.sleep(0.02)
        cache.set("new", "n")  # at capacity: expired entry purged, live one stays
        self.assertIsNone(cache.get("old"))
        self.assertEqual(cache.get("recent"), "r")
        self.assertEqual(cache.get("new"), "n")

    def test_max_size_class_attr_hook(self):
        from tools.shared.cache import TTLCache

        class Pinned(TTLCache):
            MAX_SIZE = 4

        cache = Pinned(default_ttl=60)
        self.assertEqual(cache.MAX_SIZE, 4)
        for i in range(10):
            cache.set(f"k{i}", i)
        self.assertLessEqual(len(cache.cache), 4)

    def test_max_size_instance_override_after_construction(self):
        from tools.shared.cache import TTLCache

        class Pinned(TTLCache):
            MAX_SIZE = 1000

        cache = Pinned(default_ttl=60)
        cache.MAX_SIZE = 3  # post-construction override (SimpleCache test pattern)
        for i in range(10):
            cache.set(f"k{i}", i)
        self.assertLessEqual(len(cache.cache), 3)


class TestSharedCoreTTLAndStats(unittest.TestCase):
    """TTL expiry and hit/miss stats of the shared core."""

    def test_expired_dropped_on_access(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=1)
        cache.set("ephemeral", "gone_soon", ttl=0.01)
        time.sleep(0.02)
        self.assertIsNone(cache.get("ephemeral"))
        self.assertEqual(len(cache.cache), 0)

    def test_per_entry_ttl_overrides_default(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60)
        cache.set("long", "stays")
        cache.set("short", "gone_soon", ttl=0.01)
        time.sleep(0.02)
        self.assertIsNone(cache.get("short"))
        self.assertEqual(cache.get("long"), "stays")

    def test_hit_miss_stats(self):
        from tools.shared.cache import TTLCache
        cache = TTLCache(default_ttl=60)
        cache.set("a", 1)
        cache.get("a")      # hit
        cache.get("a")      # hit
        cache.get("nope")   # miss
        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total_entries"], 1)


class TestCacheManagerFacade(unittest.TestCase):
    """CacheManager (launcher) is a thin async facade over the shared core."""

    def test_lru_delegation(self):
        from launcher.distributed_registry import CacheManager

        async def scenario():
            cache = CacheManager(max_size=3)
            await cache.set("k1", "v1", ttl=60)
            await cache.set("k2", "v2", ttl=60)
            await cache.set("k3", "v3", ttl=60)
            self.assertEqual(await cache.get("k1"), "v1")  # refresh recency
            await cache.set("k4", "v4", ttl=60)            # evicts k2 (LRU)
            self.assertEqual(len(cache.cache), 3)
            self.assertIsNone(await cache.get("k2"))
            self.assertEqual(await cache.get("k1"), "v1")
            self.assertEqual(await cache.get("k3"), "v3")
            self.assertEqual(await cache.get("k4"), "v4")

        _run(scenario())

    def test_ttl_expiry_delegation(self):
        from launcher.distributed_registry import CacheManager

        async def scenario():
            cache = CacheManager(max_size=10)
            await cache.set("short", "v", ttl=0.01)
            time.sleep(0.02)
            self.assertIsNone(await cache.get("short"))
            self.assertEqual(len(cache.cache), 0)  # expired entry dropped

        _run(scenario())

    def test_invalidate(self):
        from launcher.distributed_registry import CacheManager

        async def scenario():
            cache = CacheManager()
            await cache.set("extensions:webmcp", [{"name": "x"}], ttl=60)
            await cache.invalidate("extensions:webmcp")
            self.assertIsNone(await cache.get("extensions:webmcp"))
            # Invalidating a missing key is a no-op
            await cache.invalidate("extensions:missing")

        _run(scenario())

    def test_invalidate_prefix(self):
        from launcher.distributed_registry import CacheManager

        async def scenario():
            cache = CacheManager()
            await cache.set("extensions:webmcp", 1, ttl=60)
            await cache.set("extensions:simplemcp", 2, ttl=60)
            await cache.set("other:key", 3, ttl=60)
            await cache.invalidate_prefix("extensions:")
            self.assertIsNone(await cache.get("extensions:webmcp"))
            self.assertIsNone(await cache.get("extensions:simplemcp"))
            self.assertEqual(await cache.get("other:key"), 3)

        _run(scenario())


class TestWebmcpSimpleCacheSubclass(unittest.TestCase):
    """webmcp.SimpleCache is a thin subclass of the shared core (MED-21)."""

    def _import(self):
        sys.path.insert(0, str(PROJECT_ROOT / "tools/webmcp"))
        from webmcp_fastmcp import SimpleCache
        return SimpleCache

    def test_subclass_of_shared_core(self):
        from tools.shared.cache import TTLCache
        SimpleCache = self._import()
        self.assertTrue(issubclass(SimpleCache, TTLCache))

    def test_default_and_override_size_bound(self):
        SimpleCache = self._import()
        cache = SimpleCache(default_ttl=60)
        self.assertEqual(cache.MAX_SIZE, 1000)
        cache.MAX_SIZE = 3
        for i in range(10):
            cache.set(f"key_{i}", f"val_{i}", ttl=60)
        self.assertLessEqual(len(cache.cache), 3)

    def test_expiry_and_clear(self):
        SimpleCache = self._import()
        cache = SimpleCache(default_ttl=1)
        cache.set("short", "data", ttl=0.01)
        cache.set("other", "data")
        time.sleep(0.02)
        self.assertIsNone(cache.get("short"))
        self.assertEqual(cache.get("other"), "data")
        cache.clear()
        self.assertEqual(len(cache.cache), 0)


if __name__ == "__main__":
    unittest.main()
