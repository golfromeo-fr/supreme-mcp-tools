"""P5 — per-entry locks: parallel queries on DIFFERENT connections never
serialize. The 2026-08-30-era single global lock serialized ALL DB access;
this pins the fix.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from dialects._base import DbDialect  # noqa: E402


class SlowFakeDialect(DbDialect):
    name = "slowfake"
    REQUIRED_PARAMS = ("x",)

    def __init__(self, delay: float = 0.4):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def connect(self, params):
        return object()

    def close(self, handle):
        pass

    def ping(self, handle):
        pass

    def run_select(self, handle, sql, max_rows):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return [], False

    def execute(self, handle, sql): return -1
    def list_tables(self, handle): return []
    def describe_table(self, handle, table): return {"columns": [], "constraints": [], "foreign_keys": []}
    def explain(self, handle, sql): return ""
    def format_error(self, e): return {"error": "DB_ERROR", "code": None, "message": str(e), "offset": None}

    # --- tx stubs (E4 ABC) — concurrency tests never enter transaction flows ---
    def open_tx(self, handle, params): raise NotImplementedError
    def select_tx(self, tx_handle, sql, max_rows): raise NotImplementedError
    def execute_tx(self, tx_handle, sql): raise NotImplementedError
    def commit_tx(self, tx_handle): raise NotImplementedError
    def rollback_tx(self, tx_handle): raise NotImplementedError
    def close_tx(self, handle, tx_handle): raise NotImplementedError


@pytest.fixture()
def slow_registry(monkeypatch):
    import dialects
    from connections import ConnectionRegistry

    fake = SlowFakeDialect()
    monkeypatch.setitem(dialects.DIALECTS, "slowfake", fake)
    reg = ConnectionRegistry()
    reg.connect("a", "slowfake", {"x": 1})
    reg.connect("b", "slowfake", {"x": 2})
    return reg, fake


class TestConcurrency:
    def test_parallel_queries_on_different_connections_overlap(self, slow_registry):
        reg, fake = slow_registry
        results = []
        barrier = threading.Barrier(2)

        def query(conn_name):
            entry = reg.get(conn_name)
            with entry.lock:
                barrier.wait()
                entry.dialect = "slowfake"
                from dialects import DIALECTS
                DIALECTS["slowfake"].run_select(entry.handle, "SELECT 1", 10)

        t0 = time.monotonic()
        threads = [
            threading.Thread(target=query, args=("a",)),
            threading.Thread(target=query, args=("b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0

        assert elapsed < 0.7, f"queries serialized ({elapsed:.2f}s) — per-entry locks not independent"
        assert fake.max_active == 2, "queries never overlapped — serialization somewhere"

    def test_same_connection_still_serializes(self, slow_registry):
        reg, fake = slow_registry
        entry = reg.get("a")
        results = []
        barrier = threading.Barrier(2)

        def query():
            barrier.wait()  # outside the lock: both threads must reach it
            with entry.lock:
                from dialects import DIALECTS
                DIALECTS["slowfake"].run_select(entry.handle, "SELECT 1", 10)
                results.append(1)

        t0 = time.monotonic()
        threads = [threading.Thread(target=query) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0

        assert len(results) == 2
        assert fake.max_active == 1, "same-entry queries must serialize (one handle)"
        assert elapsed >= 0.7, "same-entry queries overlapped — lock not held"
