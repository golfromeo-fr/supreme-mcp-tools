"""
Concurrency stress test for the Turso single-connection claim.

TursoSqlStore and TursoVectorStore share a single libsql connection across all
requests. The code comments claim this is safe because libsql_experimental's C
binding serializes statements via an internal mutex. This test verifies that
claim under concurrent load:

- Multiple threads performing interleaved INSERT/SELECT operations
- No "database is locked" errors
- All writes are durable and readable
- No data corruption or lost updates

If a future libsql build drops the internal mutex, this test will catch it.
"""
import sys
import threading
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")

N_THREADS = 4
OPS_PER_THREAD = 50
DIM = 4


class TestTursoSqlConcurrency:
    """Stress test for TursoSqlStore under concurrent access."""

    def test_concurrent_upsert_and_read(self):
        """Multiple threads upserting + reading must produce consistent results."""
        from shared.impls.turso_sql import TursoSqlStore

        store = TursoSqlStore(url="file::memory:")
        errors = []
        written_ids = set()
        ids_lock = threading.Lock()

        def worker(thread_id: int):
            try:
                for i in range(OPS_PER_THREAD):
                    mid = f"t{thread_id}-m{i}"
                    store.upsert_memory(
                        mid,
                        f"concurrent memory {thread_id}-{i} about threading",
                        "concept",
                        "agent",
                        [f"thread-{thread_id}"],
                        None, None, f"agent-{thread_id}",
                        "low", "auto-delete",
                    )
                    with ids_lock:
                        written_ids.add(mid)

                    # Interleave reads
                    if i % 10 == 5:
                        mem = store.get_memory(mid)
                        if mem and mem["text"] != f"concurrent memory {thread_id}-{i} about threading":
                            errors.append(f"Data mismatch for {mid}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrency errors: {errors[:5]}"
        assert len(written_ids) == N_THREADS * OPS_PER_THREAD

        # All records must be readable after all threads finish
        all_ids = set(store.get_all_memory_ids())
        assert written_ids.issubset(all_ids), \
            f"Missing IDs: {written_ids - all_ids}"

    def test_concurrent_search_text(self):
        """Concurrent FTS5 searches must not crash or corrupt the index."""
        from shared.impls.turso_sql import TursoSqlStore

        store = TursoSqlStore(url="file::memory:")

        # Seed data
        for i in range(20):
            store.upsert_memory(
                f"seed-{i}",
                f"seed memory {i} about concurrency testing database",
                "concept", "agent", [], None, None, "a", "low", "auto-delete",
            )

        errors = []

        def searcher(thread_id: int):
            try:
                for _ in range(OPS_PER_THREAD):
                    results = store.search_text("concurrency")
                    if not isinstance(results, list):
                        errors.append(f"Thread {thread_id}: search returned non-list")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=searcher, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Search concurrency errors: {errors[:5]}"

    def test_concurrent_delete_and_read(self):
        """Concurrent deletes + reads must not corrupt the FTS5 index."""
        from shared.impls.turso_sql import TursoSqlStore

        store = TursoSqlStore(url="file::memory:")

        # Seed data
        all_ids = []
        for i in range(50):
            mid = f"del-{i}"
            store.upsert_memory(
                mid, f"deletable memory {i} unique phrase zebra{i}",
                "concept", "agent", [], None, None, "a", "low", "auto-delete",
            )
            all_ids.append(mid)

        errors = []

        def deleter(thread_id: int):
            try:
                # Each thread deletes a subset
                for i in range(thread_id, 50, N_THREADS):
                    mid = f"del-{i}"
                    store.delete_memory(mid)
                    # Verify it's gone
                    if store.get_memory(mid) is not None:
                        errors.append(f"Thread {thread_id}: {mid} still exists after delete")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        def reader(thread_id: int):
            try:
                for _ in range(OPS_PER_THREAD):
                    store.search_text("zebra")
            except Exception as e:
                errors.append(f"Reader {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=deleter, args=(t,)) for t in range(N_THREADS)]
        threads += [threading.Thread(target=reader, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Delete concurrency errors: {errors[:5]}"
        # All should be deleted
        remaining = [mid for mid in all_ids if store.get_memory(mid) is not None]
        assert not remaining, f"Not deleted: {remaining[:5]}"


class TestTursoVectorConcurrency:
    """Stress test for TursoVectorStore under concurrent access."""

    def test_concurrent_upsert_and_query(self):
        """Multiple threads upserting + querying must produce consistent results."""
        from shared.impls.turso_vector import TursoVectorStore
        from shared.store_models import PointStruct, SparseVector

        store = TursoVectorStore(url="file::memory:")
        store.ensure_collection("conc", dense_dim=DIM, sparse=True)

        errors = []
        written_ids = set()
        ids_lock = threading.Lock()

        def worker(thread_id: int):
            try:
                for i in range(OPS_PER_THREAD):
                    pid = f"t{thread_id}-p{i}"
                    vec = [float((thread_id * i % 9) + 1)] * DIM
                    store.upsert("conc", [
                        PointStruct(
                            id=pid, vector=vec,
                            payload={"text": f"concurrent chunk {thread_id}-{i}",
                                     "codeChunk": f"def f_{thread_id}_{i}(): pass"},
                        ),
                    ])
                    with ids_lock:
                        written_ids.add(pid)

                    # Interleave queries
                    if i % 10 == 5:
                        results = store.query_dense("conc", vec, limit=5)
                        if not isinstance(results, list):
                            errors.append(f"Thread {thread_id}: query returned non-list")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrency errors: {errors[:5]}"
        assert len(written_ids) == N_THREADS * OPS_PER_THREAD

        # All points must be retrievable
        all_points = list(store.iter_all("conc"))
        all_point_ids = {p.id for p in all_points}
        assert written_ids.issubset(all_point_ids), \
            f"Missing points: {written_ids - all_point_ids}"

    def test_concurrent_sparse_search(self):
        """Concurrent FTS5 sparse searches must not crash."""
        from shared.impls.turso_vector import TursoVectorStore
        from shared.store_models import PointStruct, SparseVector

        store = TursoVectorStore(url="file::memory:")
        store.ensure_collection("conc_fts", dense_dim=DIM, sparse=True)

        # Seed data
        for i in range(20):
            store.upsert("conc_fts", [
                PointStruct(
                    id=f"seed-{i}", vector=[0.1] * DIM,
                    payload={"codeChunk": f"concurrent search test chunk {i}"},
                ),
            ])

        errors = []
        empty_sparse = SparseVector(indices=[], values=[])

        def searcher(thread_id: int):
            try:
                for _ in range(OPS_PER_THREAD):
                    results = store.query_sparse(
                        "conc_fts", empty_sparse, limit=5, query_text="concurrent",
                    )
                    if not isinstance(results, list):
                        errors.append(f"Thread {thread_id}: non-list result")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=searcher, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Sparse search concurrency errors: {errors[:5]}"

    def test_concurrent_set_payload(self):
        """Concurrent set_payload calls must not corrupt payloads (C1 merge semantics)."""
        from shared.impls.turso_vector import TursoVectorStore
        from shared.store_models import PointStruct

        store = TursoVectorStore(url="file::memory:")
        store.ensure_collection("conc_merge", dense_dim=DIM)

        # Seed one point per thread
        pids = [f"merge-{t}" for t in range(N_THREADS)]
        store.upsert("conc_merge", [
            PointStruct(id=pid, vector=[0.1] * DIM,
                        payload={"text": f"base text {pid}", "counter": 0})
            for pid in pids
        ])

        errors = []

        def incrementer(thread_id: int):
            try:
                pid = f"merge-{thread_id}"
                for i in range(20):
                    # Read current, increment, write back
                    pts = store.retrieve("conc_merge", [pid], with_payload=True)
                    if pts:
                        current = pts[0].payload.get("counter", 0)
                        store.set_payload("conc_merge", {"counter": current + 1}, ids=[pid])
            except Exception as e:
                errors.append(f"Thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=incrementer, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Set_payload concurrency errors: {errors[:5]}"

        # Verify "text" key survived all the set_payload calls (C1 merge fix)
        for pid in pids:
            pts = store.retrieve("conc_merge", [pid], with_payload=True)
            assert pts[0].payload.get("text") == f"base text {pid}", \
                f"text key was clobbered for {pid}"
            # Counter should be 20 (each thread incremented its own point 20 times)
            assert pts[0].payload.get("counter") == 20, \
                f"Counter for {pid} is {pts[0].payload.get('counter')}, expected 20"
