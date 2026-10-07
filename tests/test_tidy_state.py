"""
P4 tidy-state tests (plans/odysseus-reuse-2026-10-06.md).

Layers:
1. Pure sidecar helpers — fingerprint stability/sensitivity, shrink floor
   (abs + fractional), fail-open on corrupt state.
2. mergeDuplicates / decayOrExpire guard behavior with faked stores:
   fingerprint skip (params-aware), fail-closed shrunken read (force unlocks),
   over-deletion refusal, state save after a real pass.

Run: python -m pytest tests/test_tidy_state.py -v
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools"), str(REPO / "tools" / "memorymcp")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import tidy_state
from shared.tidy_state import (
    clear_state, current_fingerprint, load_state, max_drop,
    params_fingerprint, save_state, shrink_suspicious,
)


class _Point:
    def __init__(self, pid, payload, vector=None):
        self.id = pid
        self.payload = payload
        self.vector = vector


def _payload(pid, text="t", **extra):
    p = {"id": pid, "text": text, "memory_type": "concept", "text_hash": f"hash-{pid}",
         "last_accessed": "2026-10-01T00:00:00Z", "usage_count": 1,
         "created_at": "2026-10-01T00:00:00Z", "retention_policy": "auto-delete"}
    p.update(extra)
    return p


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_TIDY_STATE_PATH", str(tmp_path / "tidy_state.json"))
    yield
    clear_state()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_stable_across_ordering(self):
        pts = [_Point("a", _payload("a")), _Point("b", _payload("b"))]
        assert current_fingerprint(pts) == current_fingerprint(list(reversed(pts)))

    def test_sensitive_to_content_changes(self):
        base = [_Point("a", _payload("a"))]
        # text changes always move text_hash in a real store (hash = f(text, owner))
        assert current_fingerprint(base) != current_fingerprint(
            [_Point("a", _payload("a", text="changed", text_hash="hash-changed"))])
        assert current_fingerprint(base) != current_fingerprint(
            [_Point("a", _payload("a", last_accessed="2026-10-02T00:00:00Z"))])
        # point-id fallback: payload carries no "id" → the point id is used
        no_id = {k: v for k, v in _payload("a").items() if k != "id"}
        assert current_fingerprint(base) != current_fingerprint([_Point("b", no_id)])

    def test_params_fingerprint(self):
        assert params_fingerprint(threshold=0.95) == params_fingerprint(threshold=0.95)
        assert params_fingerprint(threshold=0.95) != params_fingerprint(threshold=0.90)


class TestShrinkFloor:
    def test_defaults_abs_and_frac(self):
        assert max_drop(1000) == 50        # 5% = 50 → tie, abs wins tie
        assert max_drop(2000) == 100       # 5% > abs
        assert max_drop(100) == 50         # abs > 5%
        assert shrink_suspicious(1000, 950) is False
        assert shrink_suspicious(1000, 949) is True

    def test_env_tunables(self, monkeypatch):
        monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_ABS", "1")
        monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_FRAC", "0.5")
        assert max_drop(100) == 50         # 50% of 100
        assert shrink_suspicious(100, 60) is False   # floor = 100-50 = 50
        assert shrink_suspicious(100, 49) is True

    def test_no_state_never_suspicious(self):
        assert shrink_suspicious(0, 0) is False
        assert shrink_suspicious(0, 5) is False


class TestStateSidecar:
    def test_save_load_clear_roundtrip(self):
        save_state("abc", 42, params_fingerprint(ttl_days=30))
        state = load_state()
        assert state["fp"] == "abc"
        assert state["count"] == 42
        assert state["params_fp"]
        clear_state()
        assert load_state() == {}

    def test_corrupt_state_fails_open(self, tmp_path):
        tidy_state.state_path().write_text("{not json")
        assert load_state() == {}


# ---------------------------------------------------------------------------
# guard behavior on the real tools (faked stores)
# ---------------------------------------------------------------------------

class _FakeVectorStore:
    def __init__(self, points):
        self.points = points
        self.deleted = []

    def iter_all(self, collection, *, with_vectors=False):
        yield from self.points

    def delete(self, collection, *, ids=None, filter=None):
        self.deleted.extend(ids)


class _FakeSqlStore:
    is_available = False

    def decay_memories(self, ttl_days, min_usage_count, retention_policy=None):
        return 0


def _same_text_points(n, owner="alice"):
    return [_Point(f"id-{i}", _payload(f"id-{i}", text="identical memory",
                                        owner=owner, usage_count=i))
            for i in range(n)]


@pytest.fixture
def mt():
    import memory_tools as memory_tools_mod
    memory_tools_mod._recall_stats.update({"tidy_skip_count": 0})
    return memory_tools_mod


def test_merge_skips_unchanged_store(mt, monkeypatch):
    points = _same_text_points(2)
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())

    # first pass (dry) computes the fp; save it as if a real pass had run
    fp = current_fingerprint(points)
    save_state(fp, len(points), params_fingerprint(threshold=0.95))

    result = asyncio.run(mt.mergeDuplicates(threshold=0.95, dry_run=True))
    assert "fingerprint match" in result
    assert mt._recall_stats["tidy_skip_count"] == 1


def test_merge_skip_is_params_aware(mt, monkeypatch):
    points = _same_text_points(2)
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    save_state(current_fingerprint(points), len(points), params_fingerprint(threshold=0.95))

    result = asyncio.run(mt.mergeDuplicates(threshold=0.90, dry_run=True))
    assert "fingerprint match" not in result  # different threshold → runs


def test_merge_force_overrides_skip(mt, monkeypatch):
    points = _same_text_points(2)
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    save_state(current_fingerprint(points), len(points), params_fingerprint(threshold=0.95))

    result = asyncio.run(mt.mergeDuplicates(threshold=0.95, dry_run=True, force=True))
    assert "fingerprint match" not in result


def test_merge_fail_closed_on_shrunken_read(mt, monkeypatch):
    points = _same_text_points(2)
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    save_state("stale", 500, params_fingerprint(threshold=0.95))  # last pass: 500

    blocked = asyncio.run(mt.mergeDuplicates(dry_run=True))
    assert "Refusing to run" in blocked

    allowed = asyncio.run(mt.mergeDuplicates(dry_run=True, force=True))
    assert "Refusing to run" not in allowed


def test_merge_over_deletion_refused_then_allowed(mt, monkeypatch):
    points = _same_text_points(4)  # 4 identical → would merge away 3 of 4
    fake = _FakeVectorStore(points)
    monkeypatch.setattr(mt, "vector_store", fake)
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_ABS", "1")  # floor = 1
    save_state("different-fp", 4, params_fingerprint(threshold=0.95))

    refused = asyncio.run(mt.mergeDuplicates(threshold=0.95, dry_run=False))
    assert "Refusing to merge away" in refused
    assert fake.deleted == []  # nothing deleted

    monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_ABS", "10")
    ok = asyncio.run(mt.mergeDuplicates(threshold=0.95, dry_run=False))
    assert "Merged" in ok
    assert sorted(fake.deleted) == ["id-0", "id-1", "id-2"]  # highest usage survives
    state = load_state()
    assert state["count"] == 1  # post-pass state saved
    assert state["fp"] != "different-fp"


def test_decay_guards_and_state_save(mt, monkeypatch):
    # 2 old + low-usage points; last pass recorded a much larger store
    old = "2020-01-01T00:00:00Z"
    points = [_Point(f"o-{i}", _payload(f"o-{i}", last_accessed=old,
                                         created_at=old)) for i in range(2)]
    fake = _FakeVectorStore(points)
    monkeypatch.setattr(mt, "vector_store", fake)
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    monkeypatch.setattr(mt, "scroll_all", lambda collection: points)  # decay reads via scroll_all
    save_state("stale", 400, params_fingerprint(ttl_days=30, min_usage_count=0))

    blocked = asyncio.run(mt.decayOrExpire(dry_run=False))
    assert "Refusing to run" in blocked

    monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_ABS", "1")  # floor=1 < 2 deletes → guard #2 trips
    save_state("stale2", 3, params_fingerprint(ttl_days=30, min_usage_count=0))  # shrink ok now
    blocked2 = asyncio.run(mt.decayOrExpire(dry_run=False))
    assert "Refusing to delete" in blocked2

    monkeypatch.setenv("MEMORY_TIDY_MAX_DROP_ABS", "5")
    ok = asyncio.run(mt.decayOrExpire(dry_run=False))
    assert "Deleted 2" in ok
    assert load_state()["count"] == 0  # post-pass state saved


def test_decay_fingerprint_skip(mt, monkeypatch):
    points = [_Point("p-1", _payload("p-1"))]
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))
    monkeypatch.setattr(mt, "sql_store", _FakeSqlStore())
    monkeypatch.setattr(mt, "scroll_all", lambda collection: points)
    save_state(current_fingerprint(points), 1,
               params_fingerprint(ttl_days=30, min_usage_count=0))

    result = asyncio.run(mt.decayOrExpire(dry_run=False))
    assert "fingerprint match" in result
    assert mt._recall_stats["tidy_skip_count"] == 1
