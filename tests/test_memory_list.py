"""E2 — listMemories browse tool (unit, isolated from live backends).

The fake vector store implements the two members listMemories uses
(scroll). Import isolation: TURSO_DATABASE_URL is pointed at a throwaway
file BEFORE memory_core is imported (dotenv does not override existing env).
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

TMP_DB = Path("/tmp") / f"memorymcp_e2_test_{uuid.uuid4().hex[:8]}.db"
os.environ["TURSO_DATABASE_URL"] = f"file:{TMP_DB}"
os.environ.setdefault("MEMORY_COLLECTION", "memory-store-e2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
TOOL_DIR = TOOLS_DIR / "memorymcp"
for p in (str(PROJECT_ROOT), str(TOOLS_DIR), str(TOOL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.store_models import PointStruct  # noqa: E402


class FakeScrollStore:
    """Just enough of VectorStore for listMemories: the rowid-ascending
    scroll contract (next cursor = max id when the batch is full) plus
    tag MatchContains filtering."""

    name = "fake"

    def __init__(self, payloads: list[dict]):
        self.points = [
            PointStruct(id=str(i + 1), vector=[], payload=p)
            for i, p in enumerate(payloads)
        ]

    def scroll(self, collection, *, limit=1000, offset=None, with_payload=True, filter=None):
        seq = self.points
        if offset is not None:
            seq = [p for p in seq if int(p.id) > int(offset)]
        if filter is not None:  # MatchContains on tags
            for cond in getattr(filter, "must", []) or []:
                key = cond.key
                needle = getattr(cond.match, "value", None)
                seq = [p for p in seq
                       if needle in (p.payload or {}).get(key, [])]
        batch = seq[:limit]
        next_cursor = max((int(p.id) for p in batch), default=None) if len(batch) == limit else None
        return batch, next_cursor

    def get_collection(self, name):
        class _Info:
            points_count = len(self.points)
        return _Info()


@pytest.fixture()
def store(monkeypatch):
    import memory_tools

    fake = FakeScrollStore([
        {"text": f"memory {i}", "memory_type": "concept",
         "tags": (["alpha"] if i % 2 else ["beta"]),
         "sensitivity": "low", "created_at": f"2026-01-{i + 1:02d}T00:00:00",
         "last_accessed": None, "usage_count": i, "source": "test"}
        for i in range(1, 8)  # 7 records, insertion order 1..7
    ])
    monkeypatch.setattr(memory_tools, "vector_store", fake)
    return fake, memory_tools


def _run(coro):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestListMemories:
    def test_default_page_recent_first(self, store):
        _fake, mt = store
        out = _run(mt.listMemories(limit=3))
        assert [m["id"] for m in out["memories"]] == ["7", "6", "5"]  # newest first
        assert out["total"] == 7 and out["next_offset"] == 3
        assert out["memories"][0]["preview"] == "memory 7"

    def test_paging_walks_everything(self, store):
        _fake, mt = store
        seen = []
        offset = 0
        while True:
            out = _run(mt.listMemories(limit=3, offset=offset))
            seen.extend(m["id"] for m in out["memories"])
            if out["next_offset"] is None:
                break
            offset = out["next_offset"]
        assert seen == ["7", "6", "5", "4", "3", "2", "1"]  # all 7, no dupes

    def test_sort_oldest(self, store):
        _fake, mt = store
        out = _run(mt.listMemories(limit=2, sort="oldest"))
        assert [m["id"] for m in out["memories"]] == ["1", "2"]

    def test_tag_filter(self, store):
        _fake, mt = store
        out = _run(mt.listMemories(limit=10, tag="alpha"))
        assert all("alpha" in (m["tags"] or []) for m in out["memories"])
        assert [m["id"] for m in out["memories"]] == ["7", "5", "3", "1"]  # odd ids, newest first

    def test_row_shape(self, store):
        _fake, mt = store
        out = _run(mt.listMemories(limit=1, sort="oldest"))
        row = out["memories"][0]
        for key in ("id", "preview", "memory_type", "tags", "sensitivity",
                    "created_at", "last_accessed", "usage_count", "source"):
            assert key in row, key
        assert row["preview"] == "memory 1"
        assert row["usage_count"] == 1

    def test_limit_clamped(self, store):
        _fake, mt = store
        out = _run(mt.listMemories(limit=500))
        assert len(out["memories"]) == 7  # everything, no error at huge limit

    def test_no_vector_store(self, monkeypatch):
        import memory_tools

        monkeypatch.setattr(memory_tools, "vector_store", None)
        out = _run(memory_tools.listMemories())
        assert "error" in out
