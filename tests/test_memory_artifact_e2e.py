#!/usr/bin/env python3
"""
End-to-end tests for memorymcp's ArtifactStore wire-in (C2 large-text offloading).

Mocks only: no Qdrant/Turso, no S3, no live server. memory_core is stubbed
(test_memory_autouse.py pattern) while reusing the REAL shared models,
scorer, and PII redactor so payload shapes and scoring match production;
the stores are in-memory doubles exercising the same async contracts as
tools/shared/artifact_store.ArtifactStore and the VectorStore/SqlStore ABCs.

Covers:
  1. Large text (> 8192 bytes) upsert -> payload keeps artifact_key +
     text_preview + text_size_bytes, "text" removed, blob in the store
  2. Round trip: getMemory / queryMemory rehydrate the full text
  3. Small text stays inline exactly as before
  4. Threshold boundary: 8192 bytes inline, 8193 bytes artifacted
  5. Legacy payload (plain "text") still reads correctly, store untouched
  6. deleteMemory / decayOrExpire / mergeDuplicates delete the artifact
  7. Update that replaces the artifact removes the old blob; large->large
     update overwrites the same key without a spurious delete
  8. Artifact fetch failure / missing blob degrade to text_preview + warning

Run:
    python -m pytest tests/test_memory_artifact_e2e.py -v
"""

import asyncio
import logging
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

# Path setup must precede the shared.* imports below (repo root, tools/ for
# the `shared` package, tools/memorymcp for the modules under test).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "tools", _PROJECT_ROOT / "tools" / "memorymcp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from shared.store_models import PointStruct
from shared.artifact_store import ArtifactStoreError


def _run(coro):
    """Run a coroutine on a LOCAL event loop (see tests/test_memory_autouse.py)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# In-memory store doubles
# ---------------------------------------------------------------------------

class _ScoredPointLike:
    """Mimics a Qdrant ScoredPoint (id/payload/score) over a stored point."""

    def __init__(self, point, score: float = 0.9):
        self.id = point.id
        self.payload = point.payload
        self.score = score


class FakeVectorStore:
    """In-memory VectorStore double covering the surface memory_tools uses."""

    def __init__(self):
        self.points: dict[str, PointStruct] = {}
        self.deleted_ids: list[str] = []

    def reset(self):
        self.points.clear()
        self.deleted_ids.clear()

    def upsert(self, collection_name, points):
        for p in points:
            self.points[str(p.id)] = p

    def retrieve(self, collection_name, ids, with_payload=True):
        return [self.points[str(i)] for i in ids if str(i) in self.points]

    def query_dense(self, collection_name, vector, limit=10, filter=None):
        return [_ScoredPointLike(p) for p in list(self.points.values())[:limit]]

    def set_payload(self, collection_name, payload, ids):
        for i in ids:
            pt = self.points.get(str(i))
            if pt is not None:
                pt.payload.update(payload)

    def delete(self, collection_name, ids):
        for i in ids:
            if str(i) in self.points:
                del self.points[str(i)]
                self.deleted_ids.append(str(i))

    def iter_all(self, collection_name, with_vectors=False):
        yield from list(self.points.values())

    def iter_points(self):
        return list(self.points.values())


class FakeSqlStore:
    """Vector-only mode: SQL side never available."""

    is_available = False


class FakeArtifactStore:
    """In-memory double of tools/shared/artifact_store.ArtifactStore."""

    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail_load = False

    def reset(self):
        self.blobs.clear()
        self.deleted.clear()
        self.fail_load = False

    async def save(self, data, key, content_type=None, metadata=None):
        self.blobs[key] = data.encode("utf-8") if isinstance(data, str) else data
        return SimpleNamespace(key=key, size_bytes=len(self.blobs[key]))

    async def load(self, key):
        if self.fail_load:
            raise ArtifactStoreError("simulated artifact store outage")
        return self.blobs.get(key)

    async def delete(self, key):
        self.deleted.append(key)
        return self.blobs.pop(key, None) is not None


# ---------------------------------------------------------------------------
# memory_core stub + module fixture (test_memory_autouse.py pattern)
# ---------------------------------------------------------------------------

def _install_memory_core_stub(fake_vector, fake_sql):
    """Install a fake memory_core module reusing the REAL shared helpers.

    memory_item_to_payload / payload_to_memory_hit mirror the memory_core
    implementations field-for-field; the models, scoring, and PII functions
    are the real shared modules, so payload shapes match production.
    """
    repo_root = _PROJECT_ROOT
    tools_dir = repo_root / "tools"
    memorymcp_dir = repo_root / "tools" / "memorymcp"
    for p in (repo_root, tools_dir, memorymcp_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from shared.memory_models import MemoryType, RetentionPolicy, MemoryItem, MemoryHit
    from shared.relevance_scorer import (
        ScoringWeights, score_relevance, compute_recency_decay, compute_usage_boost,
    )
    from shared.pii_redactor import redact_sensitive_text, check_sensitivity

    class _FakeMCP:
        def __init__(self):
            self.registered = []

        def tool(self):
            def decorator(fn):
                self.registered.append(fn)
                return fn
            return decorator

    def _get_now_iso():
        return datetime.now(timezone.utc).isoformat()

    def _parse_memory_type(type_str):
        if not type_str:
            return MemoryType.CONCEPT
        try:
            return MemoryType(type_str.lower())
        except ValueError:
            return MemoryType.CONCEPT

    def _memory_item_to_payload(item):
        # Mirrors memory_core.memory_item_to_payload (kept in sync manually).
        return {
            "text": item.text,
            "memory_type": item.type.value if isinstance(item.type, MemoryType) else item.type,
            "source": item.source,
            "path": item.path,
            "commit": item.commit,
            "file_range": item.file_range,
            "agent_id": item.agent_id,
            "created_at": item.timestamp or _get_now_iso(),
            "last_accessed": item.timestamp or _get_now_iso(),
            "usage_count": 0,
            "retention_policy": item.retention_policy.value if isinstance(item.retention_policy, RetentionPolicy) else item.retention_policy,
            "raw_object_key": item.raw_object_key,
            "provenance": {},
            "sensitivity": "low",
            "tags": item.tags,
            "text_preview": item.text[:200] if len(item.text) > 200 else item.text,
        }

    def _payload_to_memory_hit(payload, score=0.0):
        return MemoryHit(
            id=payload.get("id", ""),
            text=payload.get("text", ""),
            type=_parse_memory_type(payload.get("memory_type")),
            source=payload.get("source", ""),
            tags=payload.get("tags", []),
            score=score,
            recency_score=compute_recency_decay(payload.get("last_accessed")),
            usage_boost=compute_usage_boost(payload.get("usage_count", 0)),
            created_at=payload.get("created_at"),
            last_accessed=payload.get("last_accessed"),
            usage_count=payload.get("usage_count", 0),
            provenance=payload.get("provenance"),
        )

    def _scroll_all(collection_name, **kwargs):
        return fake_vector.iter_points()

    core_stub = types.ModuleType("memory_core")
    core_stub.mcp = _FakeMCP()
    core_stub.logger = logging.getLogger("memorymcp-artifact-test")
    core_stub.SCRIPT_DIR = memorymcp_dir
    core_stub.COLLECTION_NAME = "memory-store-artifact-test"
    core_stub.MGMT_PORT = 0
    core_stub.TOOL_NAME = "memorymcp"
    core_stub.vector_store = fake_vector
    core_stub.sql_store = fake_sql
    core_stub.get_now_iso = _get_now_iso
    core_stub.get_memory_id = lambda: str(uuid.uuid4())
    core_stub.generate_embedding = lambda text: [0.1, 0.2, 0.3]
    core_stub.scroll_all = _scroll_all
    core_stub.parse_memory_type = _parse_memory_type
    core_stub.memory_item_to_payload = _memory_item_to_payload
    core_stub.payload_to_memory_hit = _payload_to_memory_hit
    core_stub.MemoryItem = MemoryItem
    core_stub.RetentionPolicy = RetentionPolicy
    core_stub.ScoringWeights = ScoringWeights
    core_stub.score_relevance = score_relevance
    core_stub.redact_sensitive_text = redact_sensitive_text
    core_stub.check_sensitivity = check_sensitivity
    sys.modules["memory_core"] = core_stub
    return core_stub


@pytest.fixture(scope="module")
def tools_module():
    """Import memory_tools against the stubbed memory_core, then restore."""
    real_memory_core = sys.modules.get("memory_core")
    fake_vector, fake_sql = FakeVectorStore(), FakeSqlStore()
    _install_memory_core_stub(fake_vector, fake_sql)
    sys.modules.pop("memory_tools", None)
    try:
        import memory_tools
        yield memory_tools
    finally:
        if real_memory_core is not None:
            sys.modules["memory_core"] = real_memory_core
        else:
            sys.modules.pop("memory_core", None)


@pytest.fixture()
def artifact_env(tools_module, monkeypatch):
    """Fresh stores per test; route get_artifact_store() to the in-memory fake."""
    tm = tools_module
    tm.vector_store.reset()
    fake_store = FakeArtifactStore()
    monkeypatch.setattr(tm, "get_artifact_store", lambda: fake_store)
    return tm, fake_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _big_text(min_bytes: int = 9600) -> str:
    line = "Artifact round trip payload line with plain ASCII body. "
    text = line * (min_bytes // len(line) + 1)
    assert len(text.encode("utf-8")) > 8192
    return text


def _upsert(tm, text, **kwargs):
    return _run(tm.upsertMemory(text=text, **kwargs))


# ---------------------------------------------------------------------------
# 1 & 2. Write shape + read rehydration
# ---------------------------------------------------------------------------

class TestLargeTextOffloading:

    def test_payload_keeps_reference_not_text(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        mem_id = _upsert(tm, text)

        payload = tm.vector_store.points[mem_id].payload
        key = payload["artifact_key"]
        assert key == f"memories/{mem_id}/memory.txt"
        assert "text" not in payload                      # full text NOT in payload
        assert payload["text_preview"] == text[:200]
        assert payload["text_size_bytes"] == len(text.encode("utf-8"))
        assert store.blobs[key].decode("utf-8") == text   # blob holds the full text

    def test_get_memory_rehydrates_full_text(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        mem_id = _upsert(tm, text)

        result = _run(tm.getMemory(memory_id=mem_id))
        assert text in result                 # full body, not just the preview
        assert "Warning" not in result

    def test_query_memory_rehydrates_full_text(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        _upsert(tm, text)

        result = _run(tm.queryMemory(query="artifact round trip", k=5))
        assert "Found 1 memories" in result
        assert text[:150] in result           # hit rendered from the rehydrated text


# ---------------------------------------------------------------------------
# 3 & 4. Inline behavior preserved
# ---------------------------------------------------------------------------

class TestSmallTextInline:

    def test_small_text_stays_inline(self, artifact_env):
        tm, store = artifact_env
        text = "a small memory that fits inline"
        mem_id = _upsert(tm, text)

        payload = tm.vector_store.points[mem_id].payload
        assert payload["text"] == text
        assert "artifact_key" not in payload
        assert "text_size_bytes" not in payload
        assert store.blobs == {}              # artifact store never touched

    def test_threshold_boundary(self, artifact_env):
        tm, store = artifact_env
        exact_id = _upsert(tm, "b" * 8192)    # at threshold -> inline
        over_id = _upsert(tm, "c" * 8193)     # one byte over -> artifact

        assert "artifact_key" not in tm.vector_store.points[exact_id].payload
        assert "artifact_key" in tm.vector_store.points[over_id].payload
        assert len(store.blobs) == 1

    def test_upsert_redacts_sensitive_text_before_storage(self, artifact_env):
        """C3: the redact -> store leg of the pipeline.

        upsertMemory auto-redacts (check_sensitivity/redact_sensitive_text);
        whatever the storage shape, the PII must never land in the payload
        or the artifact blob.
        """
        tm, store = artifact_env
        ssn = "123-45-6789"
        mem_id = _upsert(tm, f"small memory with ssn {ssn} inside")

        payload = tm.vector_store.points[mem_id].payload
        assert ssn not in payload["text"]
        assert payload["text"] != f"small memory with ssn {ssn} inside"
        assert len(payload["text"]) > 20            # still meaningful content


# ---------------------------------------------------------------------------
# 5. Legacy payloads
# ---------------------------------------------------------------------------

class TestLegacyPayloadCompat:

    @staticmethod
    def _seed_legacy(tm, text, **extra):
        payload = {
            "text": text,
            "text_preview": text[:200],
            "memory_type": "concept",
            "source": "pre-artifact",
            "tags": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "usage_count": 3,
            "retention_policy": "auto-delete",
        }
        payload.update(extra)
        tm.vector_store.upsert(tm.COLLECTION_NAME, [PointStruct(
            id="legacy-1", vector=[0.1, 0.2, 0.3], payload=payload,
        )])

    def test_get_memory_reads_legacy_text(self, artifact_env):
        tm, store = artifact_env
        legacy = "legacy memory stored before the artifact wire-in"
        self._seed_legacy(tm, legacy)

        result = _run(tm.getMemory(memory_id="legacy-1"))
        assert legacy in result
        assert store.blobs == {} and store.deleted == []   # store never touched

    def test_query_memory_reads_legacy_text(self, artifact_env):
        tm, store = artifact_env
        legacy = "legacy memory found through queryMemory"
        self._seed_legacy(tm, legacy)

        result = _run(tm.queryMemory(query="legacy", k=5))
        assert "Found 1 memories" in result
        assert legacy[:150] in result
        assert store.blobs == {} and store.deleted == []


# ---------------------------------------------------------------------------
# 6. Delete-path lifecycle
# ---------------------------------------------------------------------------

class TestDeletePaths:

    def test_delete_memory_deletes_artifact(self, artifact_env):
        tm, store = artifact_env
        mem_id = _upsert(tm, _big_text())
        key = tm.vector_store.points[mem_id].payload["artifact_key"]

        result = _run(tm.deleteMemory(memory_id=mem_id))
        assert result == f"Deleted memory: {mem_id}"
        assert mem_id in tm.vector_store.deleted_ids
        assert key in store.deleted
        assert key not in store.blobs

    def test_delete_memory_without_artifact_never_touches_store(self, artifact_env):
        tm, store = artifact_env
        mem_id = _upsert(tm, "small memory")

        result = _run(tm.deleteMemory(memory_id=mem_id))
        assert result == f"Deleted memory: {mem_id}"
        assert store.deleted == []

    def test_decay_deletes_artifacts(self, artifact_env):
        tm, store = artifact_env
        old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        tm.vector_store.upsert(tm.COLLECTION_NAME, [PointStruct(
            id="old-artifact-1", vector=[0.1],
            payload={
                "text_preview": "old preview",
                "artifact_key": "memories/old-artifact-1/memory.txt",
                "text_size_bytes": 9000,
                "memory_type": "concept",
                "retention_policy": "auto-delete",
                "created_at": old_iso,
                "last_accessed": old_iso,
                "usage_count": 0,
                "tags": [],
            },
        )])
        store.blobs["memories/old-artifact-1/memory.txt"] = b"old body"

        result = _run(tm.decayOrExpire(ttl_days=30, min_usage_count=0, dry_run=False))
        assert "Deleted 1 expired" in result
        assert "old-artifact-1" in tm.vector_store.deleted_ids
        assert "memories/old-artifact-1/memory.txt" in store.deleted
        assert "memories/old-artifact-1/memory.txt" not in store.blobs

    def test_decay_dry_run_keeps_artifact(self, artifact_env):
        tm, store = artifact_env
        old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        tm.vector_store.upsert(tm.COLLECTION_NAME, [PointStruct(
            id="old-artifact-2", vector=[0.1],
            payload={
                "text_preview": "old preview",
                "artifact_key": "memories/old-artifact-2/memory.txt",
                "memory_type": "concept",
                "retention_policy": "auto-delete",
                "created_at": old_iso,
                "last_accessed": old_iso,
                "usage_count": 0,
                "tags": [],
            },
        )])
        store.blobs["memories/old-artifact-2/memory.txt"] = b"old body"

        result = _run(tm.decayOrExpire(ttl_days=30, min_usage_count=0, dry_run=True))
        assert "Would delete 1 memories" in result
        assert "memories/old-artifact-2/memory.txt" in store.blobs   # untouched
        assert store.deleted == []

    def test_merge_duplicates_deletes_loser_artifact(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        id1 = _upsert(tm, text)   # identical twins, equal usage -> p2 loses
        id2 = _upsert(tm, text)

        result = _run(tm.mergeDuplicates(threshold=0.95, dry_run=False))
        assert "deleted 1 memories" in result
        assert id2 in tm.vector_store.deleted_ids
        assert f"memories/{id2}/memory.txt" in store.deleted
        assert f"memories/{id2}/memory.txt" not in store.blobs
        # Winner keeps its artifact
        assert f"memories/{id1}/memory.txt" in store.blobs
        assert f"memories/{id1}/memory.txt" not in store.deleted


# ---------------------------------------------------------------------------
# 7. Update lifecycle
# ---------------------------------------------------------------------------

class TestUpdateLifecycle:

    def test_update_to_small_text_deletes_old_artifact(self, artifact_env):
        tm, store = artifact_env
        mem_id = _upsert(tm, _big_text())
        old_key = tm.vector_store.points[mem_id].payload["artifact_key"]

        _upsert(tm, "now a small update", memory_id=mem_id)

        payload = tm.vector_store.points[mem_id].payload
        assert payload["text"] == "now a small update"
        assert "artifact_key" not in payload
        assert old_key in store.deleted                   # no orphaned blob

    def test_update_large_to_large_overwrites_same_key(self, artifact_env):
        tm, store = artifact_env
        mem_id = _upsert(tm, _big_text())
        key = tm.vector_store.points[mem_id].payload["artifact_key"]

        new_text = _big_text() + " version two"
        _upsert(tm, new_text, memory_id=mem_id)

        payload = tm.vector_store.points[mem_id].payload
        assert payload["artifact_key"] == key
        assert key not in store.deleted                   # same key: overwrite
        assert store.blobs[key].decode("utf-8") == new_text


# ---------------------------------------------------------------------------
# 8. Degraded reads
# ---------------------------------------------------------------------------

class TestDegradedReads:

    def test_fetch_failure_degrades_to_preview_with_warning(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        mem_id = _upsert(tm, text)
        store.fail_load = True                            # store "unreachable"

        result = _run(tm.getMemory(memory_id=mem_id))
        assert text[:200] in result                       # preview is shown
        assert "Warning" in result
        assert text not in result                         # full text unavailable

        found = _run(tm.queryMemory(query="anything", k=5))
        assert text[:150] in found                        # no raise; preview-backed hit

    def test_missing_artifact_degrades_to_preview_with_warning(self, artifact_env):
        tm, store = artifact_env
        text = _big_text()
        mem_id = _upsert(tm, text)
        store.blobs.clear()                               # blob vanished

        result = _run(tm.getMemory(memory_id=mem_id))
        assert text[:200] in result
        assert "artifact missing from store" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
