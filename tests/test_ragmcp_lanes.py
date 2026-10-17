"""
Unit tests for ragmcp embedding lanes (plans/odysseus-reuse-2026-10-06.md P2).

Covers the pure lane logic in tools/ragmcp/indexer/lanes.py (slug building,
lane listing, fingerprint reading, index pre-flight, search-candidate
resolution) and the dead-endpoint latch in indexer/embedding_gate.py.
No vector store or embedding model needed — stores are fakes.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools" / "ragmcp")):
    if p not in sys.path:
        sys.path.insert(0, p)

from indexer import embedding_gate
from indexer.lanes import (
    check_lane_compatibility,
    collection_fingerprint,
    dense_dim_of,
    is_lane_name,
    lane_fingerprint,
    lane_label,
    lane_slug,
    legacy_lane,
    list_lanes,
    resolve_lane,
    resolve_search_candidates,
    sanitize_slug,
)
from indexer.metadata_config import get_metadata_path

ACTIVE = {"provider": "local", "model": "bge-m3", "dimensions": 1024}
META = {"embedding_provider": "local", "embedding_model": "bge-m3",
        "embedding_dimensions": 1024, "_is_metadata": True}


class FakeInfo:
    def __init__(self, name, points=0, named=None, sparse=False):
        self.name = name
        self.points_count = points
        self.named_vectors = named or {}
        self.has_sparse = sparse

    @property
    def dim(self):
        return next(iter(self.named_vectors.values()), None)

    @property
    def has_dense(self):
        return bool(self.named_vectors)


class _Point:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class FakeStore:
    """Just enough of the VectorStore read surface for the lane logic."""

    def __init__(self, collections=None, metadata=None):
        self.collections = collections or {}   # name -> FakeInfo
        self.metadata = metadata or {}         # name -> payload dict

    def list_collections(self):
        return list(self.collections)

    def get_collection(self, name):
        if name not in self.collections:
            raise Exception(f"Collection `{name}` doesn't exist")
        return self.collections[name]

    def retrieve(self, collection, ids, with_payload=True):
        payload = self.metadata.get(collection)
        if payload is None or not ids:
            return []
        return [_Point(ids[0], payload)]


# ---------------------------------------------------------------------------
# slug building
# ---------------------------------------------------------------------------

def test_sanitize_slug_maps_forbidden_chars():
    assert sanitize_slug("BAAI/bge-m3") == "baai-bge-m3"
    assert sanitize_slug("openrouter/foo:free") == "openrouter-foo-free"
    assert sanitize_slug("Text Embedding 3") == "text-embedding-3"


def test_lane_slug_is_provider_prefixed_and_sanitized():
    slug = lane_slug("local", "bge-m3", 1024)
    assert slug == "local-bge-m3-1024"
    assert lane_slug("azure", "text-embedding-3-large", 3072) == "azure-text-embedding-3-large-3072"


def test_lane_slug_overflow_uses_hash_suffix_not_truncation():
    long_model = "zai-org/glm-4.6v-flash-free-with-an-extremely-long-organization-and-model-name"
    s1 = lane_slug("openrouter", long_model, 2048)
    s2 = lane_slug("openrouter", long_model, 4096)
    assert len(s1) <= 60
    # different dims must never collide even when sanitized prefixes truncate
    assert s1 != s2
    # deterministic
    assert s1 == lane_slug("openrouter", long_model, 2048)


def test_resolve_lane_and_helpers():
    assert resolve_lane("code_index", "local", "bge-m3", 1024) == "code_index-local-bge-m3-1024"
    assert legacy_lane("code_index") == "code_index"
    assert is_lane_name("code_index", "code_index")
    assert is_lane_name("code_index", "code_index-local-bge-m3-1024")
    assert not is_lane_name("code_index", "code_index2-local-bge-m3-1024")
    assert lane_label("code_index", "code_index") == "legacy"
    assert lane_label("code_index", "code_index-local-bge-m3-1024") == "local-bge-m3-1024"


def test_lane_fingerprint_shape():
    assert lane_fingerprint("local", "bge-m3", 1024) == "local:bge-m3:1024"


# ---------------------------------------------------------------------------
# listing + fingerprint reading
# ---------------------------------------------------------------------------

def test_list_lanes_orders_legacy_first():
    store = FakeStore(collections={
        "other-idx": FakeInfo("other-idx"),  # different base entirely
        "idx-local-bge-m3-1024": FakeInfo("idx-local-bge-m3-1024"),
        "idx": FakeInfo("idx"),
        "idx-azure-text-embedding-3-large-3072": FakeInfo("idx-azure-text-embedding-3-large-3072"),
    })
    assert list_lanes(store, "idx") == [
        "idx",
        "idx-azure-text-embedding-3-large-3072",
        "idx-local-bge-m3-1024",
    ]


def test_dense_dim_prefers_named_dense_vector():
    info = FakeInfo("hybrid", named={"dense": 1024})
    assert dense_dim_of(info) == 1024
    solo = FakeInfo("solo", named={"solo-vec": 768})
    assert dense_dim_of(solo) == 768


def test_collection_fingerprint_reads_metadata_point():
    store = FakeStore(metadata={"idx": META})
    fp = collection_fingerprint(store, "idx")
    assert fp == {"provider": "local", "model": "bge-m3", "dimensions": 1024}


def test_collection_fingerprint_absent_or_not_metadata():
    store = FakeStore(metadata={"idx": {"foo": "bar", "_is_metadata": False}})
    assert collection_fingerprint(store, "idx") is None
    assert collection_fingerprint(FakeStore(), "missing") is None


# ---------------------------------------------------------------------------
# index pre-flight
# ---------------------------------------------------------------------------

def test_preflight_passes_fresh_collection():
    store = FakeStore()
    assert check_lane_compatibility(store, "idx-local-bge-m3-1024", "local", "bge-m3", 1024) is None


def test_preflight_passes_matching_dims():
    store = FakeStore(collections={"idx": FakeInfo("idx", named={"dense": 1024})},
                      metadata={"idx": META})
    assert check_lane_compatibility(store, "idx", "local", "bge-m3", 1024) is None


def test_preflight_aborts_on_dim_mismatch_with_actionable_message():
    store = FakeStore(collections={"idx": FakeInfo("idx", named={"dense": 1024})},
                      metadata={"idx": META})
    msg = check_lane_compatibility(store, "idx", "azure", "text-embedding-3-large", 3072)
    assert msg is not None
    assert "1024d" in msg and "3072d" in msg
    assert "clear_index" in msg  # actionable escape hatch


def test_preflight_rejects_dense_indexing_into_sparse_only_collection():
    store = FakeStore(collections={"idx": FakeInfo("idx", named={}, sparse=True)})
    msg = check_lane_compatibility(store, "idx", "local", "bge-m3", 1024)
    assert msg is not None and "sparse-only" in msg


# ---------------------------------------------------------------------------
# search-candidate resolution
# ---------------------------------------------------------------------------

def _store_with_lanes():
    return FakeStore(
        collections={
            "idx": FakeInfo("idx", named={"dense": 1024}),
            "idx-local-bge-m3-1024": FakeInfo("idx-local-bge-m3-1024", named={"dense": 1024}),
            "idx-azure-text-embedding-3-large-3072": FakeInfo(
                "idx-azure-text-embedding-3-large-3072", named={"dense": 3072}),
        },
        metadata={
            "idx": META,
            "idx-local-bge-m3-1024": META,
            "idx-azure-text-embedding-3-large-3072": {
                "embedding_provider": "azure", "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 3072, "_is_metadata": True},
        },
    )


def test_candidates_no_collections_at_all():
    resolved = resolve_search_candidates(FakeStore(), "nothing", mode="auto", active=ACTIVE)
    assert resolved["candidates"] == []
    assert resolved["notes"]


def test_candidates_default_active_lane_plus_matching_legacy():
    resolved = resolve_search_candidates(_store_with_lanes(), "idx", mode="auto", active=ACTIVE)
    names = [c[0] for c in resolved["candidates"]]
    # active lane first; legacy base included (fingerprint matches); azure lane NOT included
    assert names == ["idx-local-bge-m3-1024", "idx"]
    for _coll, _label, preset in resolved["candidates"]:
        assert preset["provider"] == "local" and preset["dimensions"] == 1024


def test_candidates_default_skips_legacy_with_different_fingerprint():
    store = _store_with_lanes()
    store.metadata.pop("idx")  # legacy has no metadata point...
    store.collections["idx"] = FakeInfo("idx", named={"dense": 3072})  # ...and wrong dim
    resolved = resolve_search_candidates(store, "idx", mode="auto", active=ACTIVE)
    names = [c[0] for c in resolved["candidates"]]
    assert names == ["idx-local-bge-m3-1024"]
    assert any("legacy" in n for n in resolved["notes"])


def test_candidates_legacy_without_metadata_falls_back_to_dim_match():
    store = _store_with_lanes()
    store.metadata.pop("idx")  # no provenance
    store.collections["idx"] = FakeInfo("idx", named={"dense": 1024})  # dim matches active
    resolved = resolve_search_candidates(store, "idx", mode="auto", active=ACTIVE)
    names = [c[0] for c in resolved["candidates"]]
    assert "idx" in names  # dim-proxy keeps the legacy lane searchable


def test_candidates_sparse_mode_sweeps_all_lanes_without_presets():
    resolved = resolve_search_candidates(_store_with_lanes(), "idx", mode="sparse", active=ACTIVE)
    assert len(resolved["candidates"]) == 3
    assert all(preset is None for _c, _l, preset in resolved["candidates"])


def test_candidates_all_lanes_optin_embeds_per_fingerprint():
    resolved = resolve_search_candidates(
        _store_with_lanes(), "idx", mode="auto", active=ACTIVE, search_all_lanes=True)
    names = [c[0] for c in resolved["candidates"]]
    assert names == ["idx", "idx-azure-text-embedding-3-large-3072", "idx-local-bge-m3-1024"]
    presets = {c[0]: c[2] for c in resolved["candidates"]}
    assert presets["idx-azure-text-embedding-3-large-3072"]["provider"] == "azure"
    assert presets["idx-local-bge-m3-1024"]["provider"] == "local"


def test_candidates_all_lanes_skips_unknown_fingerprint():
    store = _store_with_lanes()
    store.metadata.pop("idx-azure-text-embedding-3-large-3072")  # no provenance
    resolved = resolve_search_candidates(
        store, "idx", mode="auto", active=ACTIVE, search_all_lanes=True)
    names = [c[0] for c in resolved["candidates"]]
    assert "idx-azure-text-embedding-3-large-3072" not in names
    assert any("no collection metadata" in n for n in resolved["notes"])


# ---------------------------------------------------------------------------
# dead-endpoint latch
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_gate():
    embedding_gate.reset()
    yield
    embedding_gate.reset()


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def test_gate_opens_after_threshold_and_blocks(monkeypatch):
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_THRESHOLD", "2")
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_SECONDS", "60")
    clock = _FakeClock()
    monkeypatch.setattr(embedding_gate.time, "monotonic", clock.monotonic)

    embedding_gate.record_failure("http://ep")
    embedding_gate.check("http://ep")  # below threshold → still open
    embedding_gate.record_failure("http://ep")
    with pytest.raises(embedding_gate.EmbeddingEndpointDown):
        embedding_gate.check("http://ep")


def test_gate_success_resets_streak(monkeypatch):
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_THRESHOLD", "2")
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_SECONDS", "60")
    monkeypatch.setattr(embedding_gate.time, "monotonic", _FakeClock().monotonic)

    embedding_gate.record_failure("http://ep")
    embedding_gate.record_success()
    embedding_gate.record_failure("http://ep")
    embedding_gate.check("http://ep")  # streak was reset → still open


def test_gate_cooldown_expires(monkeypatch):
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_THRESHOLD", "1")
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_SECONDS", "60")
    clock = _FakeClock()
    monkeypatch.setattr(embedding_gate.time, "monotonic", clock.monotonic)

    embedding_gate.record_failure("http://ep")
    with pytest.raises(embedding_gate.EmbeddingEndpointDown):
        embedding_gate.check("http://ep")
    clock.t += 61
    embedding_gate.check("http://ep")  # cooldown over → open again


def test_gate_disabled_by_zero_cooldown(monkeypatch):
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_THRESHOLD", "1")
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_SECONDS", "0")
    embedding_gate.record_failure("http://ep")
    embedding_gate.record_failure("http://ep")
    embedding_gate.check("http://ep")  # latching disabled


# ---------------------------------------------------------------------------
# metadata sidecar
# ---------------------------------------------------------------------------

def test_metadata_path_per_lane(tmp_path):
    legacy = get_metadata_path(str(tmp_path))
    assert legacy.name == "indexing_metadata.json"
    lane = get_metadata_path(str(tmp_path), "local-bge-m3-1024")
    assert lane.name == "indexing_metadata.local-bge-m3-1024.json"
    assert lane.parent == legacy.parent
    assert lane.parent.exists()  # directory auto-created
