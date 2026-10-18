"""The OPTIONAL nvidia-free embedding preset (2026-10-18 user decision).

OpenRouter's free NVIDIA embeddings are offered as a POSSIBILITY only —
availability is not guaranteed, so: the preset is never a default, failures
latch through embedding_gate, and the P2 lane machinery gives the fingerprint
its own collection (existing indexes are untouched). These tests mock the
HTTP layer (respx) — no network.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "tools" / "ragmcp",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import respx  # noqa: E402
from httpx import Response  # noqa: E402

from ragmcp_fastmcp import EMBEDDING_MODEL_PRESETS, _embed_query  # noqa: E402


def test_preset_exists_and_is_not_the_default():
    preset = EMBEDDING_MODEL_PRESETS["nvidia-free"]
    assert preset["provider"] == "openrouter"
    assert preset["dimensions"] == 2048
    assert "nvidia" in preset["model"].lower()
    # iso: the local defaults are untouched
    assert EMBEDDING_MODEL_PRESETS["auto"]["provider"] == "local"
    assert EMBEDDING_MODEL_PRESETS["fast"]["provider"] == "local"


def test_lane_slug_stable_for_the_nvidia_fingerprint():
    from indexer.lanes import lane_slug
    a = lane_slug("openrouter", "nvidia/nemotron-nemotron-nano-9b-v2", 2048)
    b = lane_slug("openrouter", "nemotron-nemotron-nano-9b-v2", 2048)
    assert a == b  # org prefix must not split the lane
    assert "baai" not in a and a.startswith("openrouter-")


@pytest.mark.asyncio
async def test_embed_query_openrouter_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")
    preset = EMBEDDING_MODEL_PRESETS["nvidia-free"]
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/embeddings").mock(return_value=Response(
            200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
        vec = await _embed_query("hello", preset)
    assert vec == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_query_openrouter_outage_latches(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY", "test-key")
    monkeypatch.setenv("RAGMCP_EMBED_LATCH_THRESHOLD", "1")  # latch on first failure
    from indexer import embedding_gate
    embedding_gate._state.update({"failures": 0, "latched_until": 0.0})  # isolate
    preset = EMBEDDING_MODEL_PRESETS["nvidia-free"]
    with respx.mock(base_url="https://openrouter.ai") as mock:
        route = mock.post("/api/v1/embeddings").mock(return_value=Response(503, json={}))
        with pytest.raises(Exception):
            await _embed_query("hello", preset)
        assert route.call_count == 1
        # the gate latched the endpoint: the NEXT call refuses without HTTP
        with pytest.raises(Exception) as exc:
            await _embed_query("hello", preset)
        assert route.call_count == 1  # no second request left the process
        assert "latched" in str(exc.value).lower()
    embedding_gate._state.update({"failures": 0, "latched_until": 0.0})


@pytest.mark.asyncio
async def test_embed_query_openrouter_without_key_names_local_route(monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    preset = EMBEDDING_MODEL_PRESETS["nvidia-free"]
    with pytest.raises(ValueError) as exc:
        await _embed_query("hello", preset)
    assert "OPENROUTER_KEY" in str(exc.value)
    assert "auto" in str(exc.value)  # points at the offline presets
