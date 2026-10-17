"""Cluster state-plane fixes (2026-09-25 review round).

Covers: CAS same-stamp self-bump + microsecond stamps (FIX 4), init-retry
after a failed first backend connect (FIX 5), env deletion tombstones
(FIX 3), mirror-outcome reporting (FIX 2), fan-out HTTP refusals no longer
marking healthy siblings unreachable (FIX 6), and the new-style launcher
entry-point wiring (FIX 1) / UI discovery union + discovery host (FIX 7/8).

Fake-backend based (conftest db_backend); no live backend, no live DB.
"""

import asyncio
import logging
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for pth in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tools")):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from tools.shared import cluster, state_docs  # noqa: E402


# ---------------------------------------------------------------------------
# FIX 4 — CAS stamp: microsecond precision + self-bump
# ---------------------------------------------------------------------------

def test_now_iso_has_microsecond_precision():
    stamp = state_docs._now_iso()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00", stamp), stamp


def test_stamp_plus_1us():
    assert state_docs._stamp_plus_1us(
        "2026-09-25T23:59:59.999999+00:00") == "2026-09-26T00:00:00.000000+00:00"
    assert state_docs._stamp_plus_1us(
        "2026-09-25T12:00:00.000000+00:00") == "2026-09-25T12:00:00.000001+00:00"


def test_cas_never_wins_against_own_stamp(db_backend, monkeypatch):
    """Two same-microsecond writers holding the same expected version can
    no longer BOTH succeed: the winner's stamp must move (bumped by 1µs),
    so the second writer's guard fails. Deterministic via a frozen clock."""
    frozen = "2026-09-25T12:00:00.000000+00:00"
    monkeypatch.setattr(state_docs, "_now_iso", lambda: frozen)

    assert state_docs.save_doc("cas_doc", {"v": 1})
    _, version = state_docs.load_doc_versioned("cas_doc")
    assert version == frozen

    # Writer A: its fresh stamp equals the row's stamp → the row's version
    # still MOVES (bump), and the write succeeds.
    assert state_docs.save_doc_cas("cas_doc", {"v": 2}, frozen)
    _, version_a = state_docs.load_doc_versioned("cas_doc")
    assert version_a == "2026-09-25T12:00:00.000001+00:00"

    # Writer B: same original expected version → loses (row stamp moved).
    assert state_docs.save_doc_cas("cas_doc", {"v": 3}, frozen) is False
    assert state_docs.load_doc_versioned("cas_doc")[0] == {"v": 2}


def test_cas_second_precision_regression(db_backend):
    """Without a frozen clock: the pre-fix failure mode was two writers in
    the same second both returning True. The row stamp must ALWAYS change
    on a successful CAS write, whatever the wall clock does."""
    assert state_docs.save_doc("race_doc", {"w": 1})
    _, v1 = state_docs.load_doc_versioned("race_doc")
    assert state_docs.save_doc_cas("race_doc", {"w": 2}, v1)
    _, v2 = state_docs.load_doc_versioned("race_doc")
    assert v2 != v1  # stamp strictly advanced even within the same second


# ---------------------------------------------------------------------------
# FIX 5 — failed first connect must not latch for the process lifetime
# ---------------------------------------------------------------------------

def test_failed_first_connect_retries_on_next_call(db_backend, monkeypatch):
    import tools.shared.sql_store as sql_store_mod

    calls = {"n": 0}

    def flaky_factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient backend outage at first use")
        return db_backend  # the fixture's FakeSqlStore

    monkeypatch.setattr(sql_store_mod, "get_sql_store", flaky_factory)

    # first use: backend unavailable → fail-open contract (None)
    assert state_docs.load_doc("retry_doc") is None

    # second call: the init is RETRIED and now succeeds
    assert state_docs.save_doc("retry_doc", {"a": 1}) is True
    assert state_docs.load_doc("retry_doc") == {"a": 1}
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# FIX 3 — env deletion tombstones
# ---------------------------------------------------------------------------

def test_env_delete_writes_tombstone_not_pop(db_backend):
    cluster.mirror_env("GONE", "v")
    cluster.mirror_env("GONE", None)
    doc = state_docs.load_doc(cluster.DOC_ENV)
    assert cluster.is_env_tombstone(doc["GONE"])
    assert doc["GONE"] == {"__deleted__": True}  # JSON-serializable shape


def test_load_env_overrides_filters_tombstones(db_backend):
    cluster.mirror_env("ALIVE", "v")
    cluster.mirror_env("DEAD", None)
    assert cluster.load_env_overrides() == {"ALIVE": "v"}


def test_adopt_tombstone_removes_from_env_and_local_file(
        db_backend, monkeypatch, tmp_path):
    import os
    from launcher import env_manager

    # writable local .env (real repo .env must never be touched by tests)
    env_file = tmp_path / ".env"
    env_file.write_text("GONE=stale\nKEEP=here\n")
    monkeypatch.setattr(env_manager, "PROJECT_ROOT", tmp_path)

    monkeypatch.setenv("GONE", "stale")
    cluster.mirror_env("GONE", "stale")
    cluster.mirror_env("GONE", None)

    applied = cluster.adopt_env_overrides()
    assert applied == ["GONE"]
    assert "GONE" not in os.environ
    content = env_file.read_text()
    assert "# GONE=stale" in content  # commented out, not deleted
    assert "KEEP=here" in content

    # idempotent: second adopt has nothing left to remove
    assert cluster.adopt_env_overrides() == []


def test_adopt_tombstone_with_explicit_target(db_backend, monkeypatch,
                                              tmp_path):
    from launcher import env_manager

    monkeypatch.setattr(env_manager, "PROJECT_ROOT", tmp_path)
    target = {"GONE": "stale"}
    cluster.mirror_env("GONE", "stale")
    cluster.mirror_env("GONE", None)

    assert cluster.adopt_env_overrides(env=target) == ["GONE"]
    assert "GONE" not in target


def test_adopt_skips_malformed_doc_entries(db_backend):
    state_docs.save_doc(cluster.DOC_ENV, {"GOOD": "v", "BAD": 42,
                                          "TOMB": {"__deleted__": True}})
    assert cluster.load_env_overrides() == {"GOOD": "v"}


# ---------------------------------------------------------------------------
# FIX 2 — mirror booleans surface as warning + response field (db mode only)
# ---------------------------------------------------------------------------

def test_mirror_failure_warns_in_db_mode(db_backend, monkeypatch, caplog):
    monkeypatch.setattr(state_docs, "save_doc_cas", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger="tools.shared.cluster"):
        assert cluster.mirror_env("MY_VAR", "v") is False
        assert cluster.mirror_auth("webmcp", "k") is False
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("env_overrides" in m and "MY_VAR" in m for m in msgs)
    assert any("auth_overrides" in m and "webmcp" in m for m in msgs)


def test_mirror_silent_in_json_mode(monkeypatch, caplog):
    """Non-db fail-open is NORMAL — False with no warning spam."""
    import tools.shared.sql_store as sql_store_mod

    def _boom():
        raise AssertionError("must not resolve a backend in json mode")
    monkeypatch.setattr(sql_store_mod, "get_sql_store", _boom)
    with caplog.at_level(logging.WARNING, logger="tools.shared.cluster"):
        assert cluster.mirror_env("MY_VAR", "v") is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_set_env_value_reports_mirror_outcome(db_backend, monkeypatch,
                                              tmp_path):
    from launcher import env_manager

    monkeypatch.setattr(env_manager, "PROJECT_ROOT", tmp_path)
    import tools.shared.cluster as cluster_mod
    monkeypatch.setattr(cluster_mod, "mirror_env", lambda n, v: False)
    assert env_manager.set_env_value("REPORT_VAR", "x", persist=False) is False
    monkeypatch.setattr(cluster_mod, "mirror_env", lambda n, v: True)
    assert env_manager.set_env_value("REPORT_VAR", "x", persist=False) is True


def test_delete_env_value_reports_mirror_outcome(db_backend, monkeypatch,
                                                 tmp_path):
    from launcher import env_manager

    monkeypatch.setattr(env_manager, "PROJECT_ROOT", tmp_path)
    import tools.shared.cluster as cluster_mod
    monkeypatch.setattr(cluster_mod, "mirror_env", lambda n, v: False)
    assert env_manager.delete_env_value("REPORT_VAR", persist=False) is False
    monkeypatch.setattr(cluster_mod, "mirror_env", lambda n, v: True)
    assert env_manager.delete_env_value("REPORT_VAR", persist=False) is True


_AUTH = {"Authorization": "Bearer cluster-test-key"}


def test_env_put_response_carries_shared_state_field(monkeypatch):
    from fastapi.testclient import TestClient

    from launcher.management_server import ManagementServer
    from launcher.service_registry import ServiceRegistry
    from launcher import management_server as ms

    monkeypatch.setattr(ms, "load_env_schema",
                        lambda name: {"MY_VAR": {"type": "string"}})
    monkeypatch.setattr(ms, "set_env_value", lambda *a, **k: False)
    monkeypatch.setattr(ms, "get_env_values", lambda name: {})

    server = ManagementServer(service_registry=ServiceRegistry(), port=8297,
                              host="127.0.0.1", api_key="cluster-test-key")
    client = TestClient(server.app)
    resp = client.put("/api/tools/simplemcp/env",
                      headers=_AUTH, json={"variables": {"MY_VAR": "x"}})
    assert resp.status_code == 200
    assert resp.json()["shared_state_updated"] is False


def test_env_delete_response_carries_shared_state_field(monkeypatch):
    from fastapi.testclient import TestClient

    from launcher.management_server import ManagementServer
    from launcher.service_registry import ServiceRegistry
    from launcher import management_server as ms

    monkeypatch.setattr(ms, "load_env_schema",
                        lambda name: {"MY_VAR": {"type": "string"}})
    monkeypatch.setattr(ms, "delete_env_value", lambda *a, **k: False)

    server = ManagementServer(service_registry=ServiceRegistry(), port=8296,
                              host="127.0.0.1", api_key="cluster-test-key")
    client = TestClient(server.app)
    resp = client.delete("/api/tools/simplemcp/env/MY_VAR", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["shared_state_updated"] is False


def test_auth_put_response_carries_shared_state_field(monkeypatch):
    """Auth PUT on a REAL tool config (read) with the atomic write
    intercepted — nothing touches tools/simplemcp/config.json."""
    import json as json_mod
    from fastapi.testclient import TestClient

    import tools.shared.atomic_io as atomic_io
    from launcher.management_server import ManagementServer
    from launcher.service_registry import ServiceRegistry
    from launcher import management_server as ms

    written = {}
    monkeypatch.setattr(atomic_io, "atomic_write_json",
                        lambda path, cfg: written.update({str(path): cfg}))
    monkeypatch.setenv("MCP_STATE_BACKEND", "db")
    import tools.shared.cluster as cluster_mod
    monkeypatch.setattr(cluster_mod, "mirror_auth", lambda t, k: False)

    server = ManagementServer(service_registry=ServiceRegistry(), port=8295,
                              host="127.0.0.1", api_key="cluster-test-key")
    client = TestClient(server.app)
    resp = client.put("/api/tools/simplemcp/auth",
                      headers=_AUTH, json={"api_key": "test-mirrored-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["shared_state_updated"] is False
    # the write was intercepted — the real config is untouched
    assert written, "atomic_write_json should have been called"
    cfg_path = PROJECT_ROOT / "tools" / "simplemcp" / "config.json"
    real = json_mod.loads(cfg_path.read_text())
    assert real["auth"]["api_key"] != "test-mirrored-key"


# ---------------------------------------------------------------------------
# FIX 6 — fan-out: HTTP refusals must not mark a healthy sibling unreachable
# ---------------------------------------------------------------------------

def _fanout_failure_is_connection_error_unit():
    import httpx

    from launcher.management_server import _fanout_failure_is_connection_error
    assert _fanout_failure_is_connection_error(httpx.ConnectError("refused"))
    assert _fanout_failure_is_connection_error(httpx.TimeoutException("slow"))
    assert _fanout_failure_is_connection_error(OSError("network down"))
    assert not _fanout_failure_is_connection_error(RuntimeError("weird"))


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeAsyncClient:
    """httpx.AsyncClient double whose post() is driven by _behavior."""

    behavior = None  # set per-test: status int or exception instance

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        b = type(self).behavior
        if isinstance(b, Exception):
            raise b
        return _FakeResponse(b)


async def _run_fanout(monkeypatch, db_backend, behavior):
    import httpx

    from launcher import management_server as ms
    from tools.shared.function_masks import MaskRequest

    monkeypatch.setenv("MCP_NODE_NAME", "node1")
    cluster.register_node("node1", "http://node1:8200")
    cluster.register_node("node2", "http://node2:8200")

    marked = []
    real_mark = cluster.mark_unreachable

    def recording_mark(name):
        marked.append(name)
        return real_mark(name)  # apply the real mark to the shared doc

    monkeypatch.setattr(cluster, "mark_unreachable", recording_mark)

    _FakeAsyncClient.behavior = behavior
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    ms._schedule_mask_fanout(MaskRequest(
        server_name="simplemcp", tool_name="echo", masked=True))
    pending = [t for t in ms._mask_fanout_tasks if not t.done()]
    assert pending, "fan-out task was not scheduled"
    await asyncio.gather(*pending)
    return marked


@pytest.mark.asyncio
async def test_fanout_http_401_does_not_mark_unreachable(db_backend,
                                                         monkeypatch,
                                                         caplog):
    marked = await _run_fanout(monkeypatch, db_backend, 401)
    assert marked == []  # reachable node keeps its fan-out membership
    # the refusal is loudly logged with status + peer
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("node2" in m and "401" in m for m in msgs)
    assert cluster.sibling_nodes("node1") == {"node2": "http://node2:8200"}


@pytest.mark.asyncio
async def test_fanout_http_500_does_not_mark_unreachable(db_backend,
                                                         monkeypatch):
    marked = await _run_fanout(monkeypatch, db_backend, 500)
    assert marked == []
    assert cluster.sibling_nodes("node1") == {"node2": "http://node2:8200"}


@pytest.mark.asyncio
async def test_fanout_connect_error_still_marks_unreachable(db_backend,
                                                            monkeypatch):
    import httpx

    marked = await _run_fanout(
        monkeypatch, db_backend, httpx.ConnectError("connection refused"))
    assert marked == ["node2"]  # genuinely down → dropped from fan-out
    assert cluster.sibling_nodes("node1") == {}  # excluded until re-register


def test_fanout_failure_classifier_unit():
    _fanout_failure_is_connection_error_unit()


# ---------------------------------------------------------------------------
# FIX 1 — new-style entry point joins the M4 plane
# ---------------------------------------------------------------------------

def test_resolve_management_api_key(monkeypatch):
    from launcher.__main__ import _resolve_management_api_key

    monkeypatch.setenv("MCP_MANAGEMENT_API_KEY", "env-key")
    assert _resolve_management_api_key(None) == "env-key"
    assert _resolve_management_api_key("cli-key") == "cli-key"  # CLI wins
    monkeypatch.delenv("MCP_MANAGEMENT_API_KEY", raising=False)
    assert _resolve_management_api_key(None) is None
    assert _resolve_management_api_key("cli-key") == "cli-key"


def test_main_entry_point_wiring():
    """The new-style entry point must replicate the legacy M4 semantics:
    adopt overrides at import, register the node at management-server
    startup, and fall back to the env key for the central API."""
    src = (PROJECT_ROOT / "launcher" / "__main__.py").read_text()
    assert "adopt_cluster_overrides()" in src            # adopt at import
    assert "is_db_mode()" in src                         # gated like legacy
    assert "cluster.register_node(" in src               # node registration
    assert "MCP_NODE_NAME" in src and "MCP_NODE_CENTRAL_URL" in src
    assert "_resolve_management_api_key(self.args.api_key)" in src
    assert 'os.environ.get("MCP_MANAGEMENT_API_KEY")' in src


# ---------------------------------------------------------------------------
# FIX 7 / FIX 8 — UI discovery union + discovery host override
# ---------------------------------------------------------------------------

def test_merge_discovered_tools_is_union():
    from launcher.tools_config import merge_discovered_tools

    config = {"tools": {"simplemcp": ["double", "masked_fn"]}}
    merged = merge_discovered_tools(config, {"simplemcp": ["double", "new"]})
    # masked_fn (hidden from live discovery) survives; new tools are added
    assert merged["tools"]["simplemcp"] == ["double", "masked_fn", "new"]
    # missing "tools" key is created
    merged = merge_discovered_tools({}, {"webmcp": ["b", "a"]})
    assert merged["tools"]["webmcp"] == ["a", "b"]


def test_ui_discovery_uses_union_and_host_override():
    ui_src = (PROJECT_ROOT / "mcp_ui" / "management_ui.py").read_text()
    assert "merge_discovered_tools(" in ui_src          # union, never replace
    assert 'os.environ.get("MCP_DISCOVERY_HOST", "127.0.0.1")' in ui_src
    assert "http://127.0.0.1:{tool_info.mcp_port}/mcp" not in ui_src


def test_compose_ui_sets_discovery_host():
    compose = (PROJECT_ROOT / "deploy" / "compose-common.yml").read_text()
    assert "MCP_DISCOVERY_HOST: node1" in compose
