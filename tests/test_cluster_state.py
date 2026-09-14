"""M4 — cluster node registry + override adoption unit tests (fake conn)."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for pth in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tools")):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from tools.shared import cluster, state_docs  # noqa: E402


# M5: the fake trio + db_backend fixture moved to tests/_fakes/sql_store.py
# and tests/conftest.py (one canonical definition for all three files).


def test_node_registry_round_trip(db_backend):
    assert cluster.register_node("node1", "http://node1:8200")
    assert cluster.register_node("node2", "http://node2:8200")
    sibs = cluster.sibling_nodes("node1")
    assert sibs == {"node2": "http://node2:8200"}
    # re-register updates, not duplicates
    cluster.register_node("node1", "http://node1:9200")
    assert cluster.sibling_nodes("node2") == {"node1": "http://node1:9200"}


def test_env_mirror_and_delete(db_backend):
    cluster.mirror_env("BRAVE_API_KEY", "new-key")
    cluster.mirror_env("OTHER", "v")
    cluster.mirror_env("OTHER", None)  # deletion
    from tools.shared import state_docs
    doc = state_docs.load_doc(cluster.DOC_ENV)
    assert doc == {"BRAVE_API_KEY": "new-key"}


def test_adopt_env_overrides(db_backend, monkeypatch):
    cluster.mirror_env("ADOPT_ME", "from-cluster")
    monkeypatch.delenv("ADOPT_ME", raising=False)
    applied = cluster.adopt_env_overrides()
    assert "ADOPT_ME" in applied
    import os
    assert os.environ["ADOPT_ME"] == "from-cluster"
    # already-correct value is not re-applied/reported
    assert "ADOPT_ME" not in cluster.adopt_env_overrides()


def test_adopt_auth_overrides(db_backend, tmp_path):
    tool_dir = tmp_path / "tools" / "webmcp"
    tool_dir.mkdir(parents=True)
    cfg = tool_dir / "config.json"
    cfg.write_text(json.dumps({"auth": {"api_key": "old-key"}}))

    cluster.mirror_auth("webmcp", "new-key")
    applied = cluster.adopt_auth_overrides(tools_dir=tmp_path / "tools")
    assert applied == ["webmcp"]
    assert json.loads(cfg.read_text())["auth"]["api_key"] == "new-key"
    # no-op when already converged
    assert cluster.adopt_auth_overrides(tools_dir=tmp_path / "tools") == []


def test_json_mode_is_noop(monkeypatch):
    import tools.shared.sql_store as sql_store_mod

    def _boom():
        raise AssertionError("must not resolve a backend in json mode")
    monkeypatch.setattr(sql_store_mod, "get_sql_store", _boom)
    assert cluster.register_node("x", "http://x") is False
    assert cluster.sibling_nodes("x") == {}
    assert cluster.mirror_env("V", "v") is False


def test_is_db_mode_gate(db_backend, monkeypatch):
    assert cluster.is_db_mode() is True
    monkeypatch.setenv("MCP_STATE_BACKEND", "json")
    assert cluster.is_db_mode() is False


def test_register_node_survives_concurrent_writer(db_backend, monkeypatch):
    """A sibling registering between our read and write must not be
    dropped: the CAS retry re-applies our key on the winner's copy."""
    real_cas = state_docs.save_doc_cas
    calls = {"n": 0}

    def racy_cas(name, doc, version):
        calls["n"] += 1
        if calls["n"] == 1:
            # concurrent winner: registers 'ghost' against the same version
            assert real_cas(name, {"ghost": {
                "central_url": "http://ghost:8200",
                "registered_at": "2026-09-13T00:00:00+00:00"}}, version)
            return False  # our write loses the race
        return real_cas(name, doc, version)

    monkeypatch.setattr(state_docs, "save_doc_cas", racy_cas)
    assert cluster.register_node("node1", "http://node1:8200")
    assert calls["n"] == 2  # one lost race, one winning retry
    doc = state_docs.load_doc(cluster.DOC_NODES)
    assert doc["node1"]["central_url"] == "http://node1:8200"
    assert doc["ghost"]["central_url"] == "http://ghost:8200"


def test_unreachable_marking_and_recovery(db_backend):
    """Failed fan-out peers drop out of sibling_nodes until re-registered."""
    cluster.register_node("n1", "http://n1:8200")
    cluster.register_node("n2", "http://n2:8200")
    assert cluster.sibling_nodes("n1") == {"n2": "http://n2:8200"}

    assert cluster.mark_unreachable("n2")
    assert cluster.sibling_nodes("n1") == {}  # excluded from fan-out
    # but still visible when asked (debug/ops)
    assert cluster.sibling_nodes("n1", include_unreachable=True) == \
        {"n2": "http://n2:8200"}

    # n2 boots again → re-registration clears the mark
    cluster.register_node("n2", "http://n2:8200")
    assert cluster.sibling_nodes("n1") == {"n2": "http://n2:8200"}


def test_mark_unreachable_unknown_node_is_noop(db_backend):
    assert cluster.mark_unreachable("ghost") is False


def test_register_node_clears_stale_mark(db_backend):
    cluster.register_node("n1", "http://n1:8200")
    cluster.mark_unreachable("n1")
    doc = state_docs.load_doc(cluster.DOC_NODES)
    assert "unreachable_since" in doc["n1"]
    cluster.register_node("n1", "http://n1:8200")
    doc = state_docs.load_doc(cluster.DOC_NODES)
    assert "unreachable_since" not in doc["n1"]
