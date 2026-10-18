"""Central full-config write endpoint (pod-UI persistence design P1).

PUT /api/tools/{tool}/config is the sanctioned writer of per-tool
config.json: validates, lock-out-guards the auth block, writes atomically,
refuses read-only mounts with a 409, mirrors redacted to state_docs in db
mode. Complements the (d) finding correction: the TOOLS-config (masks)
write was already central via save_tools_config's db mode; this endpoint
covers the per-tool config.json surface.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from launcher.management_server import ManagementServer  # noqa: E402
from launcher.service_registry import ServiceRegistry  # noqa: E402

_AUTH = {"Authorization": "Bearer cfg-endpoint-key"}


@pytest.fixture()
def server(monkeypatch):
    import tools.shared.atomic_io as atomic_io
    written = {}

    def fake_write(path, cfg):
        written[str(path)] = cfg

    monkeypatch.setattr(atomic_io, "atomic_write_json", fake_write)
    monkeypatch.setattr("tools.shared.cluster.is_db_mode", lambda: False)

    srv = ManagementServer(service_registry=ServiceRegistry(), port=8294,
                           host="127.0.0.1", api_key="cfg-endpoint-key")
    yield srv, written
    # nothing touched the real config: the write was intercepted


VALID = {
    "name": "simplemcp",
    "auth": {"api_key": "real-key-stays"},
    "transports": {"streamable-http": {"port": 8002}},
}


def test_put_config_writes_atomically(server):
    srv, written = server
    resp = TestClient(srv.app).put("/api/tools/simplemcp/config",
                                   headers=_AUTH, json=VALID)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["changed_keys"]  # something differs from the on-disk config
    assert any(p.endswith("tools/simplemcp/config.json") for p in written)
    assert list(written.values())[0]["auth"]["api_key"] == "real-key-stays"


def test_put_config_requires_auth_block(server):
    srv, _ = server
    bad = {k: v for k, v in VALID.items() if k != "auth"}
    resp = TestClient(srv.app).put("/api/tools/simplemcp/config",
                                   headers=_AUTH, json=bad)
    assert resp.status_code == 400
    assert "api_key" in resp.json()["detail"]


def test_put_config_rejects_readonly_mount(server, monkeypatch):
    import tools.shared.atomic_io as atomic_io

    def ro(path, cfg):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(atomic_io, "atomic_write_json", ro)
    resp = TestClient(server[0].app).put("/api/tools/simplemcp/config",
                                         headers=_AUTH, json=VALID)
    assert resp.status_code == 409
    assert "bundle" in resp.json()["detail"]


def test_put_config_unknown_tool_404(server):
    resp = TestClient(server[0].app).put("/api/tools/nope/config",
                                         headers=_AUTH, json=VALID)
    assert resp.status_code == 404


def test_put_config_path_traversal_400(server):
    resp = TestClient(server[0].app).put("/api/tools/..%2F..%2Fetc/config",
                                         headers=_AUTH, json=VALID)
    assert resp.status_code in (400, 404)  # never a write outside tools/


def test_put_config_db_mode_mirrors_redacted(server, monkeypatch):
    import tools.shared.cluster as cluster_mod
    import tools.shared.state_docs as state_docs

    monkeypatch.setattr("tools.shared.cluster.is_db_mode", lambda: True)
    mirrored = {}

    def fake_save(name, doc):
        mirrored[name] = doc
        return True

    monkeypatch.setattr(state_docs, "save_doc", fake_save)

    resp = TestClient(server[0].app).put("/api/tools/simplemcp/config",
                                         headers=_AUTH, json=VALID)
    assert resp.status_code == 200 and resp.json()["shared_state_updated"] is True
    doc = mirrored["tools_config_simplemcp"]
    assert doc["auth"]["api_key"] == "***"  # redacted mirror
