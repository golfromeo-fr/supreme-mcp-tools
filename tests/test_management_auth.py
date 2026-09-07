"""E3-P0 — central management API authentication.

MCP_MANAGEMENT_API_KEY set → the Bearer key is required on every route;
unset → the API stays open with a startup warning (pre-E3 behavior, kept
for zero-config local use). The verifier uses hmac.compare_digest on the
Bearer scheme only (HTTPBearer — X-API-Key headers are NOT accepted here).
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


@pytest.fixture()
def management_server(monkeypatch):
    """A ManagementServer whose app is testable without binding a port."""
    server = ManagementServer(
        service_registry=ServiceRegistry(),
        port=8299,
        host="127.0.0.1",
        api_key="central-test-key",
    )
    return server


@pytest.fixture()
def open_server():
    return ManagementServer(
        service_registry=ServiceRegistry(),
        port=8298,
        host="127.0.0.1",
        api_key=None,  # today's default: OPEN
    )


def test_set_key_rejects_missing_header(management_server):
    client = TestClient(management_server.app)
    resp = client.get("/api/disabled-tools")
    assert resp.status_code == 401


def test_set_key_rejects_wrong_key(management_server):
    client = TestClient(management_server.app)
    resp = client.get(
        "/api/disabled-tools", headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401


def test_set_key_accepts_correct_bearer(management_server):
    client = TestClient(management_server.app)
    resp = client.get(
        "/api/disabled-tools", headers={"Authorization": "Bearer central-test-key"}
    )
    assert resp.status_code == 200
    assert "disabled_tools" in resp.json()


def test_unset_key_stays_open(open_server):
    """Pre-E3 behavior preserved for zero-config local deployments."""
    client = TestClient(open_server.app)
    resp = client.get("/api/disabled-tools")  # no header at all
    assert resp.status_code == 200
