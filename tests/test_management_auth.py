"""E3-P0 — central management API authentication.

MCP_MANAGEMENT_API_KEY set → the Bearer key is required on every route;
unset → reads stay open with a startup warning, but MUTATION routes fail
closed (audit P2-2, 2026-10-17) — zero-config local use keeps working for
reads; mutations need the key. The verifier uses hmac.compare_digest on the
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


# ── E3 multi-admin: enabled admins' user keys open the central API ──

@pytest.fixture()
def _admin_central_key(monkeypatch):
    from tools.shared import users_store
    monkeypatch.setattr(
        users_store, "central_tokens",
        lambda: {"admin-central-key-1": {"client_id": "e35admin", "role": "admin"}},
    )


def test_admin_user_key_accepted(management_server, _admin_central_key, caplog):
    """Multi-admin: an enabled admin's own MCP key authenticates on 8200
    (attributed in central.access); revocation = rotate/disable."""
    client = TestClient(management_server.app)
    with caplog.at_level("INFO", logger="launcher.management_server"):
        resp = client.get(
            "/api/disabled-tools",
            headers={"Authorization": "Bearer admin-central-key-1"},
        )
    assert resp.status_code == 200
    assert any("central.access user=e35admin" in r.message for r in caplog.records)


def test_admin_key_revoked_when_not_enabled(management_server, monkeypatch):
    """central_tokens only surfaces ENABLED admins — an omitted key 401s."""
    from tools.shared import users_store
    monkeypatch.setattr(users_store, "central_tokens", lambda: {})
    client = TestClient(management_server.app)
    resp = client.get(
        "/api/disabled-tools",
        headers={"Authorization": "Bearer admin-central-key-1"},
    )
    assert resp.status_code == 401


def test_non_admin_user_key_still_rejected(management_server, _admin_central_key):
    """Only admins get central reach; role=user keys stay 401."""
    client = TestClient(management_server.app)
    resp = client.get(
        "/api/disabled-tools",
        headers={"Authorization": "Bearer some-random-user-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Audit 2026-10-01 P2-1/P2-2
# ---------------------------------------------------------------------------

def test_open_server_refuses_mutations_but_serves_reads(open_server):
    """P2-2: with NO key configured, mutation routes fail closed (403) while
    reads keep the historical keyless behavior. Deployments with a key —
    the documented state — are untouched (every other test in this file)."""
    client = TestClient(open_server.app)
    resp = client.post("/api/users", json={"username": "x", "password": "y"})
    assert resp.status_code == 403
    assert "MCP_MANAGEMENT_API_KEY" in resp.json()["error"]

    resp = client.put("/api/disabled-tools/simplemcp")
    assert resp.status_code == 403

    assert client.get("/health").status_code == 200


def test_keyed_server_mutations_still_work(management_server, _admin_central_key):
    """Iso-check: with a key configured, the mutation middleware is absent."""
    client = TestClient(management_server.app)
    resp = client.post(
        "/api/users",
        headers={"Authorization": f"Bearer {_admin_central_key}"},
        json={"username": " someone ", "password": "pw"},
    )
    assert resp.status_code != 403  # the P2-2 middleware is not installed


def test_resolve_tool_api_key_env_precedence(monkeypatch):
    """P2-1: the 81xx server resolves its key with the MCP port's env-first
    precedence (tools/shared/server_factory._resolve_api_key)."""
    from launcher.env_manager import resolve_tool_api_key

    monkeypatch.delenv("SIMPLEMCP_API_KEY", raising=False)
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    assert resolve_tool_api_key("simplemcp", "cfg-key") == "cfg-key"

    monkeypatch.setenv("MCP_API_KEY", "mcp-key")
    assert resolve_tool_api_key("simplemcp", "cfg-key") == "mcp-key"

    monkeypatch.setenv("SIMPLEMCP_API_KEY", "tool-key")
    assert resolve_tool_api_key("simplemcp", "cfg-key") == "tool-key"

    # dash-bearing tool names normalize to underscores in the env var
    monkeypatch.delenv("SIMPLEMCP_API_KEY")
    monkeypatch.delenv("MCP_API_KEY")
    assert resolve_tool_api_key("my-tool", None) is None
    monkeypatch.setenv("MY_TOOL_API_KEY", "dash-key")
    assert resolve_tool_api_key("my-tool", None) == "dash-key"
