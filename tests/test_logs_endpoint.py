"""GET /api/logs — launcher-log tail endpoint (admin-gated, like every other
central route). Reads the active root-logger FileHandler's file so it works
on the host and inside the node image alike; grep filters before the tail."""

import logging
import sys

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from launcher.management_server import ManagementServer  # noqa: E402
from launcher.service_registry import ServiceRegistry  # noqa: E402

AUTH = {"Authorization": "Bearer central-test-key"}


@pytest.fixture()
def server():
    return ManagementServer(
        service_registry=ServiceRegistry(),
        port=8297,
        host="127.0.0.1",
        api_key="central-test-key",
    )


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    """A known log file + the endpoint pointed at it."""
    f = tmp_path / "launcher.log"
    f.write_text(
        "\n".join(f"line-{i:03d} simplemcp-{i % 3} filler" for i in range(1, 51))
        + "\n"
    )
    monkeypatch.setattr(
        ManagementServer, "_launcher_log_path", staticmethod(lambda: f)
    )
    return f


def test_tail_returns_last_n_lines(server, log_file):
    client = TestClient(server.app)
    r = client.get("/api/logs", params={"tail": 5}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == str(log_file)
    assert body["lines"] == [f"line-{i:03d} simplemcp-{i % 3} filler" for i in range(46, 51)]
    assert body["total_lines"] == 50


def test_grep_filters_before_tail(server, log_file):
    client = TestClient(server.app)
    r = client.get("/api/logs", params={"tail": 2, "grep": "simplemcp-1"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    # matching lines are i in {1,4,...,49}; the LAST TWO matches win
    assert body["lines"] == [
        "line-046 simplemcp-1 filler",
        "line-049 simplemcp-1 filler",
    ]
    assert body["total_lines"] == 50  # unfiltered count still reported
    assert len(body["lines"]) == 2


def test_grep_is_case_insensitive(server, log_file):
    client = TestClient(server.app)
    r = client.get("/api/logs", params={"tail": 10, "grep": "SIMPLEMCP-2"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["lines"] and all("simplemcp-2" in l for l in body["lines"])


def test_tail_clamps_to_reasonable_maximum(server, log_file):
    client = TestClient(server.app)
    r = client.get("/api/logs", params={"tail": 10_000_000}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["tail"] == 5000


def test_requires_the_admin_key(server, log_file):
    client = TestClient(server.app)
    assert client.get("/api/logs").status_code == 401
    assert client.get("/api/logs", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_missing_log_file_is_not_an_error(server, monkeypatch):
    monkeypatch.setattr(
        ManagementServer, "_launcher_log_path", staticmethod(lambda: None)
    )
    client = TestClient(server.app)
    r = client.get("/api/logs", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["file"] is None and body["lines"] == []


def test_launcher_log_path_prefers_the_file_handler(tmp_path, monkeypatch):
    """Resolution: root-logger FileHandler wins; else logs/launcher.log only
    if it exists; else None."""
    monkeypatch.chdir(tmp_path)  # no logs/launcher.log here
    # strip any handlers pytest left on the root logger
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers[:] = []
    try:
        assert ManagementServer._launcher_log_path() is None
        real = tmp_path / "real.log"
        handler = logging.FileHandler(real)
        root.addHandler(handler)
        assert ManagementServer._launcher_log_path() == real
    finally:
        root.handlers[:] = saved
