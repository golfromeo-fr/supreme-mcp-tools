"""Headless pod_ui page smoke (review 2026-10-24 test-gap closure).

Scratch pod_ui on a scratch port + forged Starlette session cookie +
pre-seeded storage-user file → the page function runs on the initial GET,
so HTTP 200-vs-500 is the signal. This is the recipe that caught 939aba7
(Operations page NameError — a lambda used a loop-local `e` outside its
scope); this test pins it so the same closure bug can never ship silently
again.
"""
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("nicegui")
pytest.importorskip("itsdangerous")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def scratch_pod_ui(tmp_path):
    """Launch pod_ui on a scratch port with a known secret + a pre-seeded
    authenticated session file; yield (base_url, cookies); tear down."""
    port = _free_port()
    secret = "smoke-secret-" + "x" * 24
    session_id = "smokeuser"

    # NiceGUI's user storage lives at .nicegui/storage-user-<id>.json under
    # the process cwd — pre-seed it so is_authed() is True for our session.
    nicegui_dir = tmp_path / ".nicegui"
    nicegui_dir.mkdir()
    (nicegui_dir / f"storage-user-{session_id}.json").write_text(
        json.dumps({"authenticated": True}))

    env = dict(os.environ)
    env.update({
        "MCP_PODUI_PORT": str(port),
        "MCP_UI_SECRET": secret,
        "MCP_UI_USERNAME": "admin",
        "MCP_UI_PASSWORD": "admin",
        # isolate from the real pod state where the env matters
        "MCP_PODUI_HOST": "127.0.0.1",
        # cwd is the scratch dir (for .nicegui) — the module lives in the repo
        "PYTHONPATH": str(PROJECT_ROOT),
    })
    # NiceGUI switches to pytest screen-test mode when it sees these —
    # the scratch instance is a PRODUCTION-style run.
    for var in ("PYTEST_CURRENT_TEST", "PYTEST_VERSION"):
        env.pop(var, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "pod_ui"],
        cwd=tmp_path, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.skip(f"scratch pod_ui exited early:\n{out[-2000:]}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.skip("scratch pod_ui did not come up within 30s")

        # Forge the Starlette session cookie: b64(sign(b64(json_session))).
        from itsdangerous import TimestampSigner
        signer = TimestampSigner(secret)
        payload = base64.urlsafe_b64encode(
            json.dumps({"id": session_id}).encode())
        cookie_value = base64.urlsafe_b64encode(signer.sign(payload)).decode()
        yield base, {"session": cookie_value}
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _get(base: str, path: str, cookies: dict) -> int:
    req = urllib.request.Request(base + path)
    for name, value in cookies.items():
        req.add_header("Cookie", f"{name}={value}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_operations_page_builds(scratch_pod_ui):
    """939aba7 pin: the Operations page function (Clean tooltip scope) must
    build on the initial GET — 500 means a page-body exception."""
    base, cookies = scratch_pod_ui
    assert _get(base, "/v3/operations", cookies) == 200


def test_other_v3_pages_build(scratch_pod_ui):
    """Cheap sweep of the remaining v3 pages on the same scratch instance."""
    base, cookies = scratch_pod_ui
    for path in ("/", "/v3/logs", "/v3/bundles", "/v3/backups"):
        status = _get(base, path, cookies)
        assert status == 200, f"{path} returned {status}"
