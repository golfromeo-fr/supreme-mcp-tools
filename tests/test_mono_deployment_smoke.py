"""Host-mono deployment smoke (2026-10-27).

Mono mode = no ``MCP_AUTH_MODE=multi``: one static system key per tool, no
user store, no identity gate. It is the primary deployment shape (a plain
host launcher with a fresh checkout), and every mono bug found so far was
invisible to the multi-mode suite:

  2026-10-26  ragmcp: every collection denied to the mono operator
  2026-10-26  memorymcp: the legacy ownerless corpus hidden
  2026-10-26  databasemcp: every connect refused ("you are not admin")

Each tool is booted in its OWN subprocess (tests/mono_smoke_driver.py) —
a fresh process is the only faithful mono environment: the auth verifier
is frozen at import time (the factory reads MCP_AUTH_MODE then), and the
suite installs memory_core STUBS into sys.modules globally
(test_memory_artifact_e2e.py), so an in-process test cannot rely on the
real module graph. The subprocess runs the tool's PRODUCTION app on a
scratch port and drives real MCP calls with the system key.

The live stores are never touched: the vector/SQL backends are pointed at
a throwaway turso file, and databasemcp uses an in-memory libsql.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).parent / "mono_smoke_driver.py"

TOOLS = ["databasemcp", "memorymcp", "ragmcp"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mono_env(tmp_path: Path) -> dict:
    """A MONO environment: no user store, one known system key, throwaway
    data planes (the live turso/PG stores must not be touched)."""
    env = dict(os.environ)
    env.update({
        "MCP_AUTH_MODE": "mono",
        "MCP_USERS_STORE_DISABLE": "1",
        "MCP_API_KEY": "mono-smoke-key",
        "DB_PRESET_AUTOCONNECT": "0",
        "TURSO_DATABASE_URL": f"file:{tmp_path / 'smoke.db'}",
        "TURSO_AUTH_TOKEN": "",
        "MCP_STATE_BACKEND": "json",
        "MCP_USERS_BACKEND": "json",
        "PYTHONPATH": os.pathsep.join([
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / "tools" / "databasemcp"),
            str(PROJECT_ROOT / "tools" / "memorymcp"),
            str(PROJECT_ROOT / "tools" / "ragmcp"),
        ]),
    })
    return env


@pytest.mark.parametrize("tool", TOOLS)
def test_mono_deployment_smoke(tool, tmp_path):
    port = _free_port()
    env = _mono_env(tmp_path)
    env["MCP_PORT"] = str(port)
    env["MCP_MGMT_PORT"] = str(_free_port())
    proc = subprocess.run(
        [sys.executable, str(DRIVER), tool, str(port)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout.strip().splitlines()
    payload = json.loads(out[-1]) if out else {"ok": False, "checks": [],
                                               "error": "no driver output"}
    failed = [c for c in payload["checks"] if not c["ok"]]
    assert payload["ok"], (
        f"{tool} mono smoke failed\n"
        f"  error: {payload.get('error')}\n"
        f"  failed checks: {failed}\n"
        f"  stderr tail: {proc.stderr[-1200:]}")
    assert payload["checks"], f"{tool}: driver reported no checks"


def test_driver_detects_the_mono_bugs(tmp_path):
    """The smoke must FAIL when the mono role default regresses.

    Restores the pre-fix default in a COPY of the repo (the live tree is
    never touched) and runs the driver against it: a smoke that cannot
    fail proves nothing.
    """
    copy_root = tmp_path / "repo"
    shutil.copytree(
        PROJECT_ROOT, copy_root,
        ignore=shutil.ignore_patterns(".git", "logs", "my-bundles",
                                      "graphify-out", "**/__pycache__"))
    # reintroduce the mono bug in the COPY: the legacy mono map entry has no
    # role, so the pre-fix default ("user") denies the operator.
    for rel, old, new in (
        ("tools/databasemcp/connections.py",
         "role = resolve_role(entry, user_store)",
         'role = entry.get("role", "user")  # BUG RESTORED'),
        ("tools/ragmcp/ragmcp_fastmcp.py",
         'if resolve_role(entry, user_store) == "admin":',
         'if entry.get("role", "user") == "admin":  # BUG RESTORED'),
    ):
        p = copy_root / rel
        s = p.read_text(encoding="utf-8")
        assert old in s, f"pattern not found in {rel}"
        p.write_text(s.replace(old, new), encoding="utf-8")

    driver = copy_root / "tests" / "mono_smoke_driver.py"
    port = _free_port()
    env = _mono_env(tmp_path)
    env["MCP_PORT"] = str(port)
    env["MCP_MGMT_PORT"] = str(_free_port())
    env["PYTHONPATH"] = os.pathsep.join([
        str(copy_root),
        str(copy_root / "tools" / "databasemcp"),
        str(copy_root / "tools" / "memorymcp"),
        str(copy_root / "tools" / "ragmcp"),
    ])
    proc = subprocess.run(
        [sys.executable, str(driver), "databasemcp", str(port)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    # the regression MUST be detected (non-zero exit / a failed check)
    assert proc.returncode != 0, (
        "the mono smoke did NOT detect the reintroduced role-default bug "
        f"(rc=0): {proc.stdout[-600:]}")
