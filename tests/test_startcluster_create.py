"""Unit pin for `startcluster create` (custom-cluster generator) — Phase 1.2
of plans/testing-grounds-and-work-topology-2026-10-05.md.

The real script is COPIED into a sandbox repo, so REPO_ROOT (derived from
$0) resolves inside the sandbox: the registry (deploy/clusters.d/), env
files and generated composes never touch the real repo. `create` must not
boot anything — no podman needed; port checks use `ss`.

Pins the +2 tool-port offset fix (Phase-1 discovery: without it, the
databasemcp container port 8000 mapped to host port `base` = the central
API port — a guaranteed collision that never fired while clusters.d was
empty), the registry-conflict grep fix (': 20000' pretty-JSON), and the
create -b pass-through fix.

All port-base values come from free_base(): a live ground legitimately
occupies part of 20000-20999, so hardcoding a base breaks whenever a ground
is up — exactly the coexistence the grounds design promises.
"""

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy"

NODE_ENV = "MCP_UI_SECRET=sandbox-ui-secret\n"


@pytest.fixture()
def sandbox(tmp_path):
    root = tmp_path / "repo"
    (root / "deploy").mkdir(parents=True)
    for f in ("startcluster.sh", "gen_custom_compose.py",
              "bundle_node_env.py", "env_strip.py"):
        src = DEPLOY / f
        if src.exists():
            shutil.copy(src, root / "deploy" / f)
    (root / "deploy" / "node.env").write_text(NODE_ENV)
    (root / "deploy" / "clusters.d").mkdir()
    return root


def run_create(root, *args):
    return subprocess.run(["bash", str(root / "deploy" / "startcluster.sh"),
                           "create", *args],
                          capture_output=True, text=True, cwd=str(root))


def registry(root, name):
    return json.loads((root / "deploy" / "clusters.d" / f"{name}.json").read_text())


def _busy_ports():
    """Listening TCP ports from `ss -ltn` (the same source create checks)."""
    out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    busy = set()
    for line in out.splitlines():
        if line.strip().startswith("LISTEN"):
            try:
                busy.add(int(line.split()[3].rsplit(":", 1)[-1]))
            except (ValueError, IndexError):
                pass
    return busy


def _registry_ports(root):
    ports = set()
    d = root / "deploy" / "clusters.d"
    if d.exists():
        for f in d.glob("*.json"):
            e = json.loads(f.read_text())
            p = e.get("ports", {})
            if p.get("central"):
                ports.add(p["central"])
            if p.get("central2"):
                ports.add(p["central2"])
            ports.update((p.get("tools") or {}).values())
    return ports


def free_base(root):
    """First create port-base whose whole reserved window (base, base+1000,
    base+2..base+7) is free of listeners AND registry claims."""
    busy = _busy_ports() | _registry_ports(root)
    for base in range(20000, 20901, 100):
        window = {base, base + 1000} | set(range(base + 2, base + 8))
        if not (window & busy):
            return base
    pytest.skip("no free port base in 20000-20999")


@pytest.fixture()
def base(sandbox):
    return free_base(sandbox)


ALL_TOOLS = "databasemcp webmcp simplemcp convertermcp ragmcp memorymcp"


# ── happy paths ──────────────────────────────────────────────────────────

def test_create_writes_registry_and_env_no_boot(sandbox, base):
    r = run_create(sandbox, "s1x", "--port-base", str(base),
                   "--tools", "simplemcp memorymcp databasemcp")
    assert r.returncode == 0, r.stderr
    e = registry(sandbox, "s1x")
    # +2 offset pin: central owns base; tools occupy base+2..base+7
    assert e["ports"]["central"] == base
    assert e["ports"]["tools"] == {"simplemcp": base + 4, "memorymcp": base + 7,
                                   "databasemcp": base + 2}
    all_ports = [e["ports"]["central"]] + list(e["ports"]["tools"].values())
    assert len(all_ports) == len(set(all_ports)), "host port collision in registry"
    env = (sandbox / "deploy" / "s1x.env").read_text()
    assert "MCP_USERS_BACKEND=db" in env and "MCP_STATE_BACKEND=db" in env
    # contract: create does NOT boot
    assert "boot it with: startcluster s1x up" in r.stdout


def test_create_all_six_tools_no_collision(sandbox, base):
    r = run_create(sandbox, "six", "--port-base", str(base), "--tools", ALL_TOOLS)
    assert r.returncode == 0, r.stderr
    e = registry(sandbox, "six")
    assert e["ports"]["tools"] == {"databasemcp": base + 2, "webmcp": base + 3,
                                   "simplemcp": base + 4, "convertermcp": base + 5,
                                   "ragmcp": base + 6, "memorymcp": base + 7}
    every = [e["ports"]["central"], *e["ports"]["tools"].values()]
    assert len(every) == len(set(every))


def test_create_two_nodes_central2(sandbox, base):
    r = run_create(sandbox, "duo", "--nodes", "2", "--port-base", str(base),
                   "--tools", "simplemcp")
    assert r.returncode == 0, r.stderr
    e = registry(sandbox, "duo")
    assert e["nodes"] == 2 and e["ports"]["central2"] == base + 1000


def test_create_records_bundle_path(sandbox, base, tmp_path):
    b = tmp_path / "b"
    (b / "env").mkdir(parents=True)
    (b / "env" / ".env").write_text("MCP_UI_SECRET=x\n")
    r = run_create(sandbox, "bsrc", "--port-base", str(base),
                   "--tools", "simplemcp", "-b", str(b))
    assert r.returncode == 0, r.stderr + r.stdout
    e = registry(sandbox, "bsrc")
    assert e["bundle"] == str(b)


# ── validation (FATAL cases; all fail BEFORE any port work) ─────────────

@pytest.mark.parametrize("bad", ["Bad", "x", "ab", "-abc", "a" * 17,
                                 "work", "test", "a_b"])
def test_invalid_names_fatal(sandbox, bad):
    r = run_create(sandbox, bad, "--port-base", "20000", "--tools", "simplemcp")
    assert r.returncode != 0 and "FATAL" in r.stderr + r.stdout


def test_duplicate_name_fatal(sandbox, base):
    assert run_create(sandbox, "dup", "--port-base", str(base),
                      "--tools", "simplemcp").returncode == 0
    r = run_create(sandbox, "dup", "--port-base", str(base),
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "already exists" in r.stderr + r.stdout


def test_unknown_tool_fatal(sandbox):
    r = run_create(sandbox, "tls1", "--port-base", "20000", "--tools", "nope")
    assert r.returncode != 0 and "unknown tool" in r.stderr + r.stdout


def test_bad_nodes_fatal(sandbox):
    r = run_create(sandbox, "nn3", "--nodes", "3", "--port-base", "20000",
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "FATAL" in r.stderr + r.stdout


def test_external_topology_fatal_for_create(sandbox):
    r = run_create(sandbox, "ext", "--topology", "external-pg",
                   "--port-base", "20000", "--tools", "simplemcp")
    assert r.returncode != 0 and "pg|turso" in r.stderr + r.stdout


def test_missing_bundle_env_fatal(sandbox, tmp_path):
    b = tmp_path / "empty-bundle"
    b.mkdir()
    r = run_create(sandbox, "mbb", "--port-base", "20000",
                   "--tools", "simplemcp", "-b", str(b))
    assert r.returncode != 0 and "FATAL" in r.stderr + r.stdout


# ── port allocation ──────────────────────────────────────────────────────

@pytest.fixture()
def listener():
    """A real LISTEN socket — `ss -ltn` sees it, so create's busy check fires."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.0", 0))
    s.listen(1)
    time.sleep(0.05)  # let ss observe the LISTEN
    yield s.getsockname()[1]
    s.close()


@pytest.mark.skipif(shutil.which("ss") is None, reason="ss not available")
def test_explicit_port_base_busy_fatal(sandbox, listener):
    r = run_create(sandbox, "busy", "--port-base", str(listener),
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "already in use" in r.stderr + r.stdout


@pytest.mark.skipif(shutil.which("ss") is None, reason="ss not available")
def test_registry_port_conflict_fatal(sandbox, base):
    assert run_create(sandbox, "first", "--port-base", str(base),
                      "--tools", "simplemcp").returncode == 0
    r = run_create(sandbox, "second", "--port-base", str(base),
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "already mapped" in r.stderr + r.stdout


@pytest.mark.skipif(shutil.which("ss") is None, reason="ss not available")
def test_auto_port_base_skips_busy_bases(sandbox, base):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.0", base))
    s.listen(1)
    try:
        time.sleep(0.05)
        r = run_create(sandbox, "auto", "--tools", "simplemcp")
        assert r.returncode == 0, r.stderr + r.stdout
        e = registry(sandbox, "auto")
        assert e["port_base"] != base
        assert e["port_base"] % 100 == 0 and 20000 <= e["port_base"] <= 20900
    finally:
        s.close()


@pytest.mark.skipif(shutil.which("ss") is None, reason="ss not available")
def test_auto_port_base_skips_registry_claimed_base(sandbox, base):
    """Regression (Phase-1 discovery): the registry-conflict grep matched
    ':20000' against pretty-printed JSON (': 20000') and NEVER fired, so
    auto-pick could re-claim a base another cluster already registered."""
    assert run_create(sandbox, "first", "--port-base", str(base),
                      "--tools", "simplemcp").returncode == 0
    r = run_create(sandbox, "second", "--tools", "simplemcp")  # auto
    assert r.returncode == 0, r.stderr + r.stdout
    assert registry(sandbox, "second")["port_base"] != base


# ── C3: bundle identity/config override chain on `up` ────────────────────

BUNDLE_ENV = "MCP_UI_SECRET=ground-secret\nTURSO_DATABASE_URL=file:/x/y.db\n"


@pytest.fixture()
def bundle(sandbox):
    b = sandbox / "bundle"
    (b / "env").mkdir(parents=True)
    (b / "env" / ".env").write_text(BUNDLE_ENV)
    (b / "identity").mkdir()
    (b / "identity" / "users.json").write_text('{"version": 1, "users": {}}')
    (b / "config").mkdir()
    (b / "config" / "ports.json").write_text("{}")
    (b / "tools" / "simplemcp").mkdir(parents=True)
    (b / "tools" / "simplemcp" / "config.json").write_text("{}")
    return b


@pytest.fixture()
def fake_compose(sandbox, monkeypatch):
    """A podman-compose stub that records argv and exits 0 — lets the `up`
    path run end-to-end without podman (wait_http WARNING is tolerated)."""
    import os
    bin_dir = sandbox / "bin"
    bin_dir.mkdir()
    log = bin_dir / "compose-args.log"
    stub = bin_dir / "podman-compose"
    stub.write_text(f'#!/bin/bash\necho "$@" >> {log}\nexit 0\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    # the stub starts nothing: collapse wait_http's 3s backoff (40 tries)
    monkeypatch.setenv("SC_WAIT_SLEEP", "0.05")
    return log


def compose_calls(fake_compose):
    return [l.split() for l in fake_compose.read_text().strip().splitlines()]


def up(root, name):
    return subprocess.run(["bash", str(root / "deploy" / "startcluster.sh"),
                           name, "up"],
                          capture_output=True, text=True, cwd=str(root))


def test_bundle_up_chains_identity_override(sandbox, base, bundle, fake_compose):
    r = run_create(sandbox, "bsnd", "--port-base", str(base),
                   "--tools", "simplemcp", "-b", str(bundle))
    assert r.returncode == 0, r.stderr + r.stdout
    up1 = up(sandbox, "bsnd")
    assert up1.returncode == 0, up1.stderr + up1.stdout
    calls = compose_calls(fake_compose)
    assert any("compose-bsnd.bundle.yml" in c for c in calls), calls
    override = (sandbox / "deploy" / "compose-bsnd.bundle.yml").read_text()
    assert "node1:" in override and "users.json" in override
    assert "/root/.config/supreme-mcp-tools/users.json" in override
    # the ephemeral data plane: bundle grounds get a container-local turso
    # file (grounds design §3) so memorymcp/ragmcp never run as NullVectorStore
    assert "TURSO_DATABASE_URL=file:/app/ground-memory.db" in override
    # a 1-node cluster's override must NOT define a node2 service
    assert "node2:" not in override


def test_bundle_bare_reup_keeps_the_chain(sandbox, base, bundle, fake_compose):
    """E10 lesson (138b471): a bare re-up must re-apply the bundle override."""
    assert run_create(sandbox, "brep", "--port-base", str(base),
                      "--tools", "simplemcp", "-b", str(bundle)).returncode == 0
    assert up(sandbox, "brep").returncode == 0
    assert up(sandbox, "brep").returncode == 0
    calls = compose_calls(fake_compose)
    assert len(calls) == 2, calls
    assert all("compose-brep.bundle.yml" in c for c in calls), calls


def test_bundle_two_nodes_overrides_both(sandbox, base, bundle, fake_compose):
    assert run_create(sandbox, "bduo", "--nodes", "2", "--port-base", str(base),
                      "--tools", "simplemcp", "-b", str(bundle)).returncode == 0
    assert up(sandbox, "bduo").returncode == 0
    override = (sandbox / "deploy" / "compose-bduo.bundle.yml").read_text()
    assert "node1:" in override and "node2:" in override


def test_no_bundle_up_has_no_override(sandbox, base, fake_compose):
    assert run_create(sandbox, "plain", "--port-base", str(base),
                      "--tools", "simplemcp").returncode == 0
    assert up(sandbox, "plain").returncode == 0
    calls = compose_calls(fake_compose)
    assert all("bundle.yml" not in c for c in calls), calls
