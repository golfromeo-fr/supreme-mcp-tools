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


# ── work layout (grounds plan §4 — gwork grounds, Phase 2a) ──────────────

CANONICAL_WINDOW = (8000, 8001, 8002, 8003, 8004, 8005, 8200, 5433, 9080)


def canonical_window_free():
    return not (_busy_ports() & set(CANONICAL_WINDOW))


def test_create_work_layout_requires_bundle(sandbox):
    r = run_create(sandbox, "gworkx", "--layout", "work", "--port-base", "20000",
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "requires a config bundle" in r.stderr + r.stdout


def test_create_work_layout_rejects_two_nodes(sandbox, base, bundle):
    r = run_create(sandbox, "gworkx", "--layout", "work", "--nodes", "2",
                   "--port-base", str(base), "--tools", "simplemcp",
                   "-b", str(bundle))
    assert r.returncode != 0 and "single-node" in r.stderr + r.stdout


def test_create_work_layout_rejects_bad_layout(sandbox):
    r = run_create(sandbox, "gworkx", "--layout", "k8s", "--port-base", "20000",
                   "--tools", "simplemcp")
    assert r.returncode != 0 and "custom or work" in r.stderr + r.stdout


@pytest.mark.skipif(not canonical_window_free(),
                    reason="canonical window held — host stack or a gwork ground is up")
def test_create_work_layout_registry_and_env_rewrite(sandbox, bundle):
    r = run_create(sandbox, "gworkp", "--layout", "work",
                   "--tools", ALL_TOOLS, "-b", str(bundle))
    assert r.returncode == 0, r.stderr + r.stdout
    e = registry(sandbox, "gworkp")
    assert e["layout"] == "work"
    assert e["ports"]["central"] == 8200 and e["ports"]["db"] == 5433
    assert e["ports"]["tools"] == {"databasemcp": 8000, "webmcp": 8001,
                                   "simplemcp": 8002, "convertermcp": 8003,
                                   "ragmcp": 8004, "memorymcp": 8005}
    env = (sandbox / "deploy" / "gworkp.env").read_text()
    # D8: host networking — state plane reaches the db via the PUBLISHED
    # loopback port, never via podman DNS
    assert "POSTGRES_HOST=127.0.0.1" in env and "POSTGRES_PORT=5433" in env
    assert "POSTGRES_HOST=db" not in env
    # bundle data-plane lines stay stripped (a comment mentions the drop)
    assert not [l for l in env.splitlines()
                if not l.startswith("#") and "file:" in l]


@pytest.mark.skipif(not canonical_window_free(),
                    reason="canonical window held — host stack or a gwork ground is up")
def test_create_work_layout_turso_env_rewrite(sandbox, bundle):
    r = run_create(sandbox, "gworkt", "--layout", "work", "--topology", "turso",
                   "--tools", ALL_TOOLS, "-b", str(bundle))
    assert r.returncode == 0, r.stderr + r.stdout
    e = registry(sandbox, "gworkt")
    assert e["ports"]["db"] == 9080
    env = (sandbox / "deploy" / "gworkt.env").read_text()
    assert "TURSO_DATABASE_URL=http://127.0.0.1:9080" in env
    assert "http://db:8080" not in env
    assert "POSTGRES_HOST" not in env


def test_create_work_layout_canonical_port_held_fatal(sandbox, bundle):
    """Any held canonical port must refuse the work-layout create (the
    three-way mutual exclusion: host launcher XOR mcp-work XOR gwork)."""
    free = [p for p in CANONICAL_WINDOW if p not in _busy_ports()]
    if not free:
        pytest.skip("every canonical port is held — cannot stage a holder")
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.0", free[0]))
    holder.listen(1)
    try:
        time.sleep(0.05)
        r = run_create(sandbox, "gworkh", "--layout", "work",
                       "--tools", ALL_TOOLS, "-b", str(bundle))
        assert r.returncode != 0 and "mutual exclusion" in r.stderr + r.stdout
    finally:
        holder.close()


# ── Phase 2b: the work topology switch (`startcluster work pg|turso`) ────

import os


@pytest.fixture()
def work_sandbox(sandbox, monkeypatch):
    """Sandbox + the two work compose bases; PATH stubs make the full
    `work <topo>` boot path run hermetically: podman/podman-compose record
    and succeed, ss reports nothing (preflight passes), curl succeeds
    (wait_http gate)."""
    for f in ("compose-work.yml", "compose-work.turso.yml",
              "compose-common.yml", "compose-pg.yml"):
        src = DEPLOY / f
        if src.exists():
            shutil.copy(src, sandbox / "deploy" / f)
    bin_dir = sandbox / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "podman").write_text('#!/bin/bash\necho "$@" >> '
                                    f'{bin_dir}/podman-args.log\nexit 0\n')
    (bin_dir / "podman-compose").write_text(
        f'#!/bin/bash\necho "$@" >> {bin_dir}/compose-args.log\nexit 0\n')
    (bin_dir / "curl").write_text('#!/bin/bash\necho \'{"status":"healthy"}\'\nexit 0\n')
    (bin_dir / "ss").write_text('#!/bin/bash\necho "Netid State Recv-Q Send-Q"\nexit 0\n')
    for f in ("podman", "podman-compose", "curl", "ss"):
        (bin_dir / f).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return sandbox


def run_work(root, *args):
    return subprocess.run(["bash", str(root / "deploy" / "startcluster.sh"),
                           "work", *args],
                          capture_output=True, text=True, cwd=str(root))


def test_work_turso_boot_parks_pg_block_and_stamps(sandbox, work_sandbox, bundle):
    r = run_work(work_sandbox, "turso", "-b", str(bundle))
    assert r.returncode == 0, r.stderr + r.stdout
    env = (work_sandbox / "deploy" / "work.env").read_text()
    assert "[topology] turso" in env
    # the pg block is parked dormant, not deleted (password survives the trip)
    dormant = [l for l in env.splitlines()
               if l.startswith("# [topology:pg-dormant] POSTGRES_PASSWORD=")]
    assert dormant, "no dormant pg password block"
    # the turso state plane is active on the PUBLISHED loopback port
    assert "TURSO_DATABASE_URL=http://127.0.0.1:9080" in env
    assert "http://db:8080" not in env
    # the rendered compose is the turso base: sqld on the load-bearing publish
    run_yml = (work_sandbox / "deploy" / "compose-work.run.yml").read_text()
    assert "libsql-server" in run_yml and "127.0.0.1:9080:8080" in run_yml
    # (a pre-existing work.env would have been backed up before regeneration:
    #  work.env.bak-* — a first-ever boot has no prior env, so no backup)


def test_work_pg_round_trip_revives_same_password(sandbox, work_sandbox, bundle):
    """Edge 2 end-to-end: turso → pg revives the SAME POSTGRES_PASSWORD and
    an active pg block (no fresh-password bricking of the pgdata volume)."""
    r1 = run_work(work_sandbox, "turso", "-b", str(bundle))
    assert r1.returncode == 0, r1.stderr + r1.stdout
    env_turso = (work_sandbox / "deploy" / "work.env").read_text()
    dormant_pw = [l for l in env_turso.splitlines()
                  if l.startswith("# [topology:pg-dormant] POSTGRES_PASSWORD=")][0]
    r2 = run_work(work_sandbox, "pg", "-b", str(bundle))
    assert r2.returncode == 0, r2.stderr + r1.stdout
    env_pg = (work_sandbox / "deploy" / "work.env").read_text()
    assert "[topology] pg" in env_pg
    active_pw = [l for l in env_pg.splitlines()
                 if l.startswith("POSTGRES_PASSWORD=")]
    assert active_pw, "no active pg block after the return trip"
    assert active_pw[0].split("=", 1)[1] == dormant_pw.split("=", 1)[1], \
        "pg password changed across the turso round trip — pgdata would be bricked"
    run_yml = (work_sandbox / "deploy" / "compose-work.run.yml").read_text()
    assert "pgvector" in run_yml and "127.0.0.1:5433:5432" in run_yml


def test_work_turso_node_parity():
    """The turso work compose must differ from the pg one ONLY in the db
    service (+ its volume) — the node/ui services are the daily driver's
    surface and must never drift between the two files."""
    import yaml

    def services(path):
        return yaml.safe_load(Path(path).read_text())["services"]

    pg = services(DEPLOY / "compose-work.yml")
    tu = services(DEPLOY / "compose-work.turso.yml")
    assert set(pg) == {"db", "work", "ui"} and set(tu) == {"db", "work", "ui"}
    assert pg["work"] == tu["work"], "work service drifted between pg/turso bases"
    assert pg["ui"] == tu["ui"], "ui service drifted between pg/turso bases"
    assert tu["db"]["image"] == "ghcr.io/tursodatabase/libsql-server:latest"
    assert pg["db"]["image"] == "docker.io/pgvector/pgvector:pg16"
    assert "127.0.0.1:9080:8080" in tu["db"]["ports"]
    assert "127.0.0.1:5433:5432" in pg["db"]["ports"]


def test_create_two_nodes_records_tools2(sandbox, base):
    e = None
    r = run_create(sandbox, "twin", "--nodes", "2", "--port-base", str(base),
                   "--tools", "simplemcp memorymcp")
    assert r.returncode == 0, r.stderr + r.stdout
    e = registry(sandbox, "twin")
    assert e["ports"]["tools2"] == {"simplemcp": base + 1004,
                                    "memorymcp": base + 1007}
    every = [e["ports"]["central"], e["ports"]["central2"],
             *e["ports"]["tools"].values(), *e["ports"]["tools2"].values()]
    assert len(every) == len(set(every))
