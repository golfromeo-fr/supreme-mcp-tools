"""Config bundles: harvest script + startcluster bundle generators.

Covers plans/pod-config-bundle-2026-09-17.md — P1 (harvest-config.sh) and
P2 (bundle-node-env / bundle-compose through deploy/startcluster.sh hidden
subcommands, no podman needed). All filesystem work happens in a sandbox
repo copy; the real repo and its .env are never touched.
"""

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STARTCLUSTER = REPO / "deploy" / "startcluster.sh"
HARVEST = REPO / "deploy" / "harvest-config.sh"

SANDBOX_ENV = (
    "MCP_AUTH_MODE=multi\n"
    "MCP_UI_SECRET=ui-secret-value\n"
    "MCP_UI_USERNAME=admin\n"
    "BRAVE_SEARCH_API_KEY=brave-key-value\n"
    "TURSO_DATABASE_URL=file:/home/gr/turso_data/memorymcp.db\n"
    "POSTGRES_TEST_DSN=postgresql://gr:pw@192.168.1.5:5432/test\n"
    "DB_PRESET_02=postgresql://gr:pw@192.168.1.5:5432/test\n"
    "DB_PRESET_02_DESC=Local test Postgres\n"
    "DB_PRESET_03=file:/home/gr/turso_data/memorymcp.db\n"
    "DB_PRESET_03_DESC=Turso memory DB\n"
)
USERS_JSON = {
    "version": 1,
    "users": {
        "admin": {"mcp_key": "admin-key", "password_hash": "pbkdf2$abc", "role": "admin"},
        "tester": {"mcp_key": "tester-key", "password_hash": "pbkdf2$def", "role": "user"},
    },
}
TOOLCFG = {"disabled_tools": ["brave_search_web"], "tools": {"webmcp": ["fetch_url"]}, "version": 1}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A fake repo (with the real harvest script) + fake HOME with the config store."""
    root = tmp_path / "repo"
    (root / "deploy").mkdir(parents=True)
    shutil.copy(HARVEST, root / "deploy" / "harvest-config.sh")
    (root / ".env").write_text(SANDBOX_ENV)

    config = root / "config"
    config.mkdir()
    (config / "ports.json").write_text('{"ranges": {}}')
    (config / "ports.example.json").write_text('{"EXAMPLE": true}')
    (config / "launcher_config.json").write_text('{"logging": {}}')

    for tool in ("memorymcp", "webmcp"):
        d = root / "tools" / tool
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"auth": {"api_key": f"{tool}-key"}}))

    home = tmp_path / "home"
    store = home / ".config" / "supreme-mcp-tools"
    store.mkdir(parents=True)
    (store / "users.json").write_text(json.dumps(USERS_JSON))
    (store / "tools_config.json").write_text(json.dumps(TOOLCFG))
    monkeypatch.setenv("HOME", str(home))
    return root


def harvest(sandbox, *args):
    return run(["bash", sandbox / "deploy" / "harvest-config.sh", *args])


# ---------------------------------------------------------------- harvest


def test_list_writes_nothing(sandbox, tmp_path):
    target = tmp_path / "list-bundle"
    r = harvest(sandbox, "--list", str(target))
    assert r.returncode == 0, r.stderr
    assert not target.exists()
    assert "[ok" in r.stdout and "identity/users.json" in r.stdout


def test_harvest_happy_tree_and_manifest(sandbox, tmp_path):
    bundle = tmp_path / "b"
    r = harvest(sandbox, str(bundle))
    assert r.returncode == 0, r.stderr
    assert (bundle / "env" / ".env").read_text() == SANDBOX_ENV
    assert json.loads((bundle / "identity" / "users.json").read_text()) == USERS_JSON
    assert json.loads((bundle / "identity" / "tools_config.json").read_text()) == TOOLCFG
    assert (bundle / "config" / "ports.json").exists()
    assert not (bundle / "config" / "ports.example.json").exists()
    assert json.loads((bundle / "tools" / "memorymcp" / "config.json").read_text())["auth"]["api_key"] == "memorymcp-key"
    assert (bundle / "README.md").exists()

    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["bundle_version"] == 1
    assert manifest["host_data_paths"]["turso_dir"] == "/home/gr/turso_data"
    by_path = {f["path"]: f for f in manifest["files"]}
    for rel, content in [("env/.env", SANDBOX_ENV), ("config/ports.json", '{"ranges": {}}')]:
        assert by_path[rel]["sha256"] == hashlib.sha256(content.encode()).hexdigest()
        assert by_path[rel]["source"].endswith(rel.split("/")[-1])
    # no dotfiles recorded
    assert not [f for f in manifest["files"] if f["path"].startswith(".")]


def test_harvest_missing_env_exit_2(sandbox, tmp_path):
    (sandbox / ".env").unlink()
    bundle = tmp_path / "b"
    r = harvest(sandbox, str(bundle))
    assert r.returncode == 2
    assert not bundle.exists()


def test_harvest_missing_optional_records_warning(sandbox, tmp_path):
    (Path(sandbox).home() / ".config/supreme-mcp-tools/users.json").unlink()
    bundle = tmp_path / "b"
    r = harvest(sandbox, str(bundle))
    assert r.returncode == 0, r.stderr
    assert not (bundle / "identity" / "users.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert any("users.json" in w for w in manifest["warnings"])


def test_redact(sandbox, tmp_path):
    bundle = tmp_path / "b"
    r = harvest(sandbox, "--redact", str(bundle))
    assert r.returncode == 0, r.stderr
    env = (bundle / "env" / ".env").read_text()
    assert "MCP_UI_SECRET=__REDACTED__" in env
    assert "BRAVE_SEARCH_API_KEY=__REDACTED__" in env
    assert "DB_PRESET_02=__REDACTED__" in env            # DSN-carrying preset
    assert "DB_PRESET_02_DESC=Local test Postgres" in env  # labels survive
    assert "MCP_UI_USERNAME=admin" in env                # non-secret keys survive
    assert "TURSO_DATABASE_URL=file:/home/gr/turso_data/memorymcp.db" in env
    users = json.loads((bundle / "identity" / "users.json").read_text())
    assert users["users"]["admin"]["mcp_key"] == "__REDACTED__"
    assert users["users"]["tester"]["password_hash"] == "__REDACTED__"
    tool = json.loads((bundle / "tools" / "webmcp" / "config.json").read_text())
    assert tool["auth"]["api_key"] == "__REDACTED__"
    assert json.loads((bundle / "manifest.json").read_text())["redacted"] is True


def test_history_and_junk_never_harvested(sandbox, tmp_path):
    (sandbox / ".env~").write_text("OLD=1\n")
    home_store = Path(sandbox).home() / ".config/supreme-mcp-tools"
    (home_store / "users.json.lock").write_text("")
    (home_store / "webmcp.json").write_text('{"mutations": []}')
    bundle = tmp_path / "b"
    assert harvest(sandbox, str(bundle)).returncode == 0
    names = {p.name for p in bundle.rglob("*") if p.is_file()}
    assert ".env~" not in names and "users.json.lock" not in names and "webmcp.json" not in names


def test_rerun_overwrites_and_refreshes_manifest(sandbox, tmp_path):
    bundle = tmp_path / "b"
    assert harvest(sandbox, str(bundle)).returncode == 0
    (sandbox / ".env").write_text(SANDBOX_ENV + "EXTRA_KEY=1\n")
    assert harvest(sandbox, str(bundle)).returncode == 0
    env = (bundle / "env" / ".env").read_text()
    assert "EXTRA_KEY=1" in env
    manifest = json.loads((bundle / "manifest.json").read_text())
    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["env/.env"]["sha256"] == hashlib.sha256(env.encode()).hexdigest()


def test_zip_round_trip(sandbox, tmp_path):
    bundle = tmp_path / "b"
    assert harvest(sandbox, "--zip", str(bundle)).returncode == 0
    z = zipfile.ZipFile(str(bundle) + ".zip")
    assert "b/env/.env" in z.namelist() or "env/.env" in z.namelist()


def test_bundle_location_resolution(sandbox, tmp_path, monkeypatch):
    import os
    script = str(sandbox / "deploy" / "harvest-config.sh")
    inv = tmp_path / "inv"
    inv.mkdir()
    monkeypatch.chdir(inv)

    # default: <cwd>/my-bundles/bundle-<UTC ts>
    assert run(["bash", script]).returncode == 0
    made = list((inv / "my-bundles").iterdir())
    assert len(made) == 1 and made[0].name.startswith("bundle-")

    # bare name: <cwd>/my-bundles/<name>
    assert run(["bash", script, "toto"]).returncode == 0
    assert (inv / "my-bundles" / "toto" / "manifest.json").exists()

    # startcluster hands us its ORIG_PWD — bare names resolve there, not repo-root
    other = tmp_path / "other"
    env = dict(os.environ, HARVEST_INVOCATION_PWD=str(other))
    r = subprocess.run(["bash", script, "from-startcluster"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (other / "my-bundles" / "from-startcluster" / "manifest.json").exists()

    # explicit paths (absolute or containing /) pass through unchanged
    assert run(["bash", script, str(tmp_path / "explicit")]).returncode == 0
    assert (tmp_path / "explicit" / "manifest.json").exists()


# ------------------------------------------------- bundle-node-env (P2)


def startcluster(*args, env_extra=None):
    import os
    env = dict(os.environ)
    env.pop("BUNDLE_MINIO_PW", None)
    if env_extra:
        env.update(env_extra)
    return run([STARTCLUSTER, *args], env=env)


@pytest.fixture()
def harvested(sandbox, tmp_path):
    bundle = tmp_path / "b"
    r = harvest(sandbox, str(bundle))
    assert r.returncode == 0, r.stderr
    return bundle


def env_lines(text):
    return [l for l in text.splitlines() if l and not l.startswith("#")]


def test_pg_identity_kept_state_rewritten(harvested):
    r = startcluster("bundle-node-env", harvested, "pg", env_extra={"BUNDLE_MINIO_PW": "minio-secret"})
    assert r.returncode == 0, r.stderr
    keys = dict(l.split("=", 1) for l in env_lines(r.stdout) if "=" in l)
    assert keys["MCP_UI_SECRET"] == "ui-secret-value"          # identity plane verbatim
    assert keys["BRAVE_SEARCH_API_KEY"] == "brave-key-value"   # tool keys verbatim
    assert keys["POSTGRES_HOST"] == "db"                       # state plane rewritten
    assert keys["POSTGRES_PASSWORD"].startswith("cluster-")
    assert "TURSO_DATABASE_URL" not in keys                    # file: URL stripped
    assert "POSTGRES_TEST_DSN" not in keys                     # host-only DSN stripped
    assert keys["DB_PRESET_02"].startswith("postgresql://")    # network preset kept
    assert "DB_PRESET_03" not in keys                          # file: preset stripped
    assert "DB_PRESET_03_DESC" not in keys                     # …with its descriptor
    assert keys["S3_SECRET_KEY"] == "minio-secret"             # MinIO block added
    assert keys["MCP_USERS_BACKEND"] == "db" and keys["MCP_STATE_BACKEND"] == "db"


def test_turso_state_plane(harvested):
    r = startcluster("bundle-node-env", harvested, "turso", env_extra={"BUNDLE_MINIO_PW": ""})
    keys = dict(l.split("=", 1) for l in env_lines(r.stdout) if "=" in l)
    assert keys["TURSO_DATABASE_URL"] == "http://db:8080"
    assert "POSTGRES_HOST" not in keys
    assert "S3_SECRET_KEY" not in keys                         # no MinIO -> stripped


def test_work_mode_reuses_existing_password_and_keeps_file_planes(harvested, tmp_path):
    existing = tmp_path / "old-work.env"
    existing.write_text("POSTGRES_PASSWORD=keepme-pw\n")
    r = startcluster("bundle-node-env", harvested, "work", existing)
    keys = dict(l.split("=", 1) for l in env_lines(r.stdout) if "=" in l)
    assert keys["POSTGRES_HOST"] == "127.0.0.1"
    assert keys["POSTGRES_PORT"] == "5433"
    assert keys["POSTGRES_PASSWORD"] == "keepme-pw"
    assert keys["TURSO_DATABASE_URL"] == "file:/home/gr/turso_data/memorymcp.db"  # work = host-equivalent
    assert keys["DB_PRESET_03"] == "file:/home/gr/turso_data/memorymcp.db"
    assert "S3_ENDPOINT" not in keys                           # no MinIO in the work env


def test_keep_dataplanes_escape_hatch(harvested):
    r = startcluster("--keep-dataplanes", "bundle-node-env", harvested, "pg", env_extra={"BUNDLE_MINIO_PW": ""})
    keys = dict(l.split("=", 1) for l in env_lines(r.stdout) if "=" in l)
    assert keys["DB_PRESET_03"] == "file:/home/gr/turso_data/memorymcp.db"


def test_flag_needs_a_value():
    r = startcluster("--config-bundle")  # flag with no value left behind it
    assert r.returncode == 1
    assert "--config-bundle needs" in (r.stdout + r.stderr)


def test_missing_bundle_fails_loudly(tmp_path):
    r = startcluster("bundle-node-env", tmp_path / "nope", "pg")
    assert r.returncode == 1
    assert "not found" in (r.stdout + r.stderr)


def test_short_flag_b(harvested):
    r = startcluster("-b", harvested, "bundle-node-env", "pg", env_extra={"BUNDLE_MINIO_PW": ""})
    assert r.returncode == 0, r.stderr
    assert "POSTGRES_HOST=db" in r.stdout


def test_long_flag_feeds_bundle_node_env(harvested):
    r = startcluster("--config-bundle", harvested, "bundle-node-env", "pg",
                     env_extra={"BUNDLE_MINIO_PW": ""})
    assert r.returncode == 0, r.stderr
    assert "POSTGRES_HOST=db" in r.stdout


def test_bundle_node_env_without_any_bundle_fails():
    r = startcluster("bundle-node-env", "pg")
    assert r.returncode == 1
    assert "no bundle given" in (r.stdout + r.stderr)


def test_zip_bundle_resolves(tmp_path, harvested):
    zpath = tmp_path / "b.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in sorted(harvested.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(harvested.parent))
    r = startcluster("bundle-node-env", zpath, "pg", env_extra={"BUNDLE_MINIO_PW": ""})
    assert r.returncode == 0, r.stderr
    assert "POSTGRES_HOST=db" in r.stdout


# ------------------------------------------------- bundle-compose (P3/P4 writer)


def test_bundle_compose_writer(tmp_path):
    out = tmp_path / "override.yml"
    r = startcluster("bundle-compose", out,
                     "work|/b/config/ports.json|/app/config/ports.json|ro",
                     "node1|/b/identity/users.json|/root/.config/supreme-mcp-tools/users.json|ro")
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "  work:" in text and "  node1:" in text
    assert "      - /b/config/ports.json:/app/config/ports.json:ro" in text


def test_bundle_compose_writer_empty(tmp_path):
    out = tmp_path / "override.yml"
    assert startcluster("bundle-compose", out).returncode == 0
    assert "services: {}" in out.read_text()


# ------------------------------------------------- the merge assumption itself


@pytest.mark.skipif(shutil.which("podman-compose") is None,
                    reason="podman-compose not installed")
def test_podman_compose_volumes_merge_is_union(tmp_path):
    """Pin the probe fact: override-file volumes UNION into the base service
    (podman-compose rec_merge_one) — else the -f bundle override would drop
    the base /app/.env mount."""
    probe = (
        "import podman_compose\n"
        "base = {'services': {'node1': {'volumes': ['./node.env:/app/.env:ro']}}}\n"
        "over = {'services': {'node1': {'volumes': ["
        "'/b/users.json:/root/.config/supreme-mcp-tools/users.json:ro']}}}\n"
        "podman_compose.rec_merge_one(base, over)\n"
        "v = base['services']['node1']['volumes']\n"
        "assert './node.env:/app/.env:ro' in v\n"
        "assert '/b/users.json:/root/.config/supreme-mcp-tools/users.json:ro' in v\n"
        "print('ok')\n"
    )
    pc = shutil.which("podman-compose")
    py = Path(pc).read_text().splitlines()[0]
    py = py.replace("#!", "").strip() or "/usr/bin/python3"
    r = subprocess.run([py, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
