"""setup_opencode.sh tests — opencode v2 config shape + merge correctness.

The script previously (a) wrote the deprecated flat mcp.<name> shape with an
`enabled` key that opencode v2 rejects, and (b) had a FATAL merge bug: it piped
gen_config into `python3 - <<'PY'` whose stdin is already the script source, so
sys.stdin.read() was empty and no config was ever written. These tests run the
REAL script against a tmp tree (copied script + common.sh, so SMC_ROOT resolves
there) and assert the v2 output, key handling, and merge/migration behavior.

The minimal schema below mirrors the live https://opencode.ai/config.json
Mcp.RemoteConfigEncoded definition (verified 2026-11-18): servers nested under
mcp.servers, remote entry additionalProperties:false (so `enabled` is invalid),
codemode/disabled/oauth present.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Minimal offline mirror of the opencode v2 mcp.servers remote-server schema.
V2_MCP_SCHEMA = {
    "type": "object",
    "properties": {
        "mcp": {
            "type": "object",
            "properties": {
                "servers": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["type", "url"],
                        "properties": {
                            "type": {"enum": ["remote", "local"]},
                            "url": {"type": "string"},
                            "headers": {"type": "object"},
                            "oauth": {},
                            "disabled": {"type": "boolean"},
                            "codemode": {"type": "boolean"},
                            "timeout": {"type": "object"},
                            "protocol": {},
                            "pool": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
    },
}


def _run(tmp_path, tools="probe", extra=(), preconfig=None):
    """Copy the real script into a tmp tree and run it (--no-restart always)."""
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "ports.json").write_text(json.dumps(
        {"assignments": {"mcp": {"probe": 8002, "other": 8009}}}))
    (tmp_path / "tools" / "probe").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "probe" / "config.json").write_text(json.dumps(
        {"auth": {"api_key": "probe-key-123"}}))
    (tmp_path / "scripts").mkdir(exist_ok=True)
    for f in ("setup_opencode.sh", "common.sh"):
        shutil.copy(PROJECT_ROOT / "scripts" / f, tmp_path / "scripts" / f)

    cfg = tmp_path / "opencode.jsonc"
    if preconfig is not None:
        cfg.write_text(preconfig if isinstance(preconfig, str) else json.dumps(preconfig))
    cmd = ["bash", str(tmp_path / "scripts" / "setup_opencode.sh"),
           "--config", str(cfg), "--no-restart", "--tools", tools, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc, cfg


def test_v2_shape_and_keys(tmp_path):
    proc, cfg = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(cfg.read_text())
    # nested v2 shape, NOT flat mcp.<name>
    assert "servers" in data["mcp"]
    assert "probe" in data["mcp"]["servers"]
    assert "probe" not in data["mcp"]  # no flat deprecated entry
    entry = data["mcp"]["servers"]["probe"]
    assert entry["type"] == "remote"
    assert entry["url"] == "http://127.0.0.1:8002/mcp"
    assert entry["codemode"] is False
    assert entry["oauth"] is False
    assert "enabled" not in entry  # v2 rejects it
    assert entry["headers"]["X-API-Key"] == "probe-key-123"


def test_output_validates_against_v2_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    proc, cfg = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    jsonschema.validate(json.loads(cfg.read_text()), V2_MCP_SCHEMA)


def test_no_keys_omits_headers(tmp_path):
    proc, cfg = _run(tmp_path, extra=["--no-keys"])
    assert proc.returncode == 0, proc.stderr
    entry = json.loads(cfg.read_text())["mcp"]["servers"]["probe"]
    assert "headers" not in entry


def test_dry_run_writes_nothing(tmp_path):
    proc, cfg = _run(tmp_path, extra=["--dry-run"])
    assert proc.returncode == 0, proc.stderr
    assert not cfg.exists()
    assert '"servers"' in proc.stdout  # prints the v2 shape


def test_merge_preserves_non_mcp_and_existing_servers(tmp_path):
    pre = {
        "model": "some/model",
        "mcp": {
            "timeout": {"startup": 5000},
            "servers": {"keepme": {"type": "remote", "url": "http://x/mcp"}},
        },
    }
    proc, cfg = _run(tmp_path, preconfig=pre)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(cfg.read_text())
    assert data["model"] == "some/model"            # non-mcp preserved
    assert data["mcp"]["timeout"] == {"startup": 5000}  # mcp-level setting preserved
    assert "keepme" in data["mcp"]["servers"]       # existing server preserved
    assert "probe" in data["mcp"]["servers"]        # our server added


def test_flat_v1_entries_migrated_to_servers(tmp_path):
    pre = {"mcp": {"old": {"type": "remote", "url": "http://old/mcp", "enabled": True},
                   "off": {"type": "remote", "url": "http://off/mcp", "enabled": False}}}
    proc, cfg = _run(tmp_path, preconfig=pre)
    assert proc.returncode == 0, proc.stderr
    servers = json.loads(cfg.read_text())["mcp"]["servers"]
    assert "old" in servers and "enabled" not in servers["old"]
    assert "off" in servers and servers["off"]["disabled"] is True  # enabled:false -> disabled:true
    assert "probe" in servers


def test_unparseable_config_aborts_without_force(tmp_path):
    proc, cfg = _run(tmp_path, preconfig="{ // jsonc comment\n }")
    assert proc.returncode == 3
    # backup taken, original untouched
    assert cfg.read_text() == "{ // jsonc comment\n }"


def test_force_replaces_unparseable_config(tmp_path):
    proc, cfg = _run(tmp_path, preconfig="{ // jsonc comment\n }", extra=["--force"])
    assert proc.returncode == 0, proc.stderr
    assert "probe" in json.loads(cfg.read_text())["mcp"]["servers"]


def test_unknown_tool_skipped(tmp_path):
    proc, cfg = _run(tmp_path, tools="probe nope")
    assert proc.returncode == 0, proc.stderr
    assert "nope" in proc.stderr  # skipped notice
    servers = json.loads(cfg.read_text())["mcp"]["servers"]
    assert "probe" in servers and "nope" not in servers
