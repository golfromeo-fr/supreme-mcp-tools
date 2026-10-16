"""tools/*/config.json is runtime state — never tracked; examples ship instead.

Regression guard for the 2026-10-17 decision: the six per-tool configs carried
real auth.api_key values in the public repo for months (live-verified on
GitHub). config.json is now gitignored (machine-written by the management
API); config.example.json is the committed template and the launcher
bootstraps config.json from it on fresh checkouts
(tool_discovery.ToolDiscovery._bootstrap_config).
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"


def _git_tracked_configs() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "tools/*/config.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    return [line for line in out.splitlines() if line.strip()]


def test_no_tool_config_json_is_tracked():
    tracked = _git_tracked_configs()
    assert tracked == [], (
        "tools/*/config.json is runtime state carrying real api_key values — "
        f"untrack with `git rm --cached {tracked[0]}` and commit "
        "config.example.json instead"
    )


@pytest.mark.parametrize(
    "example", sorted(TOOLS_DIR.glob("*/config.example.json")), ids=lambda p: p.parent.name
)
def test_examples_are_valid_json_with_placeholder_key(example):
    cfg = json.loads(example.read_text())
    key = cfg.get("auth", {}).get("api_key", "")
    assert key.startswith("CHANGE-ME-"), (
        f"{example} must carry a CHANGE-ME- placeholder key, never a real one"
    )


def test_every_disk_config_has_an_example_sibling():
    for cfg in sorted(TOOLS_DIR.glob("*/config.json")):
        assert (cfg.parent / "config.example.json").exists(), (
            f"{cfg.parent.name}/config.json exists on disk but the committed "
            "config.example.json template is missing"
        )


def _make_tool(tmp_path: Path, name: str) -> Path:
    tool_dir = tmp_path / name
    tool_dir.mkdir()
    (tool_dir / f"{name}_fastmcp.py").write_text("mcp = None\n")
    return tool_dir


def test_discovery_bootstraps_config_from_example(tmp_path):
    from launcher.tool_discovery import ToolDiscovery

    tool_dir = _make_tool(tmp_path, "bootstrapsmcp")
    example = {"name": "bootstrapsmcp", "auth": {"api_key": "CHANGE-ME-bootstrapsmcp-api-key"}}
    (tool_dir / "config.example.json").write_text(json.dumps(example))

    ToolDiscovery([str(tmp_path)]).discover_tools()

    assert json.loads((tool_dir / "config.json").read_text()) == example


def test_discovery_never_overwrites_existing_config(tmp_path):
    from launcher.tool_discovery import ToolDiscovery

    tool_dir = _make_tool(tmp_path, "keepmcp")
    real = {"auth": {"api_key": "real-key-live-here"}}
    (tool_dir / "config.json").write_text(json.dumps(real))
    (tool_dir / "config.example.json").write_text(
        json.dumps({"auth": {"api_key": "CHANGE-ME-keepmcp-api-key"}})
    )

    ToolDiscovery([str(tmp_path)]).discover_tools()

    assert json.loads((tool_dir / "config.json").read_text()) == real
