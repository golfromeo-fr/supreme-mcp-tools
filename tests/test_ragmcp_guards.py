"""Audit 2026-10-01 P2-4/P2-5: ragmcp collection-name + workspace-root gates.

index_code must reject (a) collection names that could escape the indexing
log path (P2-5) and (b) workspaces outside `allowed_roots` (P2-4) — both
BEFORE any store or subprocess work, with iso-features semantics: every
name/path that was legitimately indexable before still is. The guards are
pure functions in path_guards (no backend init needed for tests).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "ragmcp"))

from indexer.path_guards import (  # noqa: E402
    allowed_roots_for, validate_collection_name, workspace_allowed,
)


@pytest.mark.parametrize("name", [
    "fastmcp-code",
    "folder.to.index-database-code",
    "odysseus-live-sweep-local-bge-m3-1024",
    "ALL",
])
def test_valid_names_pass(name):
    assert validate_collection_name(name) is None


@pytest.mark.parametrize("name", [
    "../../etc/passwd",
    "a/b",
    "a\\b",
    "..",
    "",
    "x" * 121,
])
def test_hostile_names_rejected(name):
    err = validate_collection_name(name)
    assert err is not None and err.startswith("Error: Invalid collection name")


def test_allowed_roots_default_covers_repo_and_tmp(tmp_path):
    cfg = tmp_path / "config.json"
    roots = allowed_roots_for(cfg)  # missing file -> default
    assert roots[-1] == "/tmp"
    assert len(roots) == 2
    # default[0] = the config's parent.parent (mirrors tools/ragmcp/config.json -> repo root)
    assert roots[0] == str(tmp_path.parent)


def test_allowed_roots_reads_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"allowed_roots": ["/srv/code"]}')
    assert allowed_roots_for(cfg) == ["/srv/code"]
    cfg.write_text('{"allowed_roots": []}')
    assert allowed_roots_for(cfg)[-1] == "/tmp"  # empty list -> default


def test_workspace_inside_allowed_root(tmp_path):
    ws = tmp_path / "project" / "src"
    ws.mkdir(parents=True)
    assert workspace_allowed(str(ws), [str(tmp_path)]) is None
    assert workspace_allowed(str(tmp_path), [str(tmp_path)]) is None


def test_workspace_outside_allowed_root_rejected(tmp_path):
    err = workspace_allowed("/etc", [str(tmp_path)])
    assert err is not None and "allowed_roots" in err


def test_workspace_relative_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ws").mkdir()
    assert workspace_allowed("ws", [str(tmp_path)]) is None


def test_dotdot_collapses_before_the_root_check(tmp_path):
    (tmp_path / "sub").mkdir()
    # resolve() collapses '..' — the RESULT decides, so a path that lands
    # inside an allowed root is fine while one that escapes it is not.
    assert workspace_allowed(str(tmp_path / "sub" / ".." / "sub"), [str(tmp_path)]) is None
    assert workspace_allowed(str(tmp_path / "sub" / ".." / ".."), [str(tmp_path)]) is not None
