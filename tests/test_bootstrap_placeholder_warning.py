"""The placeholder-key warning at bootstrap (2026-10-18 little-things #2).

Two failure modes the warning catches: (a) the bootstrap path itself
(the freshly copied placeholder) and (b) a non-bootstrap path where
`config.json` already exists with a placeholder — the bundle-mount
footgun (a `git checkout` over a tracked-but-ignored file clobbering
the live config). Either way the tool is about to boot with a
guessable system key.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.tool_discovery import _warn_placeholder_key  # noqa: E402


def _write_config(tool_dir: Path, key: str) -> None:
    (tool_dir / "config.json").write_text(json.dumps({"auth": {"api_key": key}}))


def test_copied_bootstrap_placeholder_emits_warning(tmp_path, caplog):
    tool_dir = tmp_path / "foomcp"
    tool_dir.mkdir()
    _write_config(tool_dir, "CHANGE-ME-foomcp-api-key")
    with caplog.at_level(logging.WARNING, logger="launcher.tool_discovery"):
        _warn_placeholder_key(tool_dir / "config.json", "foomcp", copied=True)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PLACEHOLDER" in m and "bootstrapped" in m for m in msgs), msgs


def test_existing_placeholder_emits_warning(tmp_path, caplog):
    tool_dir = tmp_path / "barmcp"
    tool_dir.mkdir()
    _write_config(tool_dir, "CHANGE-ME-barmcp-api-key")
    with caplog.at_level(logging.WARNING, logger="launcher.tool_discovery"):
        _warn_placeholder_key(tool_dir / "config.json", "barmcp", copied=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PLACEHOLDER" in m and "already exists" in m for m in msgs), msgs


def test_real_key_emits_no_warning(tmp_path, caplog):
    tool_dir = tmp_path / "goodmcp"
    tool_dir.mkdir()
    _write_config(tool_dir, "barmcp-test-key-real")
    with caplog.at_level(logging.WARNING, logger="launcher.tool_discovery"):
        _warn_placeholder_key(tool_dir / "config.json", "goodmcp", copied=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("PLACEHOLDER" in m for m in msgs), msgs


def test_missing_or_bogus_file_silently_no_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="launcher.tool_discovery"):
        _warn_placeholder_key(tmp_path / "nonexistent.json", "x", copied=False)
        (tmp_path / "broken.json").write_text("{not json")
        _warn_placeholder_key(tmp_path / "broken.json", "x", copied=False)
    assert not caplog.records
