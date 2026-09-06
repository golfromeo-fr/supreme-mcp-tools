"""F3 — mcp_ui session-file pruning (user decision 2026-09-06: keep ~1 month).

Every NiceGUI browser session leaves one ``storage-user-*.json`` (pure
cookie-session state); 62 accumulated in six months. The UI prunes files
idle > MCP_UI_SESSION_PRUNE_DAYS (default 30, 0 disables) at startup.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_ui.management_ui import _session_prune_days, prune_stale_session_files


def _make(directory: Path, name: str, age_days: float) -> Path:
    f = directory / name
    f.write_text('{"username": "admin", "authenticated": false}')
    stamp = time.time() - age_days * 86400
    os.utime(f, (stamp, stamp))
    return f


class TestSessionPrune:
    def test_prunes_only_files_older_than_cutoff(self, tmp_path):
        old = _make(tmp_path, "storage-user-a.json", 45)
        fresh = _make(tmp_path, "storage-user-b.json", 2)
        non_session = tmp_path / "unrelated.json"
        non_session.write_text("{}")

        removed = prune_stale_session_files(30, storage_dir=tmp_path)

        assert removed == 1
        assert not old.exists()
        assert fresh.exists()
        assert non_session.exists()

    def test_zero_days_disables_pruning(self, tmp_path):
        old = _make(tmp_path, "storage-user-a.json", 400)
        assert prune_stale_session_files(0, storage_dir=tmp_path) == 0
        assert old.exists()

    def test_missing_dir_is_silent_noop(self, tmp_path):
        assert prune_stale_session_files(30, storage_dir=tmp_path / "absent") == 0

    def test_default_days_env_and_fallback(self, monkeypatch):
        monkeypatch.delenv("MCP_UI_SESSION_PRUNE_DAYS", raising=False)
        assert _session_prune_days() == 30
        monkeypatch.setenv("MCP_UI_SESSION_PRUNE_DAYS", "14")
        assert _session_prune_days() == 14
        monkeypatch.setenv("MCP_UI_SESSION_PRUNE_DAYS", "not-a-number")
        assert _session_prune_days() == 30
        monkeypatch.setenv("MCP_UI_SESSION_PRUNE_DAYS", "0")
        assert _session_prune_days() == 0
