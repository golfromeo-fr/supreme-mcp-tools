"""M4/H2b — env/auth cluster snapshot (mirror-only) unit tests.

Contract under test: snapshots read .env/os.environ but NEVER write it;
restore is additive (missing vars only) and never modifies existing lines.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from launcher import env_manager  # noqa: E402
from tools.shared import state_docs  # noqa: E402


# M5: the fake trio + db_backend fixture moved to tests/_fakes/sql_store.py
# and tests/conftest.py (one canonical definition for all three files).


@pytest.fixture()
def schema_env(monkeypatch):
    """Two fake tools with schemas; env values set only for one tool."""
    monkeypatch.setattr(env_manager, "_schema_tools",
                        lambda: ["webmcp", "simplemcp"])

    schemas = {
        "webmcp": {"BRAVE_API_KEY": {"secret": True},
                   "WEBMCP_TIMEOUT": {"secret": False, "type": "integer"}},
        "simplemcp": {"SIMPLE_GREETING": {"secret": False}},
    }
    monkeypatch.setattr(env_manager, "load_env_schema",
                        lambda tool: schemas.get(tool, {}))
    monkeypatch.setenv("BRAVE_API_KEY", "brave-secret-raw")
    monkeypatch.setenv("WEBMCP_TIMEOUT", "30")
    # SIMPLE_GREETING deliberately unset


def test_snapshot_raw_and_complete(schema_env):
    snap = env_manager.snapshot_env_auth()
    assert snap["env"]["BRAVE_API_KEY"] == "brave-secret-raw"  # raw secrets
    assert snap["env"]["WEBMCP_TIMEOUT"] == "30"
    assert "SIMPLE_GREETING" not in snap["env"]  # unset vars are not mirrored
    assert "node" in snap and "generated_at" in snap


def test_restore_is_additive_only(tmp_path, monkeypatch):
    """Hand-built snapshot (a fresh node has none of these in os.environ —
    snapshot_env_auth would not have mirrored them)."""
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_API_KEY=do-not-touch-me\n# comment\n")
    for var in ("BRAVE_API_KEY", "WEBMCP_TIMEOUT", "SIMPLE_GREETING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(env_manager, "find_env_file", lambda: env_file)

    snap = {"generated_at": "x", "node": "n", "auth": {},
            "env": {"BRAVE_API_KEY": "brave-secret-raw",
                    "WEBMCP_TIMEOUT": "30",
                    "SIMPLE_GREETING": "hi"}}

    report = env_manager.restore_missing_env_vars(snap, apply=True)

    content = env_file.read_text()
    # existing line untouched (additive contract)
    assert "BRAVE_API_KEY=do-not-touch-me" in content
    # missing vars appended
    assert "WEBMCP_TIMEOUT=30" in content
    assert "SIMPLE_GREETING=hi" in content
    assert "BRAVE_API_KEY" not in report["added"]  # present in file → kept
    assert set(report["added"]) == {"WEBMCP_TIMEOUT", "SIMPLE_GREETING"}


def test_restore_dry_run_writes_nothing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n")
    monkeypatch.setattr(env_manager, "find_env_file", lambda: env_file)

    snap = {"env": {"NEW_VAR": "new-value"}}
    report = env_manager.restore_missing_env_vars(snap, apply=False)
    assert report["added"] == ["NEW_VAR"]
    assert "NEW_VAR" not in env_file.read_text()


def test_snapshot_round_trip_through_db(db_backend, monkeypatch, schema_env):
    """The snapshot mirrors through the shared backend (H2b transport)."""
    from tools.shared import state_docs

    snap = env_manager.snapshot_env_auth()
    assert state_docs.save_doc(state_docs.DOC_ENV_AUTH_SNAPSHOT, snap)
    loaded = state_docs.load_doc(state_docs.DOC_ENV_AUTH_SNAPSHOT)
    assert loaded["env"]["BRAVE_API_KEY"] == "brave-secret-raw"
