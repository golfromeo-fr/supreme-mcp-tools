"""pod_ui pure helpers (plans/pod-ui-plan-2026-09-19.md §5/§6/§8).
No NiceGUI imports — state.py and ops.build_command only."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pod_ui import ops, state  # noqa: E402


# ------------------------------------------------- podman ps parsing (F2/E1)


def test_parse_ps_rows_exited_row_can_carry_healthy():
    out = (
        "mcp-work_db_1|Up 6 hours (healthy)|127.0.0.1:5433->5432/tcp\n"
        "mcp-multihost_node1_1|Exited (137) About an hour ago (healthy)|0.0.0.0:18200->8200/tcp\n"
        "mcp-multihost_node2_1|Exited (137) About an hour ago|=\n"
    ).replace("=", "")
    rows = state.parse_ps_rows(out.replace("|=\n", "||\n"))
    assert [r.state for r in rows] == ["Up", "Exited", "Exited"]
    assert [r.running for r in rows] == [True, False, False]
    assert "(healthy)" in rows[1].status  # kept raw for the tooltip


def test_parse_ps_rows_ignores_blank_lines():
    assert state.parse_ps_rows("\n\n") == []


# ------------------------------------------------- grouping (E15)


def test_group_containers_routes_strays_to_other():
    rows = state.parse_ps_rows(
        "mcp-work_work_1|Up 1 min||\n"
        "mcp-multihost_db_1|Exited (0) 2 days||\n"
        "stray_container|Up 2 hours||\n"
    )
    grouped = state.group_containers(rows)
    assert [c.name for c in grouped["work"]] == ["mcp-work_work_1"]
    assert [c.name for c in grouped["test"]] == ["mcp-multihost_db_1"]
    assert [c.name for c in grouped["other"]] == ["stray_container"]


# ------------------------------------------------- env state derivation


def test_env_state_derivation():
    assert state.Environment("k", "k", "p", "p").state == "absent"
    all_up = state.parse_ps_rows("a|Up 1 min||\nb|Up 1 min||\n")
    some_up = state.parse_ps_rows("a|Up 1 min||\nb|Exited (0) 1 min||\n")
    none_up = state.parse_ps_rows("a|Exited (0) 1 min||\nb|Exited (137) 1 min||\n")
    env = state.Environment("k", "k", "p", "p")
    env.containers = all_up
    assert env.state == "running"
    env.containers = some_up
    assert env.state == "partial"
    env.containers = none_up
    assert env.state == "stopped"


# ------------------------------------------------- command map (§6, verbatim)


def test_build_command_table():
    assert ops.build_command("work", "up") == ([str(ops.SCRIPT), "work", "up"], False)
    assert ops.build_command("work", "clean") == ([str(ops.SCRIPT), "work", "clean"], True)
    assert ops.build_command("work", "restore", "/b/x.sql") == (
        [str(ops.SCRIPT), "work", "restore", "/b/x.sql"], True)
    assert ops.build_command("test", "restore", "/b/x.sql") == (
        [str(ops.SCRIPT), "test", "restore", "/b/x.sql"], True)
    assert ops.build_command("test", "clean")[1] is False  # no prompt by design
    assert ops.build_command("work", "scrape", "/b/bundle")[0] == [
        str(ops.SCRIPT), "work", "scrape", "/b/bundle"]
    assert ops.build_command("work", "up", "/b/bundle")[0] == [
        str(ops.SCRIPT), "work", "up", "/b/bundle"]
    # global verb: harvest has no env segment
    argv, wipe = ops.build_command(None, "harvest", "myname", "--zip")
    assert argv == [str(ops.SCRIPT), "harvest", "myname", "--zip"] and wipe is False


def test_wipe_actions_exactly_three():
    assert ops.WIPE_ACTIONS == {("work", "clean"), ("work", "restore"), ("test", "restore")}


def test_fatal_line_detection():
    assert ops.fatal_lines(["ok line", "[work] FATAL: ports busy"]) == ["[work] FATAL: ports busy"]


# ------------------------------------------------- listings (E3/E13)


def test_list_bundles_and_backups(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "BUNDLES_DIR", tmp_path / "my-bundles")
    monkeypatch.setattr(state, "BACKUPS_DIR", tmp_path / "backups")

    assert state.list_bundles() == []          # E13: missing dir
    assert state.list_backups() == []

    bdir = tmp_path / "my-bundles" / "b1"
    bdir.mkdir(parents=True)
    (bdir / "manifest.json").write_text('{"created_utc": "t", "git": {"describe": "v1"}}')
    bad = tmp_path / "my-bundles" / "b2"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")

    (tmp_path / "backups").mkdir()
    good = tmp_path / "backups" / "work-pg-20260919-010101.sql"
    good.write_text("dump")
    empty = tmp_path / "backups" / "test-pg-20260919-020202.sql"
    empty.write_text("")
    arts = tmp_path / "backups" / "work-artifacts-20260919-030303.tar.gz"
    arts.write_text("x")
    ignored = tmp_path / "backups" / "readme.txt"
    ignored.write_text("x")

    bundles = state.list_bundles()
    assert [b.name for b in bundles] == ["b1", "b2"]
    assert bundles[0].manifest["git"]["describe"] == "v1"
    assert bundles[1].manifest is None

    backups = state.list_backups()
    by_name = {b.path.name: b for b in backups}
    assert set(by_name) == {"work-pg-20260919-010101.sql",
                            "test-pg-20260919-020202.sql",
                            "work-artifacts-20260919-030303.tar.gz"}
    assert by_name["work-pg-20260919-010101.sql"].env == "work"
    assert by_name["test-pg-20260919-020202.sql"].env == "test"
    assert by_name["work-artifacts-20260919-030303.tar.gz"].kind == "tar.gz"
    assert by_name["test-pg-20260919-020202.sql"].size == 0    # E3: empty flag source
    assert by_name["work-pg-20260919-010101.sql"].size == 4
    # reverse-name order (timestamped names sort within each prefix)
    assert backups == sorted(backups, key=lambda b: b.path.name, reverse=True)


# ------------------------------------------------- runner contract


@pytest.mark.asyncio
async def test_runner_refuses_second_op_and_pipes_wipe(tmp_path):
    asyncio = pytest.importorskip("asyncio")
    from pod_ui.ops import OpRunner

    stub = tmp_path / "stub-startcluster.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$2" = clean ]; then read -r ans; echo "got:$ans"; exit 0; fi\n'
        'if [ "$2" = slow ]; then for i in 1 2 3 4 5; do echo "tick $i"; sleep 0.2; done; exit 0; fi\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    runner = OpRunner(script=stub, history_path=tmp_path / "hist.json")

    # wiped op: ("work","clean") is a wipe action -> the runner pipes 'wipe' into stdin (F5)
    op = await runner.start("work", "clean")
    assert op.done and op.rc == 0
    assert "got:wipe" in op.lines
    assert op.argv[0] == str(stub)  # the stub MUST be honored, not the real script

    # busy refusal (E4): start a slow op, then a second one must be refused
    import asyncio as aio

    task = aio.create_task(runner.start("work", "slow"))
    await aio.sleep(0.15)
    assert runner.running()
    assert await runner.start("work", "slow") is None
    rc = await task
    assert rc.done and rc.rc == 0
    assert len(runner.history) == 2
    wipe_entry = next(h for h in runner.history if h.get("label") == "work clean")
    assert wipe_entry["tail"][-1] == "got:wipe"


@pytest.mark.asyncio
async def test_runner_kill_marks_interrupted(tmp_path):
    from pod_ui.ops import OpRunner

    stub = tmp_path / "stub.sh"
    stub.write_text('#!/usr/bin/env bash\nsleep 30\n')
    stub.chmod(0o755)
    runner = OpRunner(script=stub, history_path=tmp_path / "hist.json")

    import asyncio as aio
    task = aio.create_task(runner.start("work", "slow"))
    await aio.sleep(0.2)
    runner.kill()
    assert runner.kill()
    await aio.wait_for(task, timeout=5)
    assert runner.current.done
    assert runner.current.interrupted


def test_env_signature_changes_when_state_changes():
    up = state.parse_ps_rows("a|Up 1 min||\nb|Up 1 min||\n")
    down = state.parse_ps_rows("a|Exited (0) 1 min||\nb|Up 1 min||\n")
    e1 = state.Environment("k", "k", "p", "p", containers=list(up), health={"h": "ok"})
    e2 = state.Environment("k", "k", "p", "p", containers=list(up), health={"h": "ok"})
    e3 = state.Environment("k", "k", "p", "p", containers=list(down), health={"h": "ok"})
    assert state.env_signature([e1]) == state.env_signature([e2])  # same => skip rebuild
    assert state.env_signature([e1]) != state.env_signature([e3])  # change => rebuild
