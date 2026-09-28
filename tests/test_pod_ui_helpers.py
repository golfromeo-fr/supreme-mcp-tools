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
    # boot-from-bundle MUST use -b: startcluster consumes the bundle only via
    # -b/--config-bundle for up/pg (a positional path is ignored -> host mode)
    assert ops.build_command("work", "up", "-b", "/b/bundle")[0] == [
        str(ops.SCRIPT), "work", "up", "-b", "/b/bundle"]
    assert ops.build_command("test", "pg", "-b", "/b/bundle")[0] == [
        str(ops.SCRIPT), "test", "pg", "-b", "/b/bundle"]
    # global verb: harvest has no env segment
    argv, wipe = ops.build_command(None, "harvest", "myname", "--zip")
    assert argv == [str(ops.SCRIPT), "harvest", "myname", "--zip"] and wipe is False


def test_wipe_actions_exactly_three():
    assert ops.WIPE_ACTIONS == {("work", "clean"), ("work", "restore"), ("test", "restore")}


def test_boot_action():
    assert ops.boot_action("work") == "up"
    assert ops.boot_action("test") == "pg"


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


# ------------------------------------------------- size parsing / cluster disk


def test_parse_size_units():
    assert state.parse_size("48.39MB") == 48_390_000
    assert state.parse_size("18.8kB") == 18_800
    assert state.parse_size("0B") == 0
    assert state.parse_size("1.094GB") == 1_094_000_000
    assert state.parse_size("garbage") == 0


def test_human_size_round_trip():
    assert state.human_size(0) == "0B"
    assert state.human_size(999) == "999B"
    assert state.human_size(48_390_000) == "48.4MB"
    assert state.human_size(1_094_000_000) == "1.1GB"


def test_cluster_disk_sums_containers_and_volumes():
    rows = state.parse_ps_rows(
        "mcp-work_db_1|Up 1 min|9.38MB|\n"
        "mcp-work_work_1|Up 1 min|1.5MB|\n")
    vols = {"mcp-work_work_pgdata": "48.26MB", "mcp-work_work_logs": "180.9kB"}
    total = state.cluster_disk(rows, vols)
    assert "containers 10.9MB" in total and "volumes 48.4MB" in total


def test_parse_size_handles_virtual_suffix():
    assert state.parse_size("1.5MB (virtual 1.2GB)") == 1_500_000
    assert state.parse_size("9.38MB") == 9_380_000


# ------------------------------------------------- V3.9: topology / node.env


NODE_ENV_FIXTURES = {
    "pg": "POSTGRES_HOST=db\nPOSTGRES_PORT=5432\n",
    "turso": "TURSO_DATABASE_URL=http://db:8080\n",
    "external-pg": "POSTGRES_HOST=192.168.1.5\nPOSTGRES_PASSWORD=x\n",
    "external-turso": "TURSO_DATABASE_URL=libsql://cloud.example.com\nTURSO_AUTH_TOKEN=t\n",
}


@pytest.fixture()
def node_env(tmp_path, monkeypatch):
    """Point state.NODE_ENV at a fixture file; returns a setter."""
    f = tmp_path / "node.env"
    monkeypatch.setattr(state, "NODE_ENV", f)

    def set_env(text: str, bundle_header: str | None = None):
        text = (f"# [bundle] GENERATED by startcluster from {bundle_header}\n"
                if bundle_header else "") + text
        f.write_text(text)
        return f
    return set_env


def test_node_env_bundle_parses_header(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")
    assert state.node_env_bundle() is None  # no file
    f = tmp_path / "node.env"
    f.write_text("# [bundle] GENERATED by startcluster from /x/my-bundles/b/env/.env\nK=1\n")
    assert state.node_env_bundle() == Path("/x/my-bundles/b/env/.env")
    f.write_text("K=1\n")
    assert state.node_env_bundle() is None  # no header


def test_test_mode_bundle_vs_host(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")
    monkeypatch.setattr(state, "REPO_ROOT", tmp_path)
    (tmp_path / "deploy").mkdir()
    assert state.test_mode() == "host"  # no header, no override compose
    (tmp_path / "node.env").write_text("# [bundle] GENERATED by startcluster from /x/b\n")
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")
    assert state.test_mode() == "bundle"


def test_node_env_topology_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")

    def set_text(text):
        (tmp_path / "node.env").write_text(text)

    set_text(NODE_ENV_FIXTURES["pg"])
    assert state.node_env_topology() == "pg"
    set_text("# POSTGRES_HOST=db\n" + NODE_ENV_FIXTURES["turso"])  # commented pg
    assert state.node_env_topology() == "turso"
    set_text(NODE_ENV_FIXTURES["external-pg"])
    assert state.node_env_topology() == "external-pg"
    set_text(NODE_ENV_FIXTURES["external-turso"])
    assert state.node_env_topology() == "external-turso"
    set_text("")
    assert state.node_env_topology() == "pg"  # default


def test_topology_choices_host_mode_pins_to_node_env(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")
    monkeypatch.setattr(state, "REPO_ROOT", tmp_path)
    (tmp_path / "node.env").write_text(NODE_ENV_FIXTURES["external-turso"])
    monkeypatch.setattr(state, "node_env_bundle", lambda: None)  # host mode
    c = state.topology_choices()
    assert c["mode"] == "host"
    assert set(c["options"]) == {"external-turso"}  # ONLY the matching topology
    assert c["current"] == "external-turso"


def test_topology_choices_bundle_mode_offers_all(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "NODE_ENV", tmp_path / "node.env")
    monkeypatch.setattr(state, "REPO_ROOT", tmp_path)
    (tmp_path / "node.env").write_text("# [bundle] GENERATED by startcluster from /x/b\n"
                                       + NODE_ENV_FIXTURES["pg"])
    monkeypatch.setattr(state, "node_env_bundle", lambda: Path("/x/b"))
    c = state.topology_choices()
    assert c["mode"] == "bundle"
    assert set(c["options"]) == {"pg", "turso", "external-pg", "external-turso"}


# ------------------------------------------------- preflight + inspect + masking


def test_parse_ss_ports():
    text = ("State  Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
            "LISTEN 0      2048         0.0.0.0:8000       0.0.0.0:*\n"
            "LISTEN 0      2048       127.0.0.1:8200       0.0.0.0:*\n"
            "LISTEN 0      2048               *:18200            *:*\n")
    assert state.parse_ss_ports(text) == {8000, 8200, 18200}


def test_foreign_port_holders_rule(tmp_path, monkeypatch):
    rows_up = state.parse_ps_rows("mcp-work_work_1|Up 1 min|||\n")
    rows_down = state.parse_ps_rows("mcp-work_work_1|Exited (0) 1 min|||\n")
    monkeypatch.setattr(state, "parse_ss_ports", lambda text={}: {8000, 8200})
    monkeypatch.setattr(state, "listening_ports", lambda: {8000, 8200})
    monkeypatch.setattr(state, "containers_by_env",
                        lambda: state.group_containers(rows_up))
    assert state.foreign_port_holders() == []          # work runs: no complaint
    monkeypatch.setattr(state, "containers_by_env",
                        lambda: state.group_containers(rows_down))
    assert state.foreign_port_holders() == [8000, 8200]  # down + ports held


def test_mask_secrets():
    lines = ["PATH=/usr/bin", "POSTGRES_PASSWORD=hunter2",
             "MCP_MANAGEMENT_API_KEY=abc123", "TURSO_DATABASE_URL=file:/x",
             "SIMPLEMCP_SECRET=topsecret"]
    masked = state.mask_secrets(lines)
    assert masked[0] == "PATH=/usr/bin"                       # not secret-named
    assert masked[1] == "POSTGRES_PASSWORD=•••"
    assert masked[2] == "MCP_MANAGEMENT_API_KEY=•••"
    assert masked[3] == "TURSO_DATABASE_URL=file:/x"          # URL, not secret-named
    assert masked[4] == "SIMPLEMCP_SECRET=•••"
    assert "hunter2" not in "\n".join(masked)


def test_minio_password_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "REPO_ROOT", tmp_path)
    (tmp_path / "deploy").mkdir()
    assert state.minio_password_ok() is False                 # no file
    (tmp_path / "deploy" / ".env").write_text("MINIO_ROOT_PASSWORD=ab\n")
    assert state.minio_password_ok() is True
    (tmp_path / "deploy" / ".env").write_text("MINIO_ROOT_PASSWORD=a\n")  # too short
    assert state.minio_password_ok() is False
