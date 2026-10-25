"""Operation runner: builds startcluster commands and runs them with a global
lock, streamed output, and persistent history. The ONLY place pod_ui spawns
processes — every button maps 1:1 to a deploy/startcluster.sh verb (golden
rule, plans/pod-ui-plan-2026-09-19.md §6)."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import state

REPO_ROOT = state.REPO_ROOT
SCRIPT = state.SCRIPT
HISTORY_PATH = REPO_ROOT / ".podui-history.json"

# Exactly the three stdin-'wipe' confirmation sites in startcluster.sh
# (work restore, work clean, test restore — probe fact F5).
WIPE_ACTIONS = {("work", "clean"), ("work", "restore"), ("test", "restore")}


def boot_action(env_key: str) -> str:
    """Boot verb per env (§6 map): work rebuilds with 'up', test with 'pg'."""
    return "up" if env_key == "work" else "pg"


def build_command(env_key: str | None, action: str, *args,
                  script: Path = SCRIPT) -> tuple[list[str], bool]:
    """UI action -> (argv, needs_wipe_stdin). Verbatim §6 map. The script is a
    parameter (default the real startcluster.sh) so tests can inject a stub —
    which MUST be honored: a test that pipes 'wipe' with the real script
    executes a real destructive clean (learned the hard way, 2026-09-19)."""
    if env_key:
        argv = [str(script), env_key, action, *[str(a) for a in args]]
    else:  # global verbs: harvest
        argv = [str(script), action, *[str(a) for a in args]]
    return argv, (env_key or "", action) in WIPE_ACTIONS


def fatal_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if "FATAL" in ln]


@dataclass
class OpState:
    label: str
    argv: list[str]
    started: str
    lines: list[str] = field(default_factory=list)
    done: bool = False
    rc: int | None = None
    duration_s: float | None = None
    interrupted: bool = False
    proc: object | None = field(default=None, repr=False)

    def result_line(self) -> str:
        if not self.done:
            return "… running"
        mark = "✓" if self.rc == 0 else "✗"
        return f"{mark} rc={self.rc} in {self.duration_s:.0f}s" + (" (interrupted)" if self.interrupted else "")


class OpRunner:
    """One operation at a time; a second request is refused (never queued)."""

    def __init__(self, script: Path = SCRIPT, history_path: Path = HISTORY_PATH):
        self.script = script
        self.history_path = history_path
        self.current: OpState | None = None
        self.history: list[dict] = self._load_history()

    def running(self) -> bool:
        return self.current is not None and not self.current.done

    def _load_history(self) -> list[dict]:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save_history(self) -> None:
        try:
            # F7 (review 2026-10-24): atomic write — a crash mid-write used to
            # corrupt .podui-history.json (plain write_text truncate).
            sys.path.insert(0, str(REPO_ROOT))
            from tools.shared.atomic_io import atomic_write_json
            atomic_write_json(self.history_path, self.history[-200:])
        except OSError:
            pass

    async def start(self, env_key: str | None, action: str, *args,
                    label: str | None = None) -> OpState | None:
        """Start an op. Returns None (and does nothing) when one is running —
        the running() guard is the mutual exclusion (no queueing).
        INVARIANT (closes the recorded TOCTOU, fix-band f): the running()
        check and the self.current = op reservation happen with NO await
        between them, so within the single-threaded event loop they are
        atomic — a second concurrent start() always observes running()."""
        if self.running():
            return None
        argv, wipe = build_command(env_key, action, *args, script=self.script)
        op = OpState(label=label or " ".join(argv[1:]),
                     argv=[str(a) for a in argv],
                     started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.current = op
        await self._run(op, wipe)
        self._record(op)
        return op

    async def _run(self, op: OpState, wipe: bool) -> None:
        started = datetime.now(timezone.utc)
        proc = await asyncio.create_subprocess_exec(
            *op.argv, cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if wipe else asyncio.subprocess.DEVNULL,
            start_new_session=True,  # own process group: Kill takes the children too
        )
        op.proc = proc
        if wipe and proc.stdin is not None:
            proc.stdin.write(b"wipe\n")
            await proc.stdin.drain()
            proc.stdin.close()
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            op.lines.append(line.decode(errors="replace").rstrip())
        op.rc = await proc.wait()
        if op.rc is None:
            op.rc = -1
        if op.rc < 0:
            op.interrupted = True
        op.duration_s = (datetime.now(timezone.utc) - started).total_seconds()
        op.done = True

    def kill(self) -> bool:
        """Force-stop the running op: SIGKILL its whole process group (bash
        alone would leave compose/podman children holding the output pipe)."""
        if self.current and not self.current.done and self.current.proc:
            try:
                os.killpg(os.getpgid(self.current.proc.pid), signal.SIGKILL)
                return True
            except (ProcessLookupError, PermissionError):
                return False
        return False

    def _record(self, op: OpState) -> None:
        self.history.append({
            "label": op.label, "argv": op.argv, "started": op.started,
            "rc": op.rc, "duration_s": op.duration_s,
            "interrupted": op.interrupted,
            "tail": op.lines[-50:],
        })
        del self.history[:-200]
        self._save_history()


runner = OpRunner()


def fire(env_key: str | None, action: str, *args) -> bool:
    """UI-button op launch. Returns False (and notifies) when an op is
    already running; True when accepted — callers notify 'started' only
    AFTER this check, so a busy refusal never flashes 'started' first
    (fix-band e). runner.start re-checks atomically (see its invariant)."""
    if runner.running():
        from nicegui import ui
        ui.notify("an operation is already running — watch the console", type="warning")
        return False
    asyncio.ensure_future(runner.start(env_key, action, *args))
    return True
