"""Atomic file writes for shared state (design pass D2, 2026-09-05).

The launcher, the management server, and the mcp_ui all write the same
config/state files. Bare ``open("w")`` truncate-writes have twice corrupted
these files under concurrency (tools_config.json truncation bug fixed in the
UI copy, 2026-08; the launcher copy stayed unsafe). The proven pattern —
exclusive flock, temp file, ``os.replace`` — lives here so every writer gets
it from one place.

The lock is taken on a stable ``<path>.lock`` sidecar, NOT on the target
file itself: flock is tied to the inode, and ``os.replace`` swaps the inode —
locking the target lets a thread that opened the pre-replace inode into the
critical section alongside a post-replace opener (each then consumes the
other's tmp file; reproduced live 2026-09-05 before the sidecar fix). The
sidecar is never renamed, so every opener locks the same inode. Readers of
the replaced file always see a complete document. fcntl is POSIX-only; on
other platforms the lock is skipped and tmp+replace atomicity remains.
"""

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to ``path`` atomically under an exclusive lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    lock = path.with_name(path.name + ".lock")
    with lock.open("a+") as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass


def atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Write ``data`` as JSON to ``path`` atomically under an exclusive lock."""
    atomic_write_text(path, json.dumps(data, indent=indent))


def atomic_mutate_json(path: Path | str, mutate, default_factory=None) -> Any:
    """Read-modify-write ``path`` as JSON under ONE exclusive flock.

    Review 2026-10-24 (F2/S1): atomic_write_json alone only serializes the
    WRITE — two concurrent read→mutate→write cycles (central 8200 vs the UI,
    two user edits) still lost one mutation's changes. The whole cycle runs
    under the same stable sidecar lock here, so concurrent mutations compose.

    ``mutate(data) -> result`` receives the loaded document (or
    ``default_factory()`` when the file is missing/corrupt — the documented
    tolerant fallbacks) and may mutate it in place; the document is written
    back after it returns. Returns ``mutate``'s result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with lock.open("a+") as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("not a JSON object")
            except (OSError, ValueError):
                data = default_factory() if default_factory else {}
            result = mutate(data)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return result
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
