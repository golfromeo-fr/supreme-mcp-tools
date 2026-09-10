"""D2 — atomic_write_json: the one shared safe-write helper.

Bare truncate-writes corrupted shared config files twice (tools_config.json
truncation bug, fixed only in the UI copy). This pins the helper: concurrent
writers never lose updates, and a crash mid-write leaves the previous file
intact. Also pins that the three converted call sites actually use it.
"""

import json
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.shared.atomic_io import atomic_write_json, atomic_write_text


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "state.json"
        atomic_write_json(path, {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}

    def test_creates_missing_parents(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "state.json"
        atomic_write_json(path, {"ok": True})
        assert json.loads(path.read_text()) == {"ok": True}

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "state.json"
        atomic_write_json(path, {"a": 1})
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "state.json",
            "state.json.lock",
        ]  # .lock sidecar is stable and kept; no .tmp litter

    def test_lock_sidecar_is_stable_across_writes(self, tmp_path):
        """The lock file must never be replaced (flock is inode-bound)."""
        path = tmp_path / "state.json"
        atomic_write_json(path, {"n": 1})
        lock_inode = (path.with_name("state.json.lock")).stat().st_ino
        atomic_write_json(path, {"n": 2})
        assert (path.with_name("state.json.lock")).stat().st_ino == lock_inode

    def test_crash_mid_write_keeps_previous_file(self, tmp_path, monkeypatch):
        """A write that dies between truncate and replace must not corrupt."""
        path = tmp_path / "state.json"
        atomic_write_json(path, {"generation": 1})

        import tools.shared.atomic_io as aio

        real_dump = json.dumps

        def exploding_dump(data, **kw):
            payload = real_dump(data, **kw)
            if payload.count("generation") and "boom" not in payload:
                raise RuntimeError("simulated crash mid-write")
            return payload

        monkeypatch.setattr(aio.json, "dumps", exploding_dump)
        try:
            atomic_write_json(path, {"generation": 2})
        except RuntimeError:
            pass

        # Previous document intact, no tmp litter
        assert json.loads(path.read_text()) == {"generation": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "state.json",
            "state.json.lock",
        ]

    def test_concurrent_writers_never_corrupt(self, tmp_path):
        """8 threads × 25 complete-document writes: every write succeeds and
        the file is always one writer's complete document.

        Scope note: the helper serializes writes and guarantees atomic
        visibility; it does NOT make read-modify-write sequences outside the
        lock transactional (last complete write wins — same semantics as the
        UI pattern this replaces, minus the corruption).
        """
        path = tmp_path / "state.json"
        failures = []
        lock = threading.Lock()

        def worker(tid: int):
            for i in range(25):
                try:
                    atomic_write_json(path, {"writer": tid, "iter": i})
                except Exception as e:  # noqa: BLE001 - record and continue
                    with lock:
                        failures.append(f"{tid}/{i}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert failures == []
        final = json.loads(path.read_text())
        assert set(final) == {"writer", "iter"}

    def test_flock_prevents_tmp_file_hijack(self, tmp_path, monkeypatch):
        """The regression the lock exists for: writer A paused between its tmp
        write and replace must not have its tmp consumed by writer B. Without
        the sidecar lock, B enters the section, truncates the shared tmp, and
        A replaces a half-written file."""
        import tools.shared.atomic_io as aio

        path = tmp_path / "state.json"
        gate = threading.Event()
        release = threading.Event()

        real_open = aio.Path.open

        def slow_tmp_open(self, mode="r", *a, **kw):
            handle = real_open(self, mode, *a, **kw)
            if self.name == "state.json.tmp" and "w" in mode:
                # First writer to create the tmp pauses mid-critical-section
                if not gate.is_set():
                    gate.set()
                    release.wait(timeout=10)
            return handle

        monkeypatch.setattr(aio.Path, "open", slow_tmp_open)
        results = {}

        def writer(payload: str):
            atomic_write_text(path, payload)
            results[payload] = True

        ta = threading.Thread(target=writer, args=("A" * 500,))
        tb = threading.Thread(target=writer, args=("B" * 500,))
        ta.start()
        assert gate.wait(timeout=5), "A never reached the tmp write"
        tb.start()
        # Let B finish fully before releasing A
        for _ in range(100):
            if results.get("B" * 500):
                break
            import time

            time.sleep(0.01)
        release.set()
        ta.join(timeout=10)
        tb.join(timeout=10)

        content = path.read_text()
        assert content in ("A" * 500, "B" * 500), "interleaved/corrupt content"
        assert results.get("A" * 500) and results.get("B" * 500), "a writer died"


class TestConvertedCallSites:
    """The three unsafe writers now route through the shared helper."""

    def test_tools_config_save_is_atomic(self):
        source = (PROJECT_ROOT / "launcher/tools_config.py").read_text()
        assert "atomic_write_json" in source
        assert "open('w')" not in source

    def test_distributed_registry_save_is_atomic(self):
        source = (PROJECT_ROOT / "launcher/distributed_registry.py").read_text()
        assert "atomic_write_json" in source

    def test_management_server_auth_write_is_atomic(self):
        source = (PROJECT_ROOT / "launcher/management_server.py").read_text()
        assert "atomic_write_json" in source

    def test_ui_writer_delegates_to_shared_helper(self):
        """The UI delegates tools_config saves to launcher.tools_config
        (M4/H2: which routes to the shared state backend in db mode); the
        atomic write itself lives there."""
        source = (PROJECT_ROOT / "mcp_ui/components/tool_settings.py").read_text()
        assert "from launcher.tools_config import save_tools_config" in source
        assert "atomic_write_json" not in source
        assert "fcntl" not in source
