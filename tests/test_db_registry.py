"""P2 — ConnectionRegistry: lifecycle, masking, legacy default, busy-skip."""

import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


from dialects._base import DbDialect  # noqa: E402


class FakeDialect(DbDialect):
    name = "fake"
    REQUIRED_PARAMS = ("x",)

    def __init__(self):
        self.connected = []
        self.closed = []

    def connect(self, params):
        self.connected.append(dict(params))
        return object()  # opaque handle

    def close(self, handle):
        self.closed.append(handle)

    def ping(self, handle): ...
    def run_select(self, handle, sql, max_rows): return [], False
    def execute(self, handle, sql): return -1
    def list_tables(self, handle): return []
    def describe_table(self, handle, table): return {"columns": [], "constraints": [], "foreign_keys": []}
    def explain(self, handle, sql): return ""
    def format_error(self, e): return {"error": "DB_ERROR", "code": None, "message": str(e), "offset": None}


@pytest.fixture()
def registry(monkeypatch):
    import dialects
    from connections import ConnectionRegistry

    fake = FakeDialect()
    monkeypatch.setitem(dialects.DIALECTS, "fake", fake)
    monkeypatch.setitem(dialects.DIALECTS, "oracle", fake)
    return ConnectionRegistry(), fake


class TestRegistryLifecycle:
    def test_connect_and_duplicate_rejected(self, registry):
        reg, fake = registry
        reg.connect("a", "fake", {"x": 1})
        with pytest.raises(ValueError, match="already exists"):
            reg.connect("a", "fake", {"x": 2})

    def test_first_connection_becomes_active(self, registry):
        reg, _ = registry
        reg.connect("a", "fake", {"x": 1})
        assert reg.get(None).name == "a"
        reg.connect("b", "fake", {"x": 2})
        assert reg.get(None).name == "a"  # active unchanged by second connect

    def test_disconnect_active_falls_back(self, registry):
        reg, fake = registry
        reg.connect("a", "fake", {"x": 1})
        reg.connect("b", "fake", {"x": 2})
        reg.set_active("a")
        msg = reg.disconnect("a")
        assert msg == "b"
        assert reg.get(None).name == "b"
        assert len(fake.closed) == 1

    def test_disconnect_unknown_raises_listing(self, registry):
        reg, _ = registry
        with pytest.raises(LookupError, match="Unknown connection"):
            reg.disconnect("nope")

    def test_disconnect_last_shows_none(self, registry):
        reg, _ = registry
        reg.connect("a", "fake", {"x": 1})
        assert reg.disconnect("a") == "none"

    def test_get_named_unknown_lists_available(self, registry):
        reg, _ = registry
        reg.connect("a", "fake", {"x": 1})
        with pytest.raises(LookupError, match="Available: \\['a'\\]"):
            reg.get("nope")


class TestLegacyEnvDefault:
    def test_lazy_default_created_from_env_once(self, registry, monkeypatch):
        reg, fake = registry
        monkeypatch.setenv("USERID", "scott/tiger")
        monkeypatch.setenv("DB_HOST", "oracle.example")
        monkeypatch.setenv("DB_AUTOCONNECT", "1")
        results = []
        barrier = threading.Barrier(2)

        def first():
            barrier.wait()
            results.append(reg.get(None).name)

        threads = [threading.Thread(target=first) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == ["default", "default"]
        assert len(fake.connected) == 1  # double-check: exactly ONE connect

    def test_autoconnect_zero_disables_default(self, registry, monkeypatch):
        reg, fake = registry
        monkeypatch.setenv("USERID", "scott/tiger")
        monkeypatch.setenv("DB_HOST", "oracle.example")
        monkeypatch.setenv("DB_AUTOCONNECT", "0")
        with pytest.raises(LookupError, match="connect_database"):
            reg.get(None)
        assert fake.connected == []

    def test_no_env_no_entries_raises_instruction(self, registry, monkeypatch):
        reg, _ = registry
        monkeypatch.delenv("USERID", raising=False)
        monkeypatch.delenv("DB_HOST", raising=False)
        with pytest.raises(LookupError, match="connect_database"):
            reg.get(None)


class TestMaskingAndCloseAll:
    def test_list_never_exposes_params(self, registry):
        reg, _ = registry
        reg.connect("a", "fake", {"x": 1, "password": "hunter2"})
        listed = reg.list()
        flat = str(listed)
        assert "hunter2" not in flat
        assert {"name", "dialect", "state", "active", "cached_tables"} <= set(listed[0])

    def test_mask_params_helper(self):
        from connections import _mask_params

        masked = _mask_params({"user": "u", "PASSWORD": "p", "auth_token": "t", "url": "file:x"})
        assert masked == {"user": "u", "PASSWORD": "***", "auth_token": "***", "url": "file:x"}

    def test_close_all_skips_busy_entries(self, registry):
        reg, fake = registry
        reg.connect("a", "fake", {"x": 1})
        entry = reg.get("a")
        with entry.lock:
            closed, skipped = reg.close_all()
        assert (closed, skipped) == (0, 1)
        assert reg.get("a").name == "a"  # busy entry kept
        closed, skipped = reg.close_all()
        assert (closed, skipped) == (1, 0)
