#!/usr/bin/env python3
"""
Unit tests for tools/memorymcp/memory_autouse.py

Dependency-free: no Qdrant, no PostgreSQL, no running MCP server.

Covers:
  1. getMemoryAutousePolicy — reads repo-local policy file
  2. getMemoryAutousePolicy — fallback when file is missing
  3. getMemoryCheatsheet — reads repo-local cheatsheet file
  4. getMemoryCheatsheet — fallback when file is missing
  5. _patch_memory_system_prompt — prepends pointer, idempotent
  6. Repo-local files exist and have expected content
  7. getMetaDecisions — 2-level + priority shortcut, default priority A
  8. getMetaDecisions — priority override B/C
  9. getMetaDecisions — invalid priority falls back to A

Run:
    python -m pytest tests/test_memory_autouse.py -v
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest


def _run(coro):
    """Run a coroutine on a LOCAL event loop, never touching the global state.

    Why not `asyncio.run(coro)`? Because `asyncio.run` calls
    `asyncio.set_event_loop(None)` on exit, and in Python 3.13+
    subsequent tests that call `asyncio.get_event_loop()` (e.g. the helper
    in tests/test_memory_text.py) then raise `RuntimeError: There is no
    current event loop in thread 'MainThread'`. Using a local loop here
    keeps the global state untouched.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _install_memory_core_stub():
    """Install a fake memory_core module in sys.modules.

    Returns (fake_mcp, core_stub). Must be paired with _uninstall_memory_core_stub
    (see the autouse_module fixture) to avoid leaking the stub into other
    test files. A leaked stub causes ImportError on `from memory_core import X`
    in any test that runs after this module is collected by pytest.
    """
    repo_root = Path(__file__).resolve().parent.parent
    memorymcp_dir = repo_root / "tools" / "memorymcp"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(memorymcp_dir))

    class _FakeMCP:
        def __init__(self):
            self.registered = []

        def tool(self):
            def decorator(fn):
                self.registered.append(fn)
                return fn
            return decorator

    fake_mcp = _FakeMCP()

    core_stub = types.ModuleType("memory_core")
    core_stub.mcp = fake_mcp
    core_stub.logger = type("L", (), {
        "info": lambda *a, **k: None,
        "debug": lambda *a, **k: None,
        "warning": lambda *a, **k: None,
    })()
    core_stub.SCRIPT_DIR = memorymcp_dir
    sys.modules["memory_core"] = core_stub
    return fake_mcp, core_stub


@pytest.fixture(scope="module")
def autouse_module():
    """Install the memory_core stub, re-import memory_autouse, yield it,
    then restore the real memory_core so other test files are not affected.
    """
    real_memory_core = sys.modules.get("memory_core")
    _install_memory_core_stub()
    sys.modules.pop("memory_autouse", None)
    try:
        import memory_autouse
        yield sys.modules["memory_autouse"]
    finally:
        # Restore the real memory_core so other test files can import it.
        if real_memory_core is not None:
            sys.modules["memory_core"] = real_memory_core
        else:
            sys.modules.pop("memory_core", None)


# ---------------------------------------------------------------------------
# 1 & 2. getMemoryAutousePolicy
# ---------------------------------------------------------------------------

def test_get_policy_reads_repo_local_file(autouse_module, tmp_path, monkeypatch):
    policy = "# test policy\n\nCall queryMemory first.\n"
    f = tmp_path / "auto_use_policy.md"
    f.write_text(policy)
    monkeypatch.setattr(autouse_module, "POLICY_FILE", f)

    result = _run(autouse_module.getMemoryAutousePolicy())
    assert "test policy" in result
    assert "queryMemory" in result


def test_get_policy_fallback_when_missing(autouse_module, tmp_path, monkeypatch):
    monkeypatch.setattr(autouse_module, "POLICY_FILE", tmp_path / "nonexistent.md")
    result = _run(autouse_module.getMemoryAutousePolicy())
    assert "auto-use policy (inline fallback)" in result
    assert "queryMemory" in result
    assert "upsertMemory" in result


# ---------------------------------------------------------------------------
# 3 & 4. getMemoryCheatsheet
# ---------------------------------------------------------------------------

def test_get_cheatsheet_reads_repo_local_file(autouse_module, tmp_path, monkeypatch):
    digest = "# cheatsheet\n- call queryMemory first\n"
    f = tmp_path / "auto_use_cheatsheet.md"
    f.write_text(digest)
    monkeypatch.setattr(autouse_module, "CHEATSHEET_FILE", f)

    result = _run(autouse_module.getMemoryCheatsheet())
    assert "cheatsheet" in result
    assert "call queryMemory first" in result


def test_get_cheatsheet_fallback_when_missing(autouse_module, tmp_path, monkeypatch):
    monkeypatch.setattr(autouse_module, "CHEATSHEET_FILE", tmp_path / "nonexistent.md")
    result = _run(autouse_module.getMemoryCheatsheet())
    assert "cheatsheet (inline fallback)" in result
    assert "queryMemory" in result


# ---------------------------------------------------------------------------
# 5. _patch_memory_system_prompt
# ---------------------------------------------------------------------------

def test_patch_prepends_pointer(autouse_module):
    fake_mt = types.ModuleType("memory_tools")

    async def original():
        return "ORIGINAL_BODY"

    fake_mt.getMemorySystemPrompt = original

    # CRITICAL: save+restore sys.modules["memory_tools"]. If we leave our
    # fake there, subsequent tests (notably test_regression's tool discovery
    # which loads memorymcp_fastmcp.py) will import from our fake and
    # fail with ImportError on the real `setup_fef_v3` symbol.
    original_modules = sys.modules.get("memory_tools")
    sys.modules["memory_tools"] = fake_mt
    try:
        autouse_module._patch_memory_system_prompt()

        patched = sys.modules["memory_tools"].getMemorySystemPrompt
        assert patched is not original
        assert getattr(patched, "_autouse_patched", False) is True

        out = _run(patched())
        assert out.startswith("## Auto-Use Policy (read first)")
        assert "ORIGINAL_BODY" in out
        assert out.index("## Auto-Use Policy") < out.index("ORIGINAL_BODY")
    finally:
        if original_modules is not None:
            sys.modules["memory_tools"] = original_modules
        else:
            sys.modules.pop("memory_tools", None)


def test_patch_is_idempotent(autouse_module):
    fake_mt = types.ModuleType("memory_tools_idem")

    async def original():
        return "BODY"

    fake_mt.getMemorySystemPrompt = original

    # CRITICAL: save+restore sys.modules["memory_tools"]. Same reason as
    # test_patch_prepends_pointer above.
    original_modules = sys.modules.get("memory_tools")
    sys.modules["memory_tools"] = fake_mt
    try:
        autouse_module._patch_memory_system_prompt()
        first = fake_mt.getMemorySystemPrompt

        autouse_module._patch_memory_system_prompt()
        second = fake_mt.getMemorySystemPrompt

        assert first is second
        out = _run(second())
        assert out.count("## Auto-Use Policy (read first)") == 1
    finally:
        if original_modules is not None:
            sys.modules["memory_tools"] = original_modules
        else:
            sys.modules.pop("memory_tools", None)


# ---------------------------------------------------------------------------
# 6. Repo-local files exist and have expected content
# ---------------------------------------------------------------------------

def test_repo_local_policy_file_exists_and_valid(autouse_module):
    policy_file = autouse_module.POLICY_FILE
    assert policy_file.exists(), f"Policy file missing: {policy_file}"

    result = _run(autouse_module.getMemoryAutousePolicy())
    assert "memorymcp" in result.lower()
    assert "queryMemory" in result
    assert "upsertMemory" in result
    assert "onAgentAction" in result
    assert "createMemoryEdge" in result


def test_repo_local_cheatsheet_file_exists_and_valid(autouse_module):
    cheatsheet_file = autouse_module.CHEATSHEET_FILE
    assert cheatsheet_file.exists(), f"Cheatsheet file missing: {cheatsheet_file}"

    result = _run(autouse_module.getMemoryCheatsheet())
    assert "queryMemory" in result
    assert "upsertMemory" in result


# ---------------------------------------------------------------------------
# 7. getMetaDecisions (2-level analysis shortcut)
# ---------------------------------------------------------------------------

def test_get_meta_decisions_filters_by_level_meta(autouse_module):
    """getMetaDecisions should call queryMemory with tags=['level:meta'] and
    the default priority:A filter.

    This is the 2-level + priority analysis shortcut: lets the LLM ask
    'why was X built this way?' and get ONLY must-know meta decisions.
    """
    fake_mt = types.ModuleType("memory_tools_for_meta")

    captured = {}

    async def fake_queryMemory(query, k=10, memory_type=None, tags=None,
                              agent_id=None, recency_weight=0.5):
        captured["query"] = query
        captured["k"] = k
        captured["memory_type"] = memory_type
        captured["tags"] = tags
        return f"FAKE: meta decisions for '{query}'"

    fake_mt.queryMemory = fake_queryMemory
    sys.modules["memory_tools_for_meta"] = fake_mt

    # The in-function import in getMetaDecisions reads sys.modules['memory_tools']
    # at call time, so we patch it here (autouse_module was already imported).
    original = sys.modules.get("memory_tools")
    sys.modules["memory_tools"] = fake_mt

    try:
        result = _run(
            autouse_module.getMetaDecisions(query="authentication", k=3)
        )

        # The fake was called with the right args
        assert captured["query"] == "authentication"
        assert captured["k"] == 3
        assert captured["tags"] == ["level:meta", "priority:A"]
        # No filter on memory_type — meta filter is via tag only
        assert captured["memory_type"] is None

        # The fake's return value is passed through unchanged
        assert "FAKE: meta decisions for 'authentication'" in result
    finally:
        # Restore sys.modules so other tests get a clean slate
        if original is not None:
            sys.modules["memory_tools"] = original
        else:
            sys.modules.pop("memory_tools", None)


def test_get_meta_decisions_priority_override(autouse_module):
    """getMetaDecisions(priority='B') should pass priority:B tag."""
    fake_mt = types.ModuleType("memory_tools_for_meta_b")

    captured = {}

    async def fake_queryMemory(query, k=10, memory_type=None, tags=None,
                              agent_id=None, recency_weight=0.5):
        captured["tags"] = tags
        return "FAKE"

    fake_mt.queryMemory = fake_queryMemory
    sys.modules["memory_tools_for_meta_b"] = fake_mt

    original = sys.modules.get("memory_tools")
    sys.modules["memory_tools"] = fake_mt

    try:
        _run(autouse_module.getMetaDecisions(query="x", k=2, priority="B"))
        assert captured["tags"] == ["level:meta", "priority:B"]

        _run(autouse_module.getMetaDecisions(query="x", k=2, priority="C"))
        assert captured["tags"] == ["level:meta", "priority:C"]
    finally:
        if original is not None:
            sys.modules["memory_tools"] = original
        else:
            sys.modules.pop("memory_tools", None)


def test_get_meta_decisions_invalid_priority_defaults_to_a(autouse_module):
    """getMetaDecisions(priority='garbage') should silently fall back to A
    rather than passing an invalid tag."""
    fake_mt = types.ModuleType("memory_tools_for_meta_invalid")

    captured = {}

    async def fake_queryMemory(query, k=10, memory_type=None, tags=None,
                              agent_id=None, recency_weight=0.5):
        captured["tags"] = tags
        return "FAKE"

    fake_mt.queryMemory = fake_queryMemory
    sys.modules["memory_tools_for_meta_invalid"] = fake_mt

    original = sys.modules.get("memory_tools")
    sys.modules["memory_tools"] = fake_mt

    try:
        _run(autouse_module.getMetaDecisions(query="x", k=2, priority="z"))
        assert captured["tags"] == ["level:meta", "priority:A"]
    finally:
        if original is not None:
            sys.modules["memory_tools"] = original
        else:
            sys.modules.pop("memory_tools", None)
