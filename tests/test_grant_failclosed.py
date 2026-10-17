"""Audit 2026-10-01 P2-3: E3.5 data-plane grants fail CLOSED in multi mode.

An installed user-store with an unresolvable identity (no HTTP context,
unknown token, resolution error) used to inherit admin. In multi mode it
must fail closed to a grant-less user; mono keeps its historical fail-open.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "tools" / "databasemcp",
          PROJECT_ROOT / "tools" / "ragmcp"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _set_mode(monkeypatch, mode):
    monkeypatch.setenv("MCP_AUTH_MODE", mode)
    monkeypatch.delenv("MCP_USERS_STORE_DISABLE", raising=False)


def _break_request(monkeypatch):
    import fastmcp.server.dependencies as deps
    def _raise():
        raise RuntimeError("no http context in this scope")
    monkeypatch.setattr(deps, "get_http_request", _raise)


def test_databasemcp_multi_mode_fails_closed(monkeypatch):
    import connections
    _set_mode(monkeypatch, "multi")
    _break_request(monkeypatch)
    assert connections._current_caller_grants() == ("unknown", "user", [])


def test_databasemcp_mono_stays_fail_open(monkeypatch):
    import connections
    _set_mode(monkeypatch, "mono")
    _break_request(monkeypatch)
    assert connections._current_caller_grants() == ("system", "admin", [])


def test_databasemcp_unresolved_beats_admin_claims(monkeypatch):
    """A grant-less 'user' can never pass preset_grant_error, even for the
    preset an admin would have."""
    import connections
    _set_mode(monkeypatch, "multi")
    _break_request(monkeypatch)
    client_id, role, presets = connections._current_caller_grants()
    assert connections.preset_grant_error("02", None, "conn", client_id, role, presets) is not None


def test_ragmcp_multi_mode_fails_closed(monkeypatch):
    import ragmcp_fastmcp as rf
    _set_mode(monkeypatch, "multi")

    class _V:
        tokens = lambda self: {}  # noqa: E731 — installed user-store map

    monkeypatch.setattr(rf.mcp, "auth", _V())
    _break_request(monkeypatch)
    assert rf._caller_rag_access() == ("unknown", set())


def test_ragmcp_mono_stays_fail_open(monkeypatch):
    import ragmcp_fastmcp as rf
    _set_mode(monkeypatch, "mono")
    monkeypatch.setattr(rf.mcp, "auth", None)  # no verifier at all
    assert rf._caller_rag_access() == (None, None)
