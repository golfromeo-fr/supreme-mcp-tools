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


class _UserStoreVerifier:
    """UserStoreVerifier shape: marker + callable token map."""
    is_user_store = True
    def tokens(self):
        return {}


class _MonoVerifier:
    """DualHeaderVerifier shape: callable token map, NO marker."""
    def tokens(self):
        return {"system-key": {"client_id": "simplemcp", "scopes": ["mcp"]}}


def _break_request(monkeypatch):
    import fastmcp.server.dependencies as deps
    def _raise():
        raise RuntimeError("no http context in this scope")
    monkeypatch.setattr(deps, "get_http_request", _raise)


def test_databasemcp_user_store_verifier_fails_closed(monkeypatch):
    import connections
    _break_request(monkeypatch)
    monkeypatch.setattr(connections, "_current_caller_grants", connections._current_caller_grants)
    import core
    monkeypatch.setattr(core.mcp, "auth", _UserStoreVerifier(), raising=False)
    assert connections._current_caller_grants() == ("unknown", "user", [])


def test_databasemcp_mono_verifier_stays_fail_open(monkeypatch):
    import connections
    _break_request(monkeypatch)
    import core
    monkeypatch.setattr(core.mcp, "auth", _MonoVerifier(), raising=False)
    assert connections._current_caller_grants() == ("system", "admin", [])


def test_databasemcp_unresolved_beats_admin_claims(monkeypatch):
    """A grant-less 'user' can never pass preset_grant_error, even for the
    preset an admin would have."""
    import connections
    _break_request(monkeypatch)
    import core
    monkeypatch.setattr(core.mcp, "auth", _UserStoreVerifier(), raising=False)
    client_id, role, presets = connections._current_caller_grants()
    assert connections.preset_grant_error("02", None, "conn", client_id, role, presets) is not None


def test_ragmcp_user_store_verifier_fails_closed(monkeypatch):
    import ragmcp_fastmcp as rf
    monkeypatch.setattr(rf.mcp, "auth", _UserStoreVerifier(), raising=False)
    _break_request(monkeypatch)
    assert rf._caller_rag_access() == ("unknown", set())


def test_ragmcp_mono_verifier_stays_fail_open(monkeypatch):
    import ragmcp_fastmcp as rf
    monkeypatch.setattr(rf.mcp, "auth", _MonoVerifier(), raising=False)
    _break_request(monkeypatch)
    assert rf._caller_rag_access() == (None, None)


def test_ragmcp_no_verifier_stays_fail_open(monkeypatch):
    import ragmcp_fastmcp as rf
    monkeypatch.setattr(rf.mcp, "auth", None, raising=False)
    assert rf._caller_rag_access() == (None, None)
