"""libsql_reconnect — stale-stream reconnect behavior unit tests.

Simulates the live failure (embedded sqld expires idle HTTP streams,
libsql_experimental never reconnects — every later statement fails) with a
fake connection that raises the exact error signature until reconnected.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for pth in (str(PROJECT_ROOT), str(PROJECT_ROOT / "tools")):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from shared.impls.libsql_reconnect import (  # noqa: E402
    ReconnectingLibsql, _looks_stale)


class _FakeCursor:
    def fetchall(self):
        return []


class _StaleConn:
    """Raises the exact live error until 'reconnected' (new instance)."""

    def __init__(self, generation: int):
        self.generation = generation

    @property
    def autocommit(self):
        return self._ac

    # accept assignment like the real conn
    _ac = True

    @autocommit.setter
    def autocommit(self, v):
        self._ac = v

    def execute(self, sql, params=()):
        if self.generation == 0:
            raise RuntimeError(
                "Hrana: `api error: `status=400 Bad Request, body="
                '{"message":"The stream has expired due to inactivity",'
                '"code":"STREAM_EXPIRED"}``')
        return _FakeCursor()


def test_stale_stream_reconnects_and_replays():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _StaleConn(calls["n"] - 1)  # gen 0 = stale, gen 1+ = healthy

    conn = ReconnectingLibsql(factory)
    cur = conn.execute("SELECT 1")          # first conn stale -> reconnect
    assert cur is not None
    assert calls["n"] == 2                  # initial + one reconnect
    conn.execute("SELECT 2")                # healthy from now on
    assert calls["n"] == 2


def test_non_stale_error_propagates_without_reconnect():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        c = _StaleConn(1)                   # healthy generation
        original = c.execute

        def execute(sql, params=()):
            raise ValueError("syntax error near 'FRM'")  # NOT stale
        c.execute = execute
        return c

    conn = ReconnectingLibsql(factory)
    with pytest.raises(ValueError):
        conn.execute("SELEC 1")
    assert calls["n"] == 1                   # no reconnect attempted


def test_second_failure_propagates():
    """A stale error on the FRESH connection too -> raise (no loops)."""

    class _AlwaysStale(_StaleConn):
        def execute(self, sql, params=()):
            raise RuntimeError("The stream has expired due to inactivity")

    conn = ReconnectingLibsql(lambda: _AlwaysStale(1))
    with pytest.raises(RuntimeError):
        conn.execute("SELECT 1")


def test_marker_detection():
    assert _looks_stale(RuntimeError(
        'body={"message":"The stream has expired due to inactivity"}'))
    assert _looks_stale(RuntimeError("Hrana websocket closed"))
    assert _looks_stale(RuntimeError("Connection reset by peer"))
    assert not _looks_stale(RuntimeError("no such table: memories"))
    assert not _looks_stale(ValueError("invalid collection name"))
