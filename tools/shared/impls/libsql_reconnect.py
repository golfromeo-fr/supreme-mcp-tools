"""Shared reconnect wrapper for libsql_experimental connections over HTTP.

The embedded-sqld (Hrana over HTTP) server expires idle streams server-side
(``The stream has expired due to inactivity`` / ``STREAM_EXPIRED``) and
libsql_experimental does NOT reconnect on its own: every later statement on
that connection fails forever (found live 2026-09-14 on the turso compose
topology — a node worked at boot, then every write failed after an idle
window).

``ReconnectingLibsql`` wraps a connect factory and retries ONCE on a fresh
connection whenever an error matches the stale-stream signature. Local
``file:`` connections never raise it, so behavior there is unchanged.

Both Turso impls hold ONE shared connection (mutex-serialized in the C
binding — see their __init__ notes); the wrapper swaps that connection
under a lock so a reconnect is visible to all threads on the next call.
"""
from __future__ import annotations

import threading

_STALE_MARKERS = (
    "stream has expired",       # sqld: The stream has expired due to inactivity
    "stream_expired",           # code form
    "hrana",                    # protocol-level errors mention Hrana
    "connection closed",
    "connection reset",
    "broken pipe",
)


def _looks_stale(err: BaseException) -> bool:
    """True when the error signature says 'the HTTP stream went away' —
    i.e. worth one reconnect-and-retry. Conservative: unknown errors are
    NOT retried (they are returned to the caller as-is)."""
    text = str(err).lower()
    return any(m in text for m in _STALE_MARKERS)


def _set_autocommit_if_supported(conn, value: bool) -> None:
    """Some libsql_experimental builds lack a settable ``.autocommit`` on
    the connection object — a bare assignment raises AttributeError.
    Tolerate that (the proxy's ``autocommit`` property reports True either
    way — sqlite's default isolation already behaves autocommit-off-until-
    commit on such builds)."""
    try:
        conn.autocommit = value
    except AttributeError:
        pass


class ReconnectingLibsql:
    """Proxy around a libsql connection: ``execute`` transparently
    reconnects once on stale-stream errors, then replays the statement.

    Only ``execute`` is proxied — it is the single primitive both store
    impls use (plus autocommit, set on every (re)connect)."""

    def __init__(self, connect):
        self._connect = connect          # () -> fresh libsql connection
        self._conn = connect()
        _set_autocommit_if_supported(self._conn, True)
        self._lock = threading.Lock()

    # -- internals --------------------------------------------------------
    def _reconnect(self) -> None:
        new = self._connect()
        _set_autocommit_if_supported(new, True)
        self._conn = new

    # -- proxied surface --------------------------------------------------
    @property
    def autocommit(self) -> bool:
        return True

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        # historical no-op parity: constructors set True on the raw conn;
        # keep accepting the assignment against the CURRENT connection.
        try:
            self._conn.autocommit = value
        except Exception:
            pass

    def execute(self, sql: str, params=()):
        try:
            return self._conn.execute(sql, params)
        except Exception as first_err:
            if not _looks_stale(first_err):
                raise
            with self._lock:
                try:
                    self._reconnect()
                except Exception as recon_err:
                    raise first_err from recon_err
            # replay ONCE on the fresh connection; a second stale/other
            # error propagates to the caller
            return self._conn.execute(sql, params)


def connect_with_reconnect(url: str, auth_token: str | None = None):
    """Drop-in replacement for ``libsql.connect(url[, auth_token])`` that
    returns a ReconnectingLibsql proxy. Local file: URLs get the wrapper
    too — it is a no-op there (file connections never raise the markers)."""
    import libsql_experimental as libsql

    if auth_token:
        return ReconnectingLibsql(
            lambda: libsql.connect(url, auth_token=auth_token))
    return ReconnectingLibsql(lambda: libsql.connect(url))
