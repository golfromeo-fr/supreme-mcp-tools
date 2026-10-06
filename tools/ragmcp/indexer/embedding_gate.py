"""
Dead-endpoint latch for remote embedding HTTP calls.

After N consecutive failures the gate latches for a cooldown window and
fails fast with an actionable message instead of re-probing a dead endpoint
on every call — each probe costs a full HTTP timeout, and indexing loops
can burn hours against an endpoint that stopped answering (or a key that
expired).

Pattern re-derived from odysseus's dead-endpoint latch; zero code taken.

Knobs:
- RAGMCP_EMBED_LATCH_THRESHOLD  consecutive failures before latching (default 3)
- RAGMCP_EMBED_LATCH_SECONDS    cooldown window; 0 disables latching (default 60)

State is per-process and in-memory by design: a fresh process should be
allowed to re-probe.
"""
from __future__ import annotations

import os
import time


class EmbeddingEndpointDown(RuntimeError):
    """Raised while the latch is closed after repeated endpoint failures."""


def _threshold() -> int:
    try:
        return max(1, int(os.getenv("RAGMCP_EMBED_LATCH_THRESHOLD", "3")))
    except ValueError:
        return 3


def _cooldown_s() -> float:
    try:
        return max(0.0, float(os.getenv("RAGMCP_EMBED_LATCH_SECONDS", "60")))
    except ValueError:
        return 60.0


_state = {"failures": 0, "latched_until": 0.0}


def check(endpoint: str) -> None:
    """
    Raise EmbeddingEndpointDown while latched. Cheap — call before each
    remote embedding request.
    """
    if _state["latched_until"] > time.monotonic():
        remaining = int(_state["latched_until"] - time.monotonic())
        raise EmbeddingEndpointDown(
            f"Embedding endpoint '{endpoint}' is latched DOWN for another "
            f"{remaining}s after {_state['failures']} consecutive failures. "
            "Fix the endpoint (or set RAGMCP_EMBED_LATCH_SECONDS=0 to disable "
            "latching) and retry."
        )


def record_success() -> None:
    """A successful call resets the failure streak and opens the latch."""
    _state["failures"] = 0
    _state["latched_until"] = 0.0


def record_failure(endpoint: str) -> None:
    """
    Count a failed call (transport errors, timeouts, 4xx/5xx). At the
    threshold the latch closes for the cooldown window.
    """
    cooldown = _cooldown_s()
    if cooldown <= 0:
        return
    _state["failures"] += 1
    if _state["failures"] >= _threshold():
        _state["latched_until"] = time.monotonic() + cooldown
        import logging
        logging.getLogger("embedding_gate").error(
            f"Embedding endpoint '{endpoint}' latched DOWN for {cooldown:.0f}s "
            f"after {_state['failures']} consecutive failures"
        )


def reset() -> None:
    """Open the latch and clear the streak (manual un-latch / tests)."""
    _state["failures"] = 0
    _state["latched_until"] = 0.0
