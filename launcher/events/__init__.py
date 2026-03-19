"""
Events Module for FEF V3

Provides event sourcing for audit trail and time-travel debugging.
"""

from .sourcing import EventStore, Event

__all__ = [
    "EventStore",
    "Event",
]
