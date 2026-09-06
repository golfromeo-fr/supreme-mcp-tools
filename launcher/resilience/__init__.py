"""
Resilience Module for FEF V3

Provides retry with exponential backoff.
(The DeadLetterQueue was deleted 2026-09-05 — never ran in production;
design pass D1, plans/launcher-persistence-design-2026-09-05.md.)
"""

from .retry import retry_with_backoff, RetryConfig, RetryExhaustedError

__all__ = [
    "retry_with_backoff",
    "RetryConfig",
    "RetryExhaustedError",
]
