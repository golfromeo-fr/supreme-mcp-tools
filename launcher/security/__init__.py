"""
Security Module for FEF V3

Provides API key authentication and rate limiting.
(The AuditLogger was deleted 2026-09-05 — never ran in production;
design pass D1, plans/launcher-persistence-design-2026-09-05.md.)
"""

from .auth import APIKeyAuth, verify_api_key, require_permission
from .rate_limit import RateLimiter

__all__ = [
    "APIKeyAuth",
    "verify_api_key",
    "require_permission",
    "RateLimiter",
]
