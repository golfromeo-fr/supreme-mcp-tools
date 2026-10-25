"""
Security Module for FEF V3

Provides rate limiting.
(The AuditLogger was deleted 2026-09-05 — never ran in production;
design pass D1: plans/launcher-persistence-design-2026-09-05.md.
The auth.py module — APIKeyAuth/verify_api_key/require_permission — was
deleted 2026-10-25, review 2026-10-24 L3: re-export-only dead code whose
dict-membership verify was timing-unsafe; every served surface
authenticates via the DualHeaderVerifier in tools/shared/server_factory.py.)
"""

from .rate_limit import RateLimiter

__all__ = [
    "RateLimiter",
]
