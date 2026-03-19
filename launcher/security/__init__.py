"""
Security Module for FEF V3

Provides API key authentication, rate limiting, and audit logging.
"""

from .auth import APIKeyAuth, verify_api_key, require_permission
from .rate_limit import RateLimiter
from .audit import AuditLogger

__all__ = [
    "APIKeyAuth",
    "verify_api_key",
    "require_permission",
    "RateLimiter",
    "AuditLogger",
]
