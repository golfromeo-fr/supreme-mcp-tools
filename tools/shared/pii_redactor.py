"""
PII Redactor - Regex-based PII detection and redaction.

Provides lightweight PII detection without external API calls.
"""

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    pii_type: str
    matched_text: str
    start: int
    end: int
    redact: bool = True


class PIIRedactor:
    """
    Regex-based PII detector and redactor.

    Detects common PII patterns and provides redaction capabilities.
    """

    # PII patterns with descriptions
    PATTERNS = {
        "email": (
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            True  # redact by default
        ),
        "phone_us": (
            r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
            True
        ),
        "ssn": (
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            True
        ),
        "credit_card": (
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            True
        ),
        "api_key_generic": (
            r'\b(?:api[_-]?key|apikey|api[_-]?secret)[=:]\s*["\']?[A-Za-z0-9_-]{20,}["\']?',
            True
        ),
        "aws_key": (
            r'\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b',
            True
        ),
        "aws_secret": (
            r'\b[A-Za-z0-9/+=]{40}\b',
            False  # too broad, may have false positives
        ),
        "private_key": (
            r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
            True
        ),
        "jwt_token": (
            r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b',
            True
        ),
        "github_token": (
            r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b',
            True
        ),
        "ip_address": (
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            False  # not typically sensitive
        ),
        "mac_address": (
            r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
            False
        ),
        "latitude": (
            r'latitude[:\s]+-?\d+\.\d+',
            False
        ),
        "longitude": (
            r'longitude[:\s]+-?\d+\.\d+',
            False
        ),
    }

    def __init__(self):
        self._compiled_patterns = {}
        for name, (pattern, default_redact) in self.PATTERNS.items():
            self._compiled_patterns[name] = (
                re.compile(pattern, re.IGNORECASE),
                default_redact
            )

    def find_all(self, text: str) -> list[PIIMatch]:
        """Find all PII matches in text."""
        matches = []
        for name, (pattern, default_redact) in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=name,
                    matched_text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    redact=default_redact
                ))
        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches

    def redact(self, text: str, mask_char: str = "█") -> tuple[str, list[PIIMatch]]:
        """
        Redact PII from text.

        Args:
            text: Input text
            mask_char: Character to use for masking

        Returns:
            Tuple of (redacted_text, list of matches)
        """
        matches = self.find_all(text)
        if not matches:
            return text, []

        # Build redacted text by replacing from end to start
        # to preserve positions
        result = text
        offset = 0
        redacted_count = 0

        for match in matches:
            if match.redact:
                masked = mask_char * len(match.matched_text)
                start = match.start + offset
                end = match.end + offset
                result = result[:start] + masked + result[end:]
                offset += len(masked) - (match.end - match.start)
                redacted_count += 1

        return result, matches

    def is_sensitive(self, text: str, threshold: int = 1) -> tuple[bool, int]:
        """
        Check if text is sensitive based on PII count.

        Args:
            text: Input text
            threshold: Minimum PII count to consider sensitive

        Returns:
            Tuple of (is_sensitive, pii_count)
        """
        matches = self.find_all(text)
        # Only count items marked for redaction
        redactable = [m for m in matches if m.redact]
        return len(redactable) >= threshold, len(redactable)

    def get_sensitivity_level(self, text: str) -> str:
        """
        Get sensitivity level (low, medium, high) based on PII count.

        Args:
            text: Input text

        Returns:
            "low", "medium", or "high"
        """
        matches = self.find_all(text)
        redactable = [m for m in matches if m.redact]
        count = len(redactable)

        if count == 0:
            return "low"
        elif count <= 2:
            return "medium"
        else:
            return "high"


# Global instance for convenience
_global_redactor: PIIRedactor | None = None


def get_redactor() -> PIIRedactor:
    """Get the global PII redactor instance."""
    global _global_redactor
    if _global_redactor is None:
        _global_redactor = PIIRedactor()
    return _global_redactor


def redact_sensitive_text(
    text: str,
    mask_char: str = "█",
    return_matches: bool = False
) -> str | tuple[str, list[PIIMatch]]:
    """
    Convenience function to redact PII from text.

    Args:
        text: Input text
        mask_char: Character for masking
        return_matches: If True, returns (redacted, matches) tuple

    Returns:
        Redacted text, or (redacted_text, matches) if return_matches=True
    """
    redactor = get_redactor()
    redacted, matches = redactor.redact(text, mask_char)
    if return_matches:
        return redacted, matches
    return redacted


def check_sensitivity(text: str) -> str:
    """
    Check text sensitivity level.

    Args:
        text: Input text

    Returns:
        "low", "medium", or "high"
    """
    return get_redactor().get_sensitivity_level(text)