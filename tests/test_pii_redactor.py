"""
Unit tests for tools/shared/pii_redactor.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


class TestPIIRedactorFindAll(unittest.TestCase):
    """Test PIIRedactor.find_all detection."""

    def setUp(self):
        from shared.pii_redactor import PIIRedactor
        self.redactor = PIIRedactor()

    def test_detect_email(self):
        matches = self.redactor.find_all("Contact me at user@example.com please")
        types = [m.pii_type for m in matches]
        self.assertIn("email", types)

    def test_detect_phone(self):
        matches = self.redactor.find_all("Call 555-123-4567 now")
        types = [m.pii_type for m in matches]
        self.assertIn("phone_us", types)

    def test_detect_api_key(self):
        matches = self.redactor.find_all('api_key=sk_live_1234567890abcdef1234')
        types = [m.pii_type for m in matches]
        self.assertIn("api_key_generic", types)

    def test_detect_github_token(self):
        matches = self.redactor.find_all("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        types = [m.pii_type for m in matches]
        self.assertIn("github_token", types)

    def test_detect_jwt(self):
        matches = self.redactor.find_all("Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456")
        types = [m.pii_type for m in matches]
        self.assertIn("jwt_token", types)

    def test_detect_aws_key(self):
        matches = self.redactor.find_all("Key: AKIAIOSFODNN7EXAMPLE")
        types = [m.pii_type for m in matches]
        self.assertIn("aws_key", types)

    def test_detect_ssn(self):
        matches = self.redactor.find_all("SSN: 123-45-6789")
        types = [m.pii_type for m in matches]
        self.assertIn("ssn", types)

    def test_detect_credit_card(self):
        matches = self.redactor.find_all("Card: 4111111111111111")
        types = [m.pii_type for m in matches]
        self.assertIn("credit_card", types)

    def test_detect_private_key(self):
        matches = self.redactor.find_all("-----BEGIN RSA PRIVATE KEY-----")
        types = [m.pii_type for m in matches]
        self.assertIn("private_key", types)

    def test_detect_ip_address(self):
        matches = self.redactor.find_all("Server at 192.168.1.100")
        types = [m.pii_type for m in matches]
        self.assertIn("ip_address", types)

    def test_no_pii_clean_text(self):
        matches = self.redactor.find_all("This is just a normal sentence about code patterns.")
        redactable = [m for m in matches if m.redact]
        self.assertEqual(len(redactable), 0)

    def test_multiple_pii(self):
        text = "Email: user@example.com Phone: 555-123-4567 Key: api_key=sk_live_abc123def456ghi789jkl"
        matches = self.redactor.find_all(text)
        types = set(m.pii_type for m in matches)
        self.assertGreaterEqual(len(types), 2)

    def test_sorted_by_position(self):
        text = "user@example.com and api_key=sk_live_abc123def456ghi789jkl"
        matches = self.redactor.find_all(text)
        for i in range(len(matches) - 1):
            self.assertLessEqual(matches[i].start, matches[i + 1].start)


class TestPIIRedactorRedact(unittest.TestCase):
    """Test PIIRedactor.redact masking."""

    def setUp(self):
        from shared.pii_redactor import PIIRedactor
        self.redactor = PIIRedactor()

    def test_redact_email(self):
        text = "My email is user@example.com thanks"
        redacted, matches = self.redactor.redact(text)
        self.assertNotIn("user@example.com", redacted)
        self.assertIn("█", redacted)

    def test_redact_custom_mask(self):
        text = "Email: user@example.com"
        redacted, _ = self.redactor.redact(text, mask_char="*")
        self.assertNotIn("user@example.com", redacted)
        self.assertIn("*", redacted)

    def test_preserves_surrounding_text(self):
        text = "Before user@example.com after"
        redacted, _ = self.redactor.redact(text)
        self.assertTrue(redacted.startswith("Before"))
        self.assertTrue(redacted.endswith("after"))

    def test_no_redaction_needed(self):
        text = "Clean text"
        redacted, matches = self.redactor.redact(text)
        self.assertEqual(redacted, text)
        self.assertEqual(len(matches), 0)

    def test_non_redactable_types_preserved(self):
        text = "Server at 192.168.1.100"
        redacted, matches = self.redactor.redact(text)
        self.assertIn("192.168.1.100", redacted)


class TestSensitivityLevel(unittest.TestCase):
    """Test sensitivity level classification."""

    def setUp(self):
        from shared.pii_redactor import PIIRedactor
        self.redactor = PIIRedactor()

    def test_low_sensitivity(self):
        level = self.redactor.get_sensitivity_level("Just some code")
        self.assertEqual(level, "low")

    def test_medium_sensitivity(self):
        text = "Contact user@example.com for details"
        level = self.redactor.get_sensitivity_level(text)
        self.assertEqual(level, "medium")

    def test_high_sensitivity(self):
        text = "Keys: api_key=sk_live_abc123def456ghi789jkl email=user@example.com phone=555-123-4567"
        level = self.redactor.get_sensitivity_level(text)
        self.assertEqual(level, "high")


class TestIsSensitive(unittest.TestCase):
    """Test is_sensitive threshold check."""

    def setUp(self):
        from shared.pii_redactor import PIIRedactor
        self.redactor = PIIRedactor()

    def test_not_sensitive(self):
        sensitive, count = self.redactor.is_sensitive("Clean text")
        self.assertFalse(sensitive)
        self.assertEqual(count, 0)

    def test_sensitive_with_email(self):
        sensitive, count = self.redactor.is_sensitive("user@example.com")
        self.assertTrue(sensitive)
        self.assertGreaterEqual(count, 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def test_redact_sensitive_text(self):
        from shared.pii_redactor import redact_sensitive_text
        result = redact_sensitive_text("user@example.com")
        self.assertNotIn("user@example.com", result)

    def test_redact_sensitive_text_with_matches(self):
        from shared.pii_redactor import redact_sensitive_text
        result = redact_sensitive_text("user@example.com", return_matches=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_check_sensitivity(self):
        from shared.pii_redactor import check_sensitivity
        level = check_sensitivity("Clean text")
        self.assertEqual(level, "low")

    def test_get_redactor_singleton(self):
        from shared.pii_redactor import get_redactor
        r1 = get_redactor()
        r2 = get_redactor()
        self.assertIs(r1, r2)


class TestPIIMatch(unittest.TestCase):
    """Test PIIMatch dataclass."""

    def test_match_fields(self):
        from shared.pii_redactor import PIIMatch
        m = PIIMatch(pii_type="email", matched_text="a@b.com", start=5, end=12, redact=True)
        self.assertEqual(m.pii_type, "email")
        self.assertEqual(m.matched_text, "a@b.com")
        self.assertEqual(m.start, 5)
        self.assertEqual(m.end, 12)
        self.assertTrue(m.redact)

    def test_default_redact(self):
        from shared.pii_redactor import PIIMatch
        m = PIIMatch(pii_type="test", matched_text="x", start=0, end=1)
        self.assertTrue(m.redact)


if __name__ == "__main__":
    unittest.main()
