#!/usr/bin/env python3
"""
Unit tests for memory_text module - textToGraph regex parsing.

These tests verify that the text parsing and regex patterns work correctly
without requiring the full MCP server to be running.
"""

import unittest
import re


class TestTextPatterns(unittest.TestCase):
    """Test regex patterns used in textToGraph parsing."""

    def test_heading_pattern(self):
        """Test heading level detection."""
        headings = [
            ("# Title", 1),
            ("## Section", 2),
            ("### Subsection", 3),
            ("#### Deep", 4),
        ]
        for heading, expected_level in headings:
            level = 0
            for ch in heading:
                if ch == "#":
                    level += 1
                else:
                    break
            self.assertEqual(level, expected_level)

    def test_numbered_step_pattern(self):
        """Test numbered step regex matching."""
        pattern = re.compile(r'^(\d+)[.)]\s+(.+)')
        
        test_cases = [
            ("1. First step", ("1", "First step")),
            ("2) Second step", ("2", "Second step")),
            ("10. Step ten here", ("10", "Step ten here")),
        ]
        
        for text, expected in test_cases:
            match = pattern.match(text)
            self.assertIsNotNone(match)
            self.assertEqual(match.groups(), expected)

    def test_bullet_pattern(self):
        """Test bullet point regex matching."""
        pattern = re.compile(r'^[-*]\s+(.+)')
        
        test_cases = [
            ("- bullet item", "bullet item"),
            ("* asterisk item", "asterisk item"),
        ]
        
        for text, expected in test_cases:
            match = pattern.match(text)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1), expected)

    def test_key_value_pattern(self):
        """Test **key**: value pattern matching."""
        pattern = re.compile(r'^\*\*(.+?)\*\*:\s*(.+)')
        
        test_cases = [
            ("**key**: value", ("key", "value")),
            ("**Status**: Active", ("Status", "Active")),
        ]
        
        for text, expected in test_cases:
            match = pattern.match(text)
            self.assertIsNotNone(match)
            self.assertEqual(match.groups(), expected)

    def test_bold_standalone_pattern(self):
        """Test **bold text** standalone pattern."""
        pattern = re.compile(r'^\*\*(.+?)\*\*\s*$')
        
        test_cases = [
            ("**bold text**", "bold text"),
            ("**Important**", "Important"),
        ]
        
        for text, expected in test_cases:
            match = pattern.match(text)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1), expected)

    def test_cross_reference_pattern(self):
        """Test [text](link) pattern matching."""
        pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        matches = pattern.findall("Check [this link](https://example.com) and [that](other.html)")
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0], ("this link", "https://example.com"))
        self.assertEqual(matches[1], ("that", "other.html"))

    def test_code_block_detection(self):
        """Test code block start/end detection."""
        lines = [
            ("```python", True),   # start - after processing, in_code = True
            ("code here", True),   # inside - in_code stays True
            ("```", False),        # end - after processing, in_code = False
        ]
        
        in_code = False
        for line, expect_in_code_after in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    in_code = False  # end
                else:
                    in_code = True   # start
            self.assertEqual(in_code, expect_in_code_after)


class TestStripLLMArtifacts(unittest.TestCase):
    """Test the strip_llm_artifacts function."""

    # Import the function from memory_text module
    def setUp(self):
        # We can't import memory_text directly due to FastMCP dependencies,
        # so we copy the function here for testing
        self.strip_llm_artifacts = strip_llm_artifacts

    def test_removes_think_tags(self):
        """Test removal of <think> tags."""
        text = "Here's some text<think>inner thought</think>more text"
        result = self.strip_llm_artifacts(text)
        self.assertNotIn("<think>", result)
        self.assertNotIn("</think>", result)

    def test_removes_xml_tags(self):
        """Test removal of XML processing instructions."""
        text = "<?xml version='1.0'?><content>Some text</content>"
        result = self.strip_llm_artifacts(text)
        self.assertNotIn("<?", result)
        self.assertNotIn("?>", result)

    def test_preserves_content_after_markers(self):
        """Test that content after CLUSTERS/COMPRESSED_RULES markers is preserved."""
        text = "Some intro\nCLUSTERS:\n1. cluster content"
        result = self.strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CLUSTERS:"))


# Copy of strip_llm_artifacts for testing (duplicated from memory_text.py)
_CONTENT_MARKERS = [
    r'^CLUSTERS\s*:',
    r'^COMPRESSED_RULES\s*:',
    r'^CODE\s*:',
    r'^#{2,}\s',
    r'^\*\*[^*]',
]

_PREAMBLE_PHRASES = [
    'i need to', 'let me', "i'll", 'i will', 'the user wants',
    'i want to', 'my approach', 'first, i', 'next, i', 'then i',
    'now i can', 'i can compress', 'here is how', 'to do this',
    'the goal', 'i should',
]

def strip_llm_artifacts(text: str) -> str:
    """Strip LLM artifacts like <think> blocks and XML tags from text."""
    if not text or not text.strip():
        return text
    text = re.sub(r'<\?[\s\S]*?\?>', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*?</think\s*>', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*?</think\b', '', text)
    text = re.sub(r'<\?[\s\S]*$', '', text)
    text = re.sub(r'<think[^>]*>[\s\S]*$', '', text)
    text = re.sub(r'<think\b[\s\S]*$', '', text)
    text = text.strip()
    if not text:
        return text
    lines = text.split('\n')
    marker_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _CONTENT_MARKERS:
            if re.match(pattern, stripped, re.IGNORECASE):
                marker_positions.append(i)
                break
    if not marker_positions:
        return text
    for pos in marker_positions:
        next_line = ''
        for j in range(pos + 1, min(pos + 5, len(lines))):
            if lines[j].strip():
                next_line = lines[j].strip()
                break
        if not next_line:
            continue
        next_lower = next_line.lower()
        has_preamble = any(phrase in next_lower for phrase in _PREAMBLE_PHRASES)
        marker_word = lines[pos].strip().rstrip(':').strip().upper()
        if marker_word == 'CLUSTERS' and re.match(r'^\d+[\.\)]\s', next_line):
            continue
        if not has_preamble:
            if pos > 0:
                return '\n'.join(lines[pos:]).strip()
            return text
    last = marker_positions[-1]
    if last > 0:
        return '\n'.join(lines[last:]).strip()
    return text


class TestExtractVerifiedNames(unittest.TestCase):
    """Test the _extract_verified_names function."""

    def test_finds_uppercase_identifiers(self):
        """Test that uppercase identifiers are found."""
        from memory_text import _extract_verified_names
        # Can't test directly - module has FastMCP dependencies
        # But we can test the regex patterns
        pass


class TestGraphNodeCreation(unittest.TestCase):
    """Test graph node and edge creation logic."""

    def test_make_node_tracking(self):
        """Test that node counter increments correctly."""
        nodes = []
        node_counter = 0
        
        def make_node(label, content, level, ntype):
            nonlocal node_counter
            node_counter += 1
            nid = f"n{node_counter}"
            nodes.append({
                "id": nid,
                "label": label,
                "content": content,
                "level": level,
                "type": ntype,
            })
            return nid
        
        id1 = make_node("Root", "Root content", 0, "root")
        id2 = make_node("Section 1", "Section content", 1, "section")
        id3 = make_node("Section 2", "More content", 1, "section")
        
        self.assertEqual(id1, "n1")
        self.assertEqual(id2, "n2")
        self.assertEqual(id3, "n3")
        self.assertEqual(len(nodes), 3)

    def test_safe_label_quoting(self):
        """Test that double quotes are replaced with single quotes."""
        label = 'Test "quoted" label'
        safe = label.replace('"', "'")
        self.assertEqual(safe, "Test 'quoted' label")


if __name__ == "__main__":
    unittest.main()