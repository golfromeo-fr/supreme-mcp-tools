#!/usr/bin/env python3
"""
Unit tests for memorymcp text utilities and textToGraph integration.

These tests verify that the text parsing, regex patterns, and graph generation
work correctly without requiring Qdrant or the full MCP server.
"""

import sys
import json
import asyncio
import unittest
import re
from pathlib import Path

_this_dir = str(Path(__file__).resolve().parent.parent / "tools" / "memorymcp")
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from text_utils import strip_llm_artifacts, extract_verified_names


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _text_to_graph(**kwargs):
    from memory_text import textToGraph
    return _run_async(textToGraph(**kwargs))


# ============================================================================
# text_utils: strip_llm_artifacts
# ============================================================================

class TestStripLLMArtifacts(unittest.TestCase):

    def test_removes_think_tags(self):
        text = "Here's some text<think inner thought</think more text"
        result = strip_llm_artifacts(text)
        self.assertNotIn("<think", result)
        self.assertNotIn("</think", result)

    def test_removes_xml_tags(self):
        text = "<?xml version='1.0'?><content>Some text</content>"
        result = strip_llm_artifacts(text)
        self.assertNotIn("<?", result)
        self.assertNotIn("?>", result)

    def test_preserves_content_after_clusters_marker(self):
        text = "Some intro\nCLUSTERS:\n1. cluster content"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CLUSTERS:"))
        self.assertIn("cluster content", result)

    def test_preserves_content_after_compressed_rules_marker(self):
        text = "Preamble here\nCOMPRESSED_RULES:\nrule1: do stuff"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("COMPRESSED_RULES:"))

    def test_preserves_content_after_code_marker(self):
        text = "Some intro\nCODE:\nprint('hello')"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CODE:"))

    def test_strips_llm_preamble_before_marker(self):
        text = "Let me compress this for you.\nI will now proceed.\nCLUSTERS:\n1. cluster A\n2. cluster B"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CLUSTERS:"))
        self.assertNotIn("Let me", result)
        self.assertNotIn("I will", result)

    def test_keeps_clusters_when_followed_by_numbered_list(self):
        text = "CLUSTERS:\n1. first cluster\n2. second cluster"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CLUSTERS:"))
        self.assertIn("first cluster", result)

    def test_empty_input(self):
        self.assertEqual(strip_llm_artifacts(""), "")
        self.assertEqual(strip_llm_artifacts("   "), "   ")

    def test_no_artifacts(self):
        text = "Clean text with no artifacts at all"
        result = strip_llm_artifacts(text)
        self.assertEqual(result, text)

    def test_unclosed_think_tag(self):
        text = "Some text <think this is unclosed"
        result = strip_llm_artifacts(text)
        self.assertNotIn("<think", result)

    def test_multiple_think_tags(self):
        text = "<think first</think middle <think second</think end"
        result = strip_llm_artifacts(text)
        self.assertNotIn("<think", result)

    def test_marker_at_line_zero_returns_full_text(self):
        text = "CLUSTERS:\n1. stuff"
        result = strip_llm_artifacts(text)
        self.assertTrue(result.startswith("CLUSTERS:"))

    def test_only_whitespace_after_marker(self):
        text = "Some preamble\nCLUSTERS:\n\n\n"
        result = strip_llm_artifacts(text)
        self.assertIn("CLUSTERS:", result)


# ============================================================================
# text_utils: extract_verified_names
# ============================================================================

class TestExtractVerifiedNames(unittest.TestCase):

    def test_finds_uppercase_identifiers(self):
        text = "Use MAX_RETRIES and DEFAULT_TIMEOUT constants"
        result = extract_verified_names(text)
        self.assertIn("MAX_RETRIES", result)
        self.assertIn("DEFAULT_TIMEOUT", result)

    def test_finds_snake_case_functions(self):
        text = "Call get_memory() and upsert_memory() functions"
        result = extract_verified_names(text)
        self.assertIn("get_memory", result)
        self.assertIn("upsert_memory", result)

    def test_finds_camel_case_functions(self):
        text = "Use XMLHttpRequest() and JSONObject()"
        result = extract_verified_names(text)
        self.assertIn("HttpRequest", result)

    def test_finds_quoted_strings(self):
        text = 'Set type to "auto-delete" and mode to "code_pattern"'
        result = extract_verified_names(text)
        self.assertIn("auto-delete", result)
        self.assertIn("code_pattern", result)

    def test_finds_quoted_mappings(self):
        text = '"memory_type" : "concept" and "retention" = "auto-delete"'
        result = extract_verified_names(text)
        self.assertIn("memory_type", result)
        self.assertIn("concept", result)
        self.assertIn("retention", result)
        self.assertIn("auto-delete", result)

    def test_excludes_noise_names(self):
        text = "SELECT FROM WHERE AND OR NOT NULL"
        result = extract_verified_names(text)
        self.assertNotIn("SELECT", result)
        self.assertNotIn("NULL", result)

    def test_excludes_short_identifiers(self):
        text = "Use AB and cd() identifiers"
        result = extract_verified_names(text)
        self.assertNotIn("AB", result)

    def test_empty_input(self):
        result = extract_verified_names("")
        self.assertEqual(result, "")

    def test_no_identifiers(self):
        result = extract_verified_names("just plain text nothing special")
        self.assertEqual(result, "")

    def test_output_starts_with_header(self):
        result = extract_verified_names("MAX_RETRIES value")
        self.assertTrue(result.startswith("VERIFIED_NAMES:"))

    def test_sorted_output(self):
        result = extract_verified_names("ZEBRA_CONST and ALPHA_CONST")
        lines = result.strip().split("\n")
        names = [l.strip() for l in lines[1:]]
        self.assertEqual(names, sorted(names))


# ============================================================================
# textToGraph integration tests — output='text' (default)
# ============================================================================

class TestTextToGraphText(unittest.TestCase):

    def test_empty_input(self):
        result = _text_to_graph(text="")
        self.assertIn("Document", result)

    def test_single_heading(self):
        result = _text_to_graph(text="## Introduction\nSome content here")
        self.assertIn("Introduction", result)

    def test_multiple_headings(self):
        md = "# Title\n## Section A\nContent A\n## Section B\nContent B"
        result = _text_to_graph(text=md)
        self.assertIn("Title", result)
        self.assertIn("Section A", result)
        self.assertIn("Section B", result)

    def test_numbered_steps(self):
        md = "## Steps\n1. First thing\n2. Second thing\n3. Third thing"
        result = _text_to_graph(text=md)
        self.assertIn("Step 1", result)
        self.assertIn("Step 2", result)
        self.assertIn("Step 3", result)

    def test_bullet_points(self):
        md = "## Items\n- item one\n- item two\n- item three"
        result = _text_to_graph(text=md)
        self.assertIn("item one", result)
        self.assertIn("item two", result)

    def test_code_block(self):
        md = "## Example\n```python\nprint('hello')\n```"
        result = _text_to_graph(text=md)
        self.assertIn("python code", result)
        self.assertIn("print", result)

    def test_bold_key_value(self):
        md = "## Config\n**Name**: Value\n**Type**: String"
        result = _text_to_graph(text=md)
        self.assertIn("Name: Value", result)
        self.assertIn("Type: String", result)

    def test_bold_standalone_subsection(self):
        md = "## Section\n**Important Note**\nDetails follow"
        result = _text_to_graph(text=md)
        self.assertIn("Important Note", result)

    def test_frontmatter(self):
        md = "---\nname: test\nversion: 1.0\n---\n## Content\nHello"
        result = _text_to_graph(text=md)
        self.assertIn("Metadata", result)
        self.assertIn("name", result)
        self.assertIn("version", result)

    def test_cross_references(self):
        md = "## Section\nCheck [docs](https://example.com) for info"
        result = _text_to_graph(text=md)
        self.assertIn("docs", result)

    def test_custom_title(self):
        result = _text_to_graph(text="## Hello", title="MyDoc")
        self.assertIn("MyDoc", result)

    def test_default_title(self):
        result = _text_to_graph(text="## Hello")
        self.assertIn("Document", result)

    def test_prose_paragraphs(self):
        md = "## Section\n\nFirst paragraph here.\n\nSecond paragraph here.\n"
        result = _text_to_graph(text=md)
        self.assertIn("First paragraph", result)
        self.assertIn("Second paragraph", result)

    def test_heading_hierarchy(self):
        md = "# Root\n## A\n### A1\n### A2\n## B\n### B1"
        result = _text_to_graph(text=md)
        self.assertIn("Root", result)
        self.assertIn("A1", result)
        self.assertIn("B1", result)

    def test_xref_detection(self):
        md = "## See pctech31 and commontech5\nContent"
        result = _text_to_graph(text=md)
        self.assertIn("pctech31", result)
        self.assertIn("commontech5", result)


# ============================================================================
# textToGraph — output='json'
# ============================================================================

class TestTextToGraphJSON(unittest.TestCase):

    def test_json_valid(self):
        result = _text_to_graph(text="## Section\nContent", output="json")
        data = json.loads(result)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

    def test_json_root_node(self):
        result = _text_to_graph(text="## Hello", title="TestDoc", output="json")
        data = json.loads(result)
        root = next(n for n in data["nodes"] if n["type"] == "root")
        self.assertEqual(root["label"], "TestDoc")

    def test_json_heading_node(self):
        result = _text_to_graph(text="## My Section", output="json")
        data = json.loads(result)
        section = next(n for n in data["nodes"] if n["type"] == "section")
        self.assertEqual(section["label"], "My Section")

    def test_json_code_node(self):
        md = "```python\nx = 1\n```"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        code = next(n for n in data["nodes"] if n["type"] == "code")
        self.assertIn("x = 1", code["content"])

    def test_json_step_chaining(self):
        md = "1. Step A\n2. Step B\n3. Step C"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        then_edges = [e for e in data["edges"] if e["relation"] == "then"]
        self.assertEqual(len(then_edges), 2)

    def test_json_frontmatter_nodes(self):
        md = "---\nfoo: bar\n---\n## Content"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        meta = next(n for n in data["nodes"] if n["type"] == "metadata")
        self.assertEqual(meta["label"], "Frontmatter")

    def test_json_xref_edges(self):
        md = "## See also pctech42\nContent"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        xrefs = [e for e in data["edges"] if e["relation"] == "xref"]
        self.assertGreater(len(xrefs), 0)

    def test_json_bullet_items(self):
        md = "- alpha\n- beta\n- gamma"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        items = [n for n in data["nodes"] if n["type"] == "item"]
        self.assertEqual(len(items), 3)

    def test_json_key_value_properties(self):
        md = "**key1**: val1\n**key2**: val2"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        props = [n for n in data["nodes"] if n["type"] == "property"]
        self.assertEqual(len(props), 2)

    def test_json_edges_have_valid_ids(self):
        md = "# Title\n## A\n- item1\n- item2"
        result = _text_to_graph(text=md, output="json")
        data = json.loads(result)
        node_ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)


# ============================================================================
# textToGraph — output='adjacency'
# ============================================================================

class TestTextToGraphAdjacency(unittest.TestCase):

    def test_adjacency_format(self):
        result = _text_to_graph(text="## Section\n- item", output="adjacency")
        self.assertIn("n1 [root]", result)
        self.assertIn("->", result)

    def test_adjacency_leaf_nodes(self):
        result = _text_to_graph(text="## Section\n- item", output="adjacency")
        self.assertIn("content:", result)


# ============================================================================
# textToGraph — output='dot'
# ============================================================================

class TestTextToGraphDot(unittest.TestCase):

    def test_dot_format(self):
        result = _text_to_graph(text="## Section", output="dot")
        self.assertTrue(result.startswith("digraph {"))
        self.assertIn("}", result)

    def test_dot_root_shape(self):
        result = _text_to_graph(text="## Hello", title="Root", output="dot")
        self.assertIn("doublecircle", result)


# ============================================================================
# textToGraph — output='mermaid' and 'both'
# ============================================================================

class TestTextToGraphMermaid(unittest.TestCase):

    def test_mermaid_format(self):
        result = _text_to_graph(text="## Section", output="mermaid")
        self.assertTrue(result.startswith("graph TD"))

    def test_both_format(self):
        result = _text_to_graph(text="## Section", output="both")
        self.assertIn("graph TD", result)
        self.assertIn("```json", result)

    def test_mermaid_edge_styles(self):
        md = "## A\n### B\n- item"
        result = _text_to_graph(text=md, output="mermaid")
        self.assertIn("has_section", result)


if __name__ == "__main__":
    unittest.main()
