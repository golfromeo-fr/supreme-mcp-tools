#!/usr/bin/env python3
"""
Text utility functions for memorymcp — no FastMCP or Qdrant dependencies.

This module can be safely imported from tests without triggering any
server-side initialization.
"""

import re

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

_UNIVERSAL_ID_RE = re.compile(
    r'\b([A-Z][A-Z0-9_]{2,})\b'
    r'|\b([a-z][a-z0-9_]*_[a-z0-9_]+)\s*\('
    r'|\b([A-Z][A-Za-z0-9]+[a-z][a-z0-9]*)\s*\('
)
_QUOTED_STRING_RE = re.compile(
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
)
_QUOTED_MAPPING_RE = re.compile(
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
    r'\s*[=:]\s*'
    r'["\']([a-zA-Z][a-zA-Z0-9_-]{2,})["\']'
)
_NOISE_NAMES = frozenset({
    'NULL', 'EXEC', 'SQL', 'BEGIN', 'END', 'DECLARE', 'SECTION', 'INTO',
    'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'SELECT', 'INSERT', 'UPDATE',
    'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'SET',
    'VALUES', 'INT', 'LONG', 'SHORT', 'CHAR', 'VOID', 'RETURN', 'IF',
    'ELSE', 'FOR', 'WHILE', 'DO', 'SWITCH', 'CASE', 'BREAK', 'CONTINUE',
    'DEFAULT', 'STRUCT', 'TYPEDEF', 'DEFINE', 'INCLUDE', 'PRINTF', 'SPRINTF',
    'MALLOC', 'FREE', 'SIZEOF', 'ATOL', 'ATOI', 'STRLEN', 'STRCPY', 'STRCAT',
    'MEMSET', 'MEMCPY', 'STDIN', 'STDOUT', 'STDERR', 'EOF', 'EXIT',
    'TRUE', 'FALSE', 'VARCHAR', 'STRING',
    'CONCEPT', 'LESSON', 'TRICK', 'PATTERN', 'IDEA', 'PLAN',
    'PUBLIC', 'STATIC', 'FINAL', 'ABSTRACT', 'PRIVATE', 'PROTECTED',
    'CLASS', 'INTERFACE', 'EXTENDS', 'IMPLEMENTS', 'PACKAGE', 'IMPORT',
    'FUNCTION', 'VAR', 'LET', 'CONST', 'TYPE', 'ASYNC', 'AWAIT',
    'MODULE', 'EXPORT', 'REQUIRE', 'DEFAULT', 'ARGS', 'ERROR', 'INFO', 'DEBUG',
    'NONE', 'SELF', 'THIS', 'SUPER', 'TRAIT', 'ENUM', 'match', 'case',
    'impl', 'pub', 'mut', 'ref', 'static', 'async', 'await', 'yield',
    'raise', 'try', 'catch', 'finally', 'with', 'as', 'from', 'import',
    'lambda', 'map', 'filter', 'reduce', 'zip', 'enumerate', 'range',
    'open', 'close', 'read', 'write', 'seek', 'tell', 'flush', 'fileno',
})


def strip_llm_artifacts(text: str) -> str:
    """Strip LLM artifacts like <think<?>> blocks and XML tags from text."""
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


def extract_verified_names(text: str) -> str:
    """Extract verified technical names from source text."""
    found = set()
    for match in _UNIVERSAL_ID_RE.finditer(text):
        for g in match.groups():
            if g and g not in _NOISE_NAMES:
                found.add(g)
    for match in _QUOTED_STRING_RE.finditer(text):
        val = match.group(1)
        if val not in _NOISE_NAMES:
            found.add(val)
    for match in _QUOTED_MAPPING_RE.finditer(text):
        k, v = match.group(1), match.group(2)
        if k not in _NOISE_NAMES:
            found.add(k)
        if v not in _NOISE_NAMES:
            found.add(v)
    if not found:
        return ""
    lines = ["VERIFIED_NAMES:"]
    for name in sorted(found):
        lines.append(f"  {name}")
    return "\n".join(lines)
