#!/usr/bin/env python3
"""
Generic file filtering rules for code indexing.

Supports any language: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, SQL, etc.
"""

import re
from pathlib import Path
from typing import Tuple

EXCLUSION_PATTERNS = [
    r'.*\.\d{3}\.\w+$',
    r'.*/save\d*/.*',
    r'.*(\.backup|\.bak|\.save|\.old|\.copy|\.orig|\.swp|\.swo).*',
    r'.*\.log$',
    r'.*\.pyc$',
    r'.*\.pyo$',
    r'.*\.egg-info/.*',
    r'.*__pycache__/.*',
    r'.*/node_modules/.*',
    r'.*/\.git/.*',
    r'.*/\.tox/.*',
    r'.*/\.mypy_cache/.*',
    r'.*/\.pytest_cache/.*',
    r'.*/\.venv/.*',
    r'.*/venv/.*',
    r'.*/dist/.*',
    r'.*/build/.*',
    r'.*/\.eggs/.*',
    r'.*/htmlcov/.*',
    r'.*/\.coverage.*',
    r'.*\.lock$',
    r'.*\.min\.(js|css)$',
]

CODE_EXTENSIONS = {
    '.py', '.pyx', '.pxd', '.pyi',
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '.java', '.kt', '.kts', '.scala', '.groovy',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',
    '.go', '.rs', '.zig',
    '.rb', '.gemspec',
    '.php',
    '.swift', '.m', '.mm',
    '.sh', '.bash', '.zsh', '.fish',
    '.sql', '.plsql', '.pls',
    '.r', '.R',
    '.lua',
    '.vim',
    '.el', '.clj', '.ex', '.exs', '.erl', '.hs',
    '.toml', '.yaml', '.yml', '.json', '.xml', '.ini', '.cfg', '.conf',
    '.vue', '.svelte',
    '.dockerfile', '.containerfile',
    '.pc', '.pkg', '.pkb',
    '.md', '.rst', '.txt',
}

VALID_PATTERNS = {}

SCAN_DIRS = {}


def should_keep_file(rel_path: str) -> Tuple[bool, str]:
    for pattern in EXCLUSION_PATTERNS:
        if re.match(pattern, rel_path, re.IGNORECASE):
            return False, f"blacklist: {pattern}"

    ext = Path(rel_path).suffix.lower()
    if ext and ext in CODE_EXTENSIONS:
        return True, "valid code file"

    if not ext and Path(rel_path).name.startswith('.'):
        return False, "hidden file"

    return True, "no filtering rules"
