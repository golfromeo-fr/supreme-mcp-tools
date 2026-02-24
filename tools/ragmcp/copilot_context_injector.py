#!/usr/bin/env python3
"""
Copilot Context Injector - Format retrieved code for GitHub Copilot consumption.

This module retrieves relevant code chunks using sparse vector search and formats
them as inline comments that GitHub Copilot can understand and use for better
code suggestions.

Usage:
    from copilot_context_injector import CopilotContextInjector

    injector = CopilotContextInjector()
    chunks = search_results  # from sparse vector search
    context = injector.format_context_comment(chunks)
    # Inject context above cursor in VSCode
"""

from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger("copilot_context")


class CopilotContextInjector:
    """
    Format retrieved code chunks for GitHub Copilot context injection.

    Copilot's context window includes:
    - Current file content
    - Open tabs (limited)
    - Inline comments (THIS IS WHERE WE INJECT)

    By injecting relevant code as comments, we make Copilot "see" the entire
    project context without needing to open all files.
    """

    def __init__(self, max_context_lines: int = 50):
        """
        Initialize the context injector.

        Args:
            max_context_lines: Maximum lines of context to inject (default: 50)
        """
        self.max_context_lines = max_context_lines

    def format_context_comment(
        self,
        chunks: List[Dict],
        max_lines: Optional[int] = None,
        language: str = "c"
    ) -> str:
        """
        Format retrieved chunks as multi-line comment for Copilot.

        Args:
            chunks: List of search result chunks with payload data
            max_lines: Override max context lines (default: use instance setting)
            language: Programming language for comment style (c, python, sql)

        Returns:
            Formatted comment block ready for injection

  
        """
        if not chunks:
            return self._format_no_context(language)

        max_lines = max_lines or self.max_context_lines

        # Choose comment style
        if language in ("python", "py", "bash", "sh"):
            return self._format_python_style(chunks, max_lines)
        elif language in ("sql", "plsql"):
            return self._format_sql_style(chunks, max_lines)
        else:
            # Default: C-style (works for C, C++, Java, JavaScript, TypeScript, Pro*C)
            return self._format_c_style(chunks, max_lines)

    def _format_c_style(self, chunks: List[Dict], max_lines: int) -> str:
        """Format as C-style /* */ block comment."""
        lines = []
        lines.append("/*")
        lines.append(" * === CONTEXT: Related Code (Auto-retrieved for Copilot) ===")
        lines.append(" *")

        total_lines = 3

        for i, chunk in enumerate(chunks, 1):
            if total_lines >= max_lines:
                remaining = len(chunks) - i + 1
                lines.append(f" * ... ({remaining} more result{'s' if remaining > 1 else ''} omitted)")
                break

            # Extract chunk data
            file_path = chunk.get('filePath', 'Unknown')
            function_name = chunk.get('functionName', '')
            code = chunk.get('codeChunk', '')
            start_line = chunk.get('startLine', '')
            file_type = chunk.get('fileType', '')

            # Add header with file path
            header = f" * [{i}] {file_path}"
            if function_name:
                header += f":{function_name}()"
            if start_line:
                header += f" (line {start_line})"

            lines.append(header)

            # Add file type hint
            if file_type:
                lines.append(f" *     Type: {file_type}")

            lines.append(" *")
            total_lines += 3

            # Add code snippet (first 8 lines per chunk to stay compact)
            code_lines = code.split('\n')[:8]
            for code_line in code_lines:
                if total_lines >= max_lines:
                    lines.append(" *     ...")
                    break

                # Clean and format code line
                cleaned = code_line.rstrip()
                if cleaned:
                    lines.append(f" *     {cleaned}")
                total_lines += 1

            lines.append(" *")
            total_lines += 1

        lines.append(" */")
        return '\n'.join(lines)

    def _format_python_style(self, chunks: List[Dict], max_lines: int) -> str:
        """Format as Python # comment style."""
        lines = []
        lines.append("# === CONTEXT: Related Code (Auto-retrieved for Copilot) ===")
        lines.append("#")

        total_lines = 2

        for i, chunk in enumerate(chunks, 1):
            if total_lines >= max_lines:
                remaining = len(chunks) - i + 1
                lines.append(f"# ... ({remaining} more result{'s' if remaining > 1 else ''} omitted)")
                break

            file_path = chunk.get('filePath', 'Unknown')
            function_name = chunk.get('functionName', '')
            code = chunk.get('codeChunk', '')
            start_line = chunk.get('startLine', '')

            # Header
            header = f"# [{i}] {file_path}"
            if function_name:
                header += f":{function_name}()"
            if start_line:
                header += f" (line {start_line})"

            lines.append(header)
            lines.append("#")
            total_lines += 2

            # Code snippet
            code_lines = code.split('\n')[:8]
            for code_line in code_lines:
                if total_lines >= max_lines:
                    lines.append("#     ...")
                    break

                cleaned = code_line.rstrip()
                if cleaned:
                    lines.append(f"#     {cleaned}")
                total_lines += 1

            lines.append("#")
            total_lines += 1

        return '\n'.join(lines)

    def _format_sql_style(self, chunks: List[Dict], max_lines: int) -> str:
        """Format as SQL -- comment style."""
        lines = []
        lines.append("-- === CONTEXT: Related Code (Auto-retrieved for Copilot) ===")
        lines.append("--")

        total_lines = 2

        for i, chunk in enumerate(chunks, 1):
            if total_lines >= max_lines:
                remaining = len(chunks) - i + 1
                lines.append(f"-- ... ({remaining} more result{'s' if remaining > 1 else ''} omitted)")
                break

            file_path = chunk.get('filePath', 'Unknown')
            function_name = chunk.get('functionName', '')
            code = chunk.get('codeChunk', '')

            # Header
            header = f"-- [{i}] {file_path}"
            if function_name:
                header += f":{function_name}"

            lines.append(header)
            lines.append("--")
            total_lines += 2

            # Code snippet
            code_lines = code.split('\n')[:8]
            for code_line in code_lines:
                if total_lines >= max_lines:
                    lines.append("--     ...")
                    break

                cleaned = code_line.rstrip()
                if cleaned:
                    lines.append(f"--     {cleaned}")
                total_lines += 1

            lines.append("--")
            total_lines += 1

        return '\n'.join(lines)

    def _format_no_context(self, language: str) -> str:
        """Format when no context is available."""
        if language in ("python", "py"):
            return "# No relevant context found"
        elif language in ("sql", "plsql"):
            return "-- No relevant context found"
        else:
            return "/* No relevant context found */"

    def format_sidebar_context(self, chunks: List[Dict]) -> str:
        """
        Format for VSCode sidebar panel (Markdown format).

        This is an alternative to inline comments, useful for:
        - Displaying context in a separate panel
        - Providing richer formatting
        - Allowing longer context without cluttering the editor

        Returns:
            Markdown-formatted context for display
        """
        if not chunks:
            return "# No Relevant Context Found\n\nTry refining your search query."

        lines = []
        lines.append("# Retrieved Context for Current File\n")
        lines.append("*Auto-retrieved using sparse vector search*\n")

        for i, chunk in enumerate(chunks, 1):
            file_path = chunk.get('filePath', 'Unknown')
            function_name = chunk.get('functionName', '')
            code = chunk.get('codeChunk', '')
            start_line = chunk.get('startLine', '')
            end_line = chunk.get('endLine', '')
            file_type = chunk.get('fileType', 'unknown')

            # Header
            lines.append(f"## [{i}] {file_path}")

            # Metadata
            if function_name:
                lines.append(f"**Function:** `{function_name}()`")

            if start_line and end_line:
                lines.append(f"**Lines:** {start_line}-{end_line}")

            if file_type:
                lines.append(f"**Type:** `{file_type}`")

            lines.append("")

            # Code block with syntax highlighting
            lang_hint = self._get_language_hint(file_type)
            lines.append(f"```{lang_hint}")
            lines.append(code.rstrip())
            lines.append("```")
            lines.append("")

        return '\n'.join(lines)

    def _get_language_hint(self, file_type: str) -> str:
        """Get syntax highlighting hint for markdown code blocks."""
        type_map = {
            'proc': 'c',
            'pc': 'c',
            'plsql': 'sql',
            'sql': 'sql',
            'py': 'python',
            'python': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'c': 'c',
            'cpp': 'cpp',
            'h': 'c'
        }
        return type_map.get(file_type.lower(), '')

    def extract_keywords_from_context(self, current_code: str) -> List[str]:
        """
        Extract relevant keywords from current cursor context.

        This helps build better search queries by identifying:
        - Table names (uppercase identifiers)
        - Function names (lowercase with underscores)
        - SQL keywords (SELECT, FROM, etc.)

        Args:
            current_code: Code snippet around cursor position

        Returns:
            List of extracted keywords for search
        """
        keywords = []

        # Extract SQL table names (all caps, 3+ chars)
        sql_tables = re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', current_code)
        keywords.extend(sql_tables)

        # Extract function names (snake_case identifiers)
        functions = re.findall(r'\b[a-z_][a-z0-9_]*\b', current_code)
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'in', 'to', 'from', 'with', 'for', 'of', 'a', 'an'}
        keywords.extend([f for f in functions if f not in stop_words and len(f) > 3])

        # Extract camelCase identifiers
        camel_case = re.findall(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b', current_code)
        keywords.extend(camel_case)

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def should_inject_context(self, current_code: str) -> bool:
        """
        Determine if context injection would be helpful.

        Skip injection if:
        - Too few keywords extracted
        - Already have inline context
        - In the middle of typing a string

        Args:
            current_code: Current code around cursor

        Returns:
            True if context should be injected
        """
        # Don't inject if already have context comment
        if "=== CONTEXT:" in current_code:
            return False

        # Don't inject inside strings
        if re.search(r'["\'].*\|.*["\']', current_code):
            return False

        # Need at least 2 meaningful keywords
        keywords = self.extract_keywords_from_context(current_code)
        return len(keywords) >= 2


# Singleton instance for convenience
_injector_instance = None

def get_injector(max_context_lines: int = 50) -> CopilotContextInjector:
    """Get singleton context injector instance."""
    global _injector_instance
    if _injector_instance is None:
        _injector_instance = CopilotContextInjector(max_context_lines)
    return _injector_instance
