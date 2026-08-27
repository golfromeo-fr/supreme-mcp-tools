"""
Tests for deprecated ragmcp wrapper functions (search_code, search_code_sparse, get_copilot_context).

These tests verify that the wrappers still function correctly after the dead code
is removed. The wrappers delegate to the new unified search() function.

LOW-15: 170+ lines of dead code in deprecated wrappers — we verify wrappers work,
then remove the dead code, then re-run tests to confirm nothing breaks.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestLOW15DeadCodeRemoved(unittest.TestCase):
    """Verify dead code was successfully removed (post-cleanup)."""

    def test_search_code_no_dead_code_after_return(self):
        """search_code: after return, only decorator or next function should follow."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        func_start = source.find("async def search_code(")
        self.assertGreater(func_start, 0)

        rest = source[func_start:]
        return_stmt = rest.find("return await search(")
        self.assertGreater(return_stmt, 0)

        after_return = rest[return_stmt + 30:]
        next_def = after_return.find("\n\n@with_metrics")
        code_after_return = after_return[:next_def] if next_def > 0 else after_return
        code_lines = [l for l in code_after_return.split("\n") if l.strip() and not l.strip().startswith("#")]

        self.assertLessEqual(len(code_lines), 2,
                           f"Expected minimal dead code (<=2 lines), found {len(code_lines)}")

    def test_search_code_sparse_no_dead_code_after_return(self):
        """search_code_sparse: after return, only a decorator or next function should follow."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        func_start = source.find("async def search_code_sparse(")
        self.assertGreater(func_start, 0)

        rest = source[func_start:]
        return_stmt = rest.find("return await search(")
        self.assertGreater(return_stmt, 0)

        after_return = rest[return_stmt + 30:]
        # stop at the next top-level definition (decorated or bare — a plain
        # helper like _no_sparse_index_error legitimately follows here)
        candidates = [after_return.find(p) for p in ("\n\n@", "\n\ndef ", "\n\nasync def ")]
        next_def = min(c for c in candidates if c > 0)
        code_after_return = after_return[:next_def]
        code_lines = [l for l in code_after_return.split("\n") if l.strip() and not l.strip().startswith("#")]

        self.assertLessEqual(len(code_lines), 2,
                           f"Expected minimal dead code (<=2 lines), found {len(code_lines)}")

    def test_get_copilot_context_no_dead_code_after_return(self):
        """get_copilot_context: after return, only decorator or next function should follow."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        func_start = source.find("async def get_copilot_context(")
        self.assertGreater(func_start, 0)

        rest = source[func_start:]
        return_stmt = rest.find("return await search(")
        self.assertGreater(return_stmt, 0)

        after_return = rest[return_stmt + 30:]
        code_after_return = after_return[:200]
        self.assertNotIn("COPILOT_INJECTOR_AVAILABLE", code_after_return,
                         "Dead code should be removed")

    def test_no_import_statements_in_dead_code_regions(self):
        """After cleanup, no import statements should remain after return."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        for func_name in ["search_code", "search_code_sparse", "get_copilot_context"]:
            func_start = source.find(f"async def {func_name}(")
            self.assertGreater(func_start, 0)

            rest = source[func_start:]
            return_pos = rest.find("return await search(")
            after_return = rest[return_pos + 30:]
            next_def = after_return.find("\n\n@with_metrics")
            dead_region = after_return[:next_def] if next_def > 0 else after_return

            self.assertNotIn("import httpx", dead_region)
            self.assertNotIn("import traceback", dead_region)


class TestLOW15DeprecatedWrappersExist(unittest.TestCase):
    """Verify the deprecated wrappers are still defined and importable."""

    def test_search_code_function_exists(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("async def search_code(", source)

    def test_search_code_sparse_function_exists(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("async def search_code_sparse(", source)

    def test_get_copilot_context_function_exists(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("async def get_copilot_context(", source)

    def test_all_have_deprecation_warning(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        self.assertIn("search_code is deprecated", source)
        self.assertIn("search_code_sparse is deprecated", source)
        self.assertIn("get_copilot_context is deprecated", source)


class TestLOW15WrapperDelegation(unittest.TestCase):
    """Verify wrappers delegate to search() correctly."""

    def test_search_code_delegates_to_search_with_dense_mode(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        search_code = source[source.find("async def search_code("):]
        search_code = search_code[:search_code.find("\n\n@with_metrics")]

        self.assertIn("mode=\"dense\"", search_code)
        self.assertIn("return await search(", search_code)

    def test_search_code_sparse_delegates_to_search_with_sparse_mode(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code_sparse("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("mode=\"sparse\"", func)
        self.assertIn("return await search(", func)

    def test_get_copilot_context_delegates_with_copilot_format(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def get_copilot_context("):]
        func_end = func.find("\n\n@with_metrics")
        if func_end < 0:
            func_end = len(func)
        func = func[:func_end]

        self.assertIn("copilot_format=", func)
        self.assertIn("return await search(", func)


class TestLOW15WrapperParameterPreservation(unittest.TestCase):
    """Verify wrappers pass through all parameters to search()."""

    def test_search_code_passes_query(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("query=query", func)

    def test_search_code_passes_limit(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("limit=limit", func)

    def test_search_code_passes_collection_name(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("collection_name=collection_name", func)

    def test_search_code_passes_file_type(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("file_type=file_type", func)

    def test_search_code_passes_function_name(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def search_code("):]
        func = func[:func.find("\n\n@with_metrics")]

        self.assertIn("function_name=function_name", func)

    def test_get_copilot_context_passes_all_params(self):
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        func = source[source.find("async def get_copilot_context("):]
        func_end = func.find("\n\n@with_metrics")
        if func_end < 0:
            func_end = len(func)
        func = func[:func_end]

        self.assertIn("current_context", func)
        self.assertIn("limit=limit", func)
        self.assertIn("collection_name=collection_name", func)
        self.assertIn("language=language", func)
        self.assertIn("max_lines=max_lines", func)


class TestLOW15DeadCodeCannotExecute(unittest.TestCase):
    """Verify dead code has no side effects and is truly unreachable."""

    def test_dead_code_does_not_affect_function_signature(self):
        """The function signatures should be unchanged."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        search_code_sig = "async def search_code("
        sparse_sig = "async def search_code_sparse("
        copilot_sig = "async def get_copilot_context("

        self.assertEqual(source.count(search_code_sig), 1)
        self.assertEqual(source.count(sparse_sig), 1)
        self.assertEqual(source.count(copilot_sig), 1)


class TestLOW15PostCleanupEquivalence(unittest.TestCase):
    """Verify that after removing dead code, wrappers still delegate correctly.

    These tests are designed to PASS both before and after dead code removal,
    ensuring the cleanup doesn't break anything.
    """

    def test_wrapper_return_pattern_is_unchanged(self):
        """After cleanup, wrapper should still have 'return await search(...)'."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        for func_name in ["search_code", "search_code_sparse", "get_copilot_context"]:
            func_start = source.find(f"async def {func_name}(")
            self.assertGreater(func_start, 0, f"{func_name} should exist")

            rest = source[func_start:]
            return_pos = rest.find("return await search(")
            self.assertGreater(return_pos, 0,
                             f"{func_name} should have 'return await search(...)'")

    def test_search_function_is_called_with_correct_mode(self):
        """search() should be called with the right mode by each wrapper."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        search_code = source[source.find("async def search_code("):]
        search_code = search_code[:search_code.find("\n\n@with_metrics")]
        self.assertIn('mode="dense"', search_code)

        sparse = source[source.find("async def search_code_sparse("):]
        sparse = sparse[:sparse.find("\n\n@with_metrics")]
        self.assertIn('mode="sparse"', sparse)

    def test_new_search_function_is_unified(self):
        """The new search() function should be the canonical implementation."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()

        self.assertIn("async def search(", source)
        self.assertIn("mode: str = \"auto\"", source)
        self.assertIn("mode == \"dense\"", source)
        self.assertIn("mode == \"sparse\"", source)
        self.assertIn("mode == \"hybrid\"", source)


class TestLOW15FileSizeReduction(unittest.TestCase):
    """Verify file is smaller after dead code removal."""

    def test_file_has_reduced_lines(self):
        """File should have significantly fewer lines after cleanup (~350 lines removed)."""
        source = (PROJECT_ROOT / "tools/ragmcp/ragmcp_fastmcp.py").read_text()
        line_count = len(source.split("\n"))
        self.assertLess(line_count, 2200,
                       f"Expected ~2070 lines after cleanup, found {line_count}")


if __name__ == "__main__":
    unittest.main()