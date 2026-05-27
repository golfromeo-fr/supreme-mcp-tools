#!/usr/bin/env python3
"""
Tests for critical fixes to FastMCP tools.
Covers: bug fixes, security, thread safety, type hints.

Run with: pytest tests/test_fastmcp_critical_fixes.py -v
"""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSimpleMCPFixes:
    """Verify simplemcp critical fixes."""

    def test_fef_manager_defined_before_tools(self):
        content = Path("tools/simplemcp/simplemcp_fastmcp.py").read_text()
        fef_pos = content.find("fef_manager = None")
        first_tool_pos = content.find("@mcp.tool()")
        assert fef_pos != -1, "fef_manager = None not found"
        assert first_tool_pos != -1, "@mcp.tool() not found"
        assert fef_pos < first_tool_pos, (
            f"fef_manager defined at {fef_pos} but first tool at {first_tool_pos}"
        )

    def test_no_unused_lifespan_globals(self):
        content = Path("tools/simplemcp/simplemcp_fastmcp.py").read_text()
        assert "fef_lifespan_manager" not in content
        assert "fef_lifespan_registry" not in content
        assert "fef_lifespan_http_server" not in content


class TestWebMCPFixes:
    """Verify webmcp critical fixes."""

    def test_internal_ip_blocking_function_exists(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "_is_internal_url" in content, "_is_internal_url function missing"

    def test_internal_ip_blocks_localhost(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "'localhost'" in content or '"localhost"' in content
        assert "'127.0.0.1'" in content or '"127.0.0.1"' in content
        assert "169.254.169.254" in content

    def test_internal_ip_blocks_private_ranges(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "ipaddress" in content, "ipaddress module not imported for IP validation"
        assert "is_private" in content, "is_private check missing"

    def test_fetch_url_uses_internal_ip_check(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        fetch_fn_start = content.find('async def fetch_url(')
        assert fetch_fn_start != -1
        fetch_body = content[fetch_fn_start:fetch_fn_start + 2000]
        assert "_is_internal_url" in fetch_body

    def test_thread_safe_cache_has_lock(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "threading.RLock()" in content
        assert "with self.lock:" in content

    def test_thread_safe_cache_all_methods_locked(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        cache_start = content.find("class SimpleCache:")
        cache_end = content.find("_cache = SimpleCache")
        cache_class = content[cache_start:cache_end]
        method_count = content[cache_start:cache_end].count("\n    def ") - 1  # exclude __init__
        lock_count = cache_class.count("with self.lock:")
        assert lock_count >= method_count, (
            f"Found {method_count} methods but only {lock_count} locks"
        )

    def test_brave_search_api_metrics_on_missing_key(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        fn_start = content.find('async def brave_search_api(')
        fn_end = content.find('@mcp.tool()', fn_start + 1) if content.find('@mcp.tool()', fn_start + 1) != -1 else len(content)
        fn_body = content[fn_start:fn_end]
        assert 'webmcp_metrics["search_errors"]' in fn_body or "search_errors" in fn_body

    def test_google_search_api_metrics_on_missing_key(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        fn_start = content.find('async def google_search_api(')
        fn_end = content.find('@mcp.tool()', fn_start + 1) if content.find('@mcp.tool()', fn_start + 1) != -1 else len(content)
        fn_body = content[fn_start:fn_end]
        assert 'webmcp_metrics["search_errors"]' in fn_body or "search_errors" in fn_body

    def test_no_unused_lifespan_globals(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "fef_lifespan_manager" not in content
        assert "fef_lifespan_registry" not in content
        assert "fef_lifespan_http_server" not in content

    def test_uvicorn_has_lifespan(self):
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert 'lifespan="on"' in content


class TestConverterMCPFixes:
    """Verify convertermcp critical fixes."""

    def test_uses_type_imports(self):
        content = Path("tools/convertermcp/convertermcp_fastmcp.py").read_text()
        assert "from typing import Any" in content

    def test_function_signature_uses_optional_style(self):
        content = Path("tools/convertermcp/convertermcp_fastmcp.py").read_text()
        assert "str | None" in content or "Optional[str]" in content

    def test_no_dict_pipe_types_in_function_sigs(self):
        content = Path("tools/convertermcp/convertermcp_fastmcp.py").read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'async def ' in line or 'def ' in line:
                if ' -> ' in line or '(' in line:
                    assert 'dict | None' not in line, (
                        f"Line {i+1} uses dict | None: {line.strip()}"
                    )

    def test_symlink_safe_path_validation(self):
        content = Path("tools/convertermcp/convertermcp_fastmcp.py").read_text()
        assert ".resolve()" in content


class TestOracleMCPFixes:
    """Verify oraclemcp critical fixes."""

    def test_sql_query_logging_in_execute_query(self):
        content = Path("tools/oraclemcp/oraclemcp_fastmcp.py").read_text()
        assert "[SQL]" in content

    def test_sql_query_logging_in_execute_sql(self):
        content = Path("tools/oraclemcp/oraclemcp_fastmcp.py").read_text()
        execute_sql_start = content.find('async def execute_sql(')
        execute_sql_body = content[execute_sql_start:execute_sql_start + 3000]
        assert "[SQL]" in execute_sql_body

    def test_no_unused_lifespan_globals(self):
        content = Path("tools/oraclemcp/oraclemcp_fastmcp.py").read_text()
        assert "fef_lifespan_manager" not in content
        assert "fef_lifespan_registry" not in content
        assert "fef_lifespan_http_server" not in content


class TestRagMCPFixes:
    """Verify ragmcp critical fixes."""

    def test_null_check_in_record_metrics(self):
        content = Path("tools/ragmcp/ragmcp_fastmcp.py").read_text()
        fn_start = content.find('def _record_metrics(')
        fn_end = content.find('\n\n', fn_start + 10)
        fn_body = content[fn_start:fn_end]
        assert "if fef_manager is not None:" in fn_body

    def test_no_unused_lifespan_globals(self):
        content = Path("tools/ragmcp/ragmcp_fastmcp.py").read_text()
        assert "fef_lifespan_manager" not in content
        assert "fef_lifespan_registry" not in content
        assert "fef_lifespan_http_server" not in content


class TestInternalIPFunction:
    """Test _is_internal_url function directly."""

    def _import_function(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "webmcp"))
        from webmcp_fastmcp import _is_internal_url
        return _is_internal_url

    def test_blocks_localhost(self):
        fn = self._import_function()
        assert fn("http://localhost:8080") is True
        assert fn("http://127.0.0.1:8080") is True

    def test_blocks_aws_metadata(self):
        fn = self._import_function()
        assert fn("http://169.254.169.254/latest/meta-data/") is True

    def test_blocks_private_ip(self):
        fn = self._import_function()
        assert fn("http://10.0.0.1/") is True
        assert fn("http://192.168.1.1/") is True
        assert fn("http://172.16.0.1/") is True

    def test_allows_external_urls(self):
        fn = self._import_function()
        assert fn("https://example.com") is False
        assert fn("https://google.com/search") is False

    def test_blocks_0_0_0_0(self):
        fn = self._import_function()
        assert fn("http://0.0.0.0:8080") is True


class TestThreadSafeCache:
    """Test SimpleCache thread safety directly."""

    def _import_cache(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "webmcp"))
        from webmcp_fastmcp import SimpleCache
        return SimpleCache

    def test_concurrent_access(self):
        Cache = self._import_cache()
        cache = Cache(default_ttl=60)
        errors = []

        def writer(thread_id):
            try:
                for i in range(100):
                    cache.set(f"key_{thread_id}_{i}", f"value_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(100):
                    cache.get(f"key_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(5):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_cleanup_expired_thread_safe(self):
        Cache = self._import_cache()
        cache = Cache(default_ttl=1)
        cache.set("key1", "value1", ttl=0.01)
        time.sleep(0.02)

        errors = []

        def cleanup():
            try:
                cache.cleanup_expired()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Cleanup thread safety errors: {errors}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
