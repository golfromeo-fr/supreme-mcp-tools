"""
Regression tests for code review fixes.

Tests verify specific bugs identified by graphify analysis:
1. Port leak on partial startup failure (launchmcp.py)
2. Exception chaining preserved in fef_integration.py
3. No duplicate exception handlers in webmcp streamable
4. get_registry_for_tool helper exists and works
"""
import ast
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExceptionChainingFefIntegration:
    """Test that fef_integration.py uses 'raise ... from e' to preserve exception traces."""

    def test_value_error_preserves_chaining(self):
        """Verify ValueError in fef_integration uses 'raise ... from e'."""
        content = Path("tools/fef_integration.py").read_text()

        assert "raise ValueError(" in content
        assert "from e" in content, (
            "fef_integration.py should use 'raise ValueError(...) from e' "
            "to preserve original exception traceback"
        )


class TestGlobalRegistriesSafeAccess:
    """Test that registry access is safe for concurrent tool startup."""

    def test_get_registry_for_tool_exists(self):
        """Verify get_registry_for_tool async helper exists."""
        from launcher.tool_extensions.registry import get_registry_for_tool
        import inspect
        assert asyncio.iscoroutinefunction(get_registry_for_tool), (
            "get_registry_for_tool should be an async function"
        )
        sig = inspect.signature(get_registry_for_tool)
        assert 'tool_name' in sig.parameters

    def test_registry_lock_exists(self):
        """Verify _registry_lock is defined."""
        from launcher.tool_extensions.registry import _registry_lock
        assert isinstance(_registry_lock, asyncio.Lock)


class TestPortLeakFix:
    """Test that failed server startups release their allocated ports."""

    def test_failed_startups_release_ports(self):
        """Regression: ports for failed tools must be released even when some succeed."""
        content = Path("launchmcp.py").read_text()

        assert "failed_startups" in content, (
            "launchmcp.py should track failed_startups to release their ports"
        )
        assert "release_port" in content, (
            "launchmcp.py should call port_manager.release_port for failed tools"
        )
        assert "for tool_name in failed_startups" in content, (
            "launchmcp.py should iterate over failed_startups to release ports"
        )


class TestWebmcpTransportSwitching:
    """Test that _fastmcp.py builds its app via get_transport_app."""

    def test_fastmcp_app_export_exists(self):
        """webmcp_fastmcp.py must export app built via get_transport_app."""
        content = Path("tools/webmcp/webmcp_fastmcp.py").read_text()
        assert "get_transport_app" in content


class TestTransportSwitchingAllTools:
    """Non-regression: every _fastmcp.py must build its app via get_transport_app."""

    TOOLS = ["simplemcp", "webmcp", "oraclemcp", "convertermcp", "ragmcp", "memorymcp"]

    @pytest.mark.parametrize("tool", TOOLS)
    def test_fastmcp_app_via_factory(self, tool):
        """Each _fastmcp.py must build its app via get_transport_app (multi default)."""
        fpath = Path(f"tools/{tool}/{tool}_fastmcp.py")
        if not fpath.exists():
            pytest.skip(f"{tool} has no _fastmcp.py")
        content = fpath.read_text()
        assert "get_transport_app" in content, f"{tool} must build its app via get_transport_app(mcp)"

    @pytest.mark.parametrize("tool", TOOLS)
    def test_streamable_http_default(self, tool):
        """Each _fastmcp.py must advertise the versatile multi-transport app."""
        fpath = Path(f"tools/{tool}/{tool}_fastmcp.py")
        if not fpath.exists():
            pytest.skip(f"{tool} has no _fastmcp.py")
        content = fpath.read_text()
        assert "multi-transport" in content, f"{tool} must log the multi-transport app"

    def test_legacy_files_removed(self):
        """No _sse.py or _streamable.py files should exist for tools with _fastmcp.py."""
        for tool in self.TOOLS:
            sse = Path(f"tools/{tool}/{tool}_sse.py")
            streamable = Path(f"tools/{tool}/{tool}_streamable.py")
            assert not sse.exists(), f"Legacy {sse} should be deleted (replaced by _fastmcp.py)"
            assert not streamable.exists(), f"Legacy {streamable} should be deleted (replaced by _fastmcp.py)"

    def test_launcher_has_transport_flag(self):
        """Launcher must accept --transport with both transports (sse = legacy compat)."""
        content = Path("launchmcp.py").read_text()
        assert "--transport" in content
        assert "streamable-http" in content
        assert '"sse"' in content, (
            "SSE was re-added 2026-08-30 as legacy compatibility for outdated "
            "harnesses; --transport must offer it again"
        )

    def test_launcher_sets_env_var(self):
        """Launcher must set MCP_TRANSPORT env var before importing tools."""
        content = Path("launchmcp.py").read_text()
        assert 'os.environ["MCP_TRANSPORT"]' in content

    def test_main_has_transport_flag(self):
        """launcher/__main__.py must accept --transport with both transports."""
        content = Path("launcher/__main__.py").read_text()
        assert "--transport" in content
        assert "MCP_TRANSPORT" in content
        assert '"sse"' in content, (
            "SSE was re-added 2026-08-30 as legacy compatibility; "
            "--transport must offer it again"
        )


class TestTransportDefaultUnpinned:
    """The versatile multi-transport default must be reachable via startlauncher.

    Regression 2026-08-30 (fix 7801773): DEFAULT_CONFIG carried
    ``"transport": "streamable-http"``, so launchmcp.py's resolution chain
    (args.transport > config > MCP_TRANSPORT > multi default) never reached
    its fallback — every tool came up single-transport and /sse 404'd for
    legacy clients. These tests pin the config layers above the tools, not
    just get_transport_app.
    """

    def test_default_config_has_no_transport_key(self):
        """DEFAULT_CONFIG must not pin a transport (regression 7801773)."""
        from launcher.launcher_config import Config

        assert "transport" not in Config.DEFAULT_CONFIG, (
            "DEFAULT_CONFIG must not carry a 'transport' key — it pins every "
            "tool and makes the multi-transport default unreachable via "
            "startlauncher"
        )

    def test_launcher_config_resolves_to_multi_default(self, monkeypatch):
        """The real Config() that launchmcp.main() builds (machine
        config/launcher_config.json merged over DEFAULT_CONFIG) must leave the
        launchmcp.py resolution chain empty -> multi default."""
        import os

        from launcher.launcher_config import Config

        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        config_path = str(
            Path(__file__).parent.parent / "config" / "launcher_config.json"
        )
        config = Config(config_path)

        # Exact chain from launchmcp.py (args.transport is None without --transport)
        resolved = None or config.config.get("transport") or os.environ.get("MCP_TRANSPORT")
        assert resolved is None, (
            f"transport resolved to {resolved!r} — a config layer pins it; "
            "expected no pin (multi-transport default)"
        )

    def test_resolution_chain_still_consults_config(self):
        """launchmcp.py's chain must still read the config layer — keeps
        test_launcher_config_resolves_to_multi_default honest if the chain
        moves elsewhere."""
        content = Path("launchmcp.py").read_text()
        assert 'config.config.get("transport")' in content, (
            "launchmcp.py resolution chain must consult config.config — the "
            "unpinned-default regression test covers the wrong layer otherwise"
        )


class TestConfigGodObjectComment:
    """Test that Config class has architectural debt comment."""

    def test_config_has_debt_comment(self):
        """Regression: Config class should document its god object status."""
        content = Path("launcher/launcher_config.py").read_text()

        assert "ARCHITECTURAL DEBT" in content, (
            "Config class should have comment documenting god object debt"
        )
        assert "god object" in content.lower(), (
            "Config class comment should mention 'god object'"
        )
        assert "getter method" in content.lower() or "get_*" in content, (
            "Config class comment should mention getter methods"
        )


class TestStartServersCleanup:
    """Test start_servers releases ports for all failure cases."""

    def test_port_release_on_partial_failure(self):
        """Ports must be released when some servers succeed and some fail."""
        content = Path("launchmcp.py").read_text()

        cleanup_section = content[content.find("if successful == 0"):content.find("return started_tools")]
        assert "allocated_tools" in cleanup_section, (
            "Port cleanup should use allocated_tools list"
        )

        assert "for tool_name in allocated_tools" in content or "for tool_name in failed_startups" in content, (
            "Port release should iterate over tools to clean up"
        )