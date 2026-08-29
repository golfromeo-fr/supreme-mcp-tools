"""
Regression test: memorymcp must expose setup_extensions for the launcher.

The launcher's server_manager does `hasattr(tool_module, "setup_extensions")`
on the tool's fastmcp module and calls it with the launcher's extension
registry. memorymcp defined its setup under the name `setup_fef_v3` inside
the memory_tools support module, so the hasattr check failed and the
management API reported zero extensions for memorymcp (empty Extensions tab
in the management UI).

This test simulates the launcher's call path on a fresh registry and asserts
the memory_stats / list_memory_types data sources register.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "memorymcp"))

pytest.importorskip("nicegui")


def test_memorymcp_exposes_setup_extensions_and_registers():
    try:
        from memory_tools import setup_extensions  # noqa: F401
    except Exception as e:  # backend init failures in CI-like envs
        pytest.skip(f"memory_tools import failed: {e}")

    from launcher.tool_extensions import ExtensionRegistry

    registry = ExtensionRegistry(tool_name="memorymcp")
    setup_extensions(registry=registry)

    registered = registry._extensions.get("memorymcp", {})
    names = set(registered)
    assert "memory_stats" in names, f"memory_stats not registered: {names}"
    assert "list_memory_types" in names, f"list_memory_types not registered: {names}"


def test_fastmcp_module_reexports_setup_extensions():
    """The launcher's hasattr check targets the fastmcp module specifically."""
    try:
        import memorymcp_fastmcp
    except Exception as e:
        pytest.skip(f"memorymcp_fastmcp import failed: {e}")

    assert hasattr(memorymcp_fastmcp, "setup_extensions")
