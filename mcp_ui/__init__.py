"""
NiceGUI Management UI for MCP Tools.

A web-based management interface for the MCP tools system.
"""

__version__ = "0.1.0"

# Lazy import to avoid registering middleware when importing other modules
# This allows other modules in the package to add their own middleware without conflicts
def __getattr__(name):
    if name == "app":
        from .management_ui import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["app"]
