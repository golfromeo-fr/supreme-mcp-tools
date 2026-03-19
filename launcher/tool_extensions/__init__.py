"""
Tool Extensions Package

Provides the Flexible Extensibility Framework V3 components for MCP tools.
"""

from .registry import Extension, ExtensionRegistry, ExtensionType
from .http_server import ExtensionHTTPServer

__all__ = [
    "Extension",
    "ExtensionRegistry",
    "ExtensionType",
    "ExtensionHTTPServer",
]
