"""
Plugins Module for FEF V3

Provides dynamic plugin loading for extensions.
"""

from .loader import PluginLoader

__all__ = [
    "PluginLoader",
]
