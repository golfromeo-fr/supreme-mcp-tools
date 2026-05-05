"""
Shared UI constants for the MCP management UI.

Defines color mappings, status labels, and other constants used across
multiple components to avoid duplication.
"""

from ..models import ToolStatus

STATUS_COLORS: dict[ToolStatus, str] = {
    ToolStatus.RUNNING: "green",
    ToolStatus.HEALTHY: "green",
    ToolStatus.STOPPED: "grey",
    ToolStatus.ERROR: "red",
    ToolStatus.UNHEALTHY: "red",
    ToolStatus.DEGRADED: "yellow",
    ToolStatus.UNKNOWN: "orange",
}