"""
Data Models for the Management UI.

Pydantic models defining the data structures used throughout the UI.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Any
from enum import Enum


class ToolStatus(str, Enum):
    """Tool status enum."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"
    # Health-based statuses from ServiceRegistry
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

    def is_active(self) -> bool:
        """Check if tool is in an active state."""
        return self in (ToolStatus.RUNNING, ToolStatus.HEALTHY, ToolStatus.DEGRADED)


class ExtensionType(str, Enum):
    """Extension type enum (mirrors launcher/tool_extensions/registry.py)."""
    DATA_SOURCE = "data_source"
    MUTATOR = "mutator"
    ACTION = "action"
    EVENT = "event"
    STREAM = "stream"


class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    status: ToolStatus
    management_url: str | None = None
    mcp_port: int | None = None
    capabilities: dict[str, Any] | None = None
    last_check: float | None = None  # Timestamp as float


class ExtensionSchema(BaseModel):
    """Schema definition for an extension."""
    type: ExtensionType
    name: str
    description: str | None = None
    json_schema: dict[str, Any] = {}  # JSON Schema for input/output


class Extension(BaseModel):
    """An extension with its schema and current state."""
    name: str
    type: ExtensionType
    # Use alias to map 'schema' from API to 'json_schema' in model
    json_schema: dict[str, Any] = Field(default_factory=dict, validation_alias='schema')
    # Map metadata.description to description field
    description: str | None = None
    data: dict[str, Any] | None = None  # For data sources
    # Store full metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def extract_metadata_fields(cls, data):
        """Extract description and other fields from metadata if not provided directly."""
        if isinstance(data, dict):
            # If description not provided at root, extract from metadata
            if 'description' not in data or data.get('description') is None:
                metadata = data.get('metadata', {})
                if isinstance(metadata, dict) and 'description' in metadata:
                    data['description'] = metadata['description']
        return data


class ToolDetail(BaseModel):
    """Detailed tool information with extensions."""
    name: str
    status: ToolStatus
    management_url: str | None = None
    mcp_port: int | None = None
    capabilities: dict[str, Any] | None = None
    last_check: float | None = None  # Timestamp as float
    registered_at: float | None = None  # Timestamp as float
    extensions: list[Extension] = []

    @property
    def is_active(self) -> bool:
        """Check if tool is in an active state."""
        return self.status.is_active()

    @property
    def extension_count(self) -> int:
        """Total number of extensions."""
        return len(self.extensions)

    def get_extension_summary(self) -> dict[str, int]:
        """Get summary of extension counts by type."""
        counts = {"data_sources": 0, "mutators": 0, "actions": 0, "events": 0, "streams": 0}
        for ext in self.extensions:
            if ext.type == ExtensionType.DATA_SOURCE:
                counts["data_sources"] += 1
            elif ext.type == ExtensionType.MUTATOR:
                counts["mutators"] += 1
            elif ext.type == ExtensionType.ACTION:
                counts["actions"] += 1
            elif ext.type == ExtensionType.EVENT:
                counts["events"] += 1
            elif ext.type == ExtensionType.STREAM:
                counts["streams"] += 1
        return counts


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool
    data: Any | None = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any = None) -> "APIResponse":
        """Create a successful response."""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "APIResponse":
        """Create a failed response."""
        return cls(success=False, error=error)


class EnvVariable(BaseModel):
    """An environment variable for a tool."""
    name: str
    description: str = ""
    required: bool = False
    secret: bool = True
    value_masked: str = ""
    value_raw: str = ""
    is_set: bool = False
    default: str = ""
    options: list[str] = []
    type: str = "string"  # string, integer, number, boolean
    minimum: float | None = None
    maximum: float | None = None
