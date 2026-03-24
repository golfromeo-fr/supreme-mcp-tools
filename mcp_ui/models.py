"""
Data Models for the Management UI.

Pydantic models defining the data structures used throughout the UI.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional
from datetime import datetime
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


class ExtensionType(str, Enum):
    """Extension type enum."""
    DATA_SOURCE = "data_source"
    MUTATOR = "mutator"
    ACTION = "action"


class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    status: ToolStatus
    management_url: Optional[str] = None
    mcp_port: Optional[int] = None
    capabilities: Optional[Dict[str, Any]] = None  # Can be dict with extensions_count, etc.
    last_check: Optional[float] = None  # Timestamp as float


class ExtensionSchema(BaseModel):
    """Schema definition for an extension."""
    type: ExtensionType
    name: str
    description: Optional[str] = None
    json_schema: Dict[str, Any] = {}  # JSON Schema for input/output


class Extension(BaseModel):
    """An extension with its schema and current state."""
    name: str
    type: ExtensionType
    # Use alias to map 'schema' from API to 'json_schema' in model
    json_schema: Dict[str, Any] = Field(default_factory=dict, validation_alias='schema')
    # Map metadata.description to description field
    description: Optional[str] = None
    data: Optional[Dict[str, Any]] = None  # For data sources
    
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
    management_url: Optional[str] = None
    mcp_port: Optional[int] = None
    capabilities: Optional[Dict[str, Any]] = None  # Can be dict with extensions_count, etc.
    last_check: Optional[float] = None  # Timestamp as float
    registered_at: Optional[float] = None  # Timestamp as float
    extensions: List[Extension] = []


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class UIState(BaseModel):
    """Global UI state (per-client, managed by NiceGUI)."""
    selected_tool: Optional[str] = None
    is_loading: bool = False
    connection_status: str = "connected"  # connected, disconnected, error
    last_refresh: Optional[datetime] = None
    error_message: Optional[str] = None


class LoginForm(BaseModel):
    """Login form data."""
    username: str = ""
    password: str = ""


class MutatorForm(BaseModel):
    """Dynamic mutator form data."""
    extension_name: str
    values: Dict[str, Any] = {}
