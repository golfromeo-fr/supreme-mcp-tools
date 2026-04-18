"""
Memory MCP Models - Pydantic models for memory operations.
"""

from typing import Any
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Memory type classification for organizing different kinds of memories."""
    CODE_PATTERN = "code_pattern"
    ARCHITECTURAL_DECISION = "architectural_decision"
    TRICK = "trick"
    PLAN = "plan"
    LESSON = "lesson"
    CONCEPT = "concept"
    IDEA = "idea"
    DECISION = "decision"


class RetentionPolicy(str, Enum):
    """Memory retention policies."""
    PERMANENT = "permanent"
    TEMP = "temp"
    AUTO_DELETE = "auto-delete"


class Sensitivity(str, Enum):
    """PII sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MemoryItem(BaseModel):
    """Complete memory item with all metadata."""
    id: str | None = None
    text: str = Field(..., description="The core memory content")
    embedding: list[float] | None = None
    type: MemoryType = MemoryType.CONCEPT
    source: str = "agent_action"
    path: str | None = None
    commit: str | None = None
    file_range: str | None = None
    agent_id: str | None = None
    timestamp: str | None = None
    tags: list[str] = Field(default_factory=list)
    raw_object_key: str | None = None
    retention_policy: RetentionPolicy = RetentionPolicy.AUTO_DELETE


class MemoryHit(BaseModel):
    """A retrieved memory with scoring info."""
    id: str
    text: str
    type: MemoryType
    source: str
    tags: list[str]
    score: float
    recency_score: float = 0.0
    usage_boost: float = 0.0
    created_at: str | None = None
    last_accessed: str | None = None
    usage_count: int = 0
    provenance: dict[str, Any] | None = None


class FilterSpec(BaseModel):
    """Filter specification for memory queries."""
    memory_type: MemoryType | None = None
    agent_id: str | None = None
    tags: list[str] | None = None
    source: str | None = None
    path: str | None = None
    min_usage_count: int | None = None
    created_after: str | None = None
    created_before: str | None = None
    sensitivity: Sensitivity | None = None


class ProvenanceRecord(BaseModel):
    """Provenance tracking for memory sources."""
    source: str
    model_version: str | None = None
    confidence: float | None = None
    timestamp: str
    notes: str | None = None


class AgentAction(BaseModel):
    """Agent action for automatic memory capture."""
    action_type: str  # file_open, file_edit, test_run, commit, question
    context: str  # Freeform context to remember
    path: str | None = None
    commit: str | None = None
    agent_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class MemoryMetrics(BaseModel):
    """Memory system metrics."""
    total_memories: int = 0
    by_type: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    avg_latency_ms: float = 0.0
    storage_used_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class DecayPolicy(BaseModel):
    """Policy for memory decay/cleanup."""
    ttl_days: int = 30
    min_usage_count: int = 0
    delete_sensitivity_high: bool = False
    archive_before_delete: bool = True