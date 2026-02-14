from __future__ import annotations
"""Data models for QC (Quality Control) analysis.

This module defines the core data structures for:
- File changes and change sets
- Concerns raised by Tier B agents
- Analysis results from individual agents
- QC reports synthesized by the Tier C agent
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ConcernSeverity(str, Enum):
    """Severity levels for concerns."""

    CRITICAL = "critical"  # Must fix before merge
    HIGH = "high"  # Should fix before merge
    MEDIUM = "medium"  # Should address soon
    LOW = "low"  # Nice to have / minor improvement
    INFO = "info"  # Informational only


class ConcernCategory(str, Enum):
    """Categories of concerns."""

    BREAKING_CHANGE = "breaking_change"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MISSING_TESTS = "missing_tests"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    DEPRECATION = "deprecation"
    TYPE_SAFETY = "type_safety"
    ERROR_HANDLING = "error_handling"
    COMPATIBILITY = "compatibility"
    ARCHITECTURE = "architecture"
    OTHER = "other"


class FileChange(BaseModel):
    """Represents a single file change."""

    path: str
    change_type: str  # "created" | "modified" | "deleted"
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None


class ChangeSet(BaseModel):
    """A set of file changes to analyze."""

    change_id: str = Field(..., description="Unique ID for this change set")
    timestamp: datetime = Field(default_factory=datetime.now)
    files: list[FileChange] = Field(default_factory=list)
    commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    source: str = "file_watcher"  # "file_watcher" | "git" | "manual"


class Concern(BaseModel):
    """A single concern raised by a Tier B agent."""

    concern_id: str = Field(..., description="Unique concern ID")
    agent_id: str = Field(..., description="Agent that raised this concern")
    domain: str = Field(..., description="Domain/module affected")
    category: ConcernCategory
    severity: ConcernSeverity
    title: str = Field(..., description="Short description")
    description: str = Field(..., description="Detailed explanation")
    affected_files: list[str] = Field(default_factory=list)
    affected_functions: list[str] = Field(default_factory=list)
    suggestion: Optional[str] = None  # How to fix
    code_snippet: Optional[str] = None  # Relevant code
    references: list[str] = Field(default_factory=list)  # Related docs/files
    confidence: float = Field(default=0.8, ge=0, le=1)  # Agent's confidence
    metadata: dict[str, Any] = Field(default_factory=dict)
    raised_at: datetime = Field(default_factory=datetime.now)


class AgentAnalysisResult(BaseModel):
    """Result from a single Tier B agent's analysis."""

    agent_id: str
    domain: str
    analyzed_files: list[str]
    concerns: list[Concern]
    analysis_time_ms: int
    tokens_used: int
    skipped_reason: Optional[str] = None  # If analysis was skipped


class ActionItem(BaseModel):
    """A suggested action item from the QC Agent."""

    action_id: str
    priority: int = Field(ge=1, le=5, description="1=highest priority")
    title: str
    description: str
    related_concerns: list[str]  # Concern IDs
    assignee_hint: Optional[str] = None  # Suggested domain/agent
    estimated_effort: Optional[str] = None  # "small" | "medium" | "large"


class ConcernReport(BaseModel):
    """Synthesized report from QC Agent."""

    report_id: str
    change_set_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Summary counts
    total_concerns: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

    # Concerns by category
    concerns_by_category: dict[str, list[Concern]] = Field(default_factory=dict)

    # All concerns (sorted by severity)
    all_concerns: list[Concern] = Field(default_factory=list)

    # Action items
    action_items: list[ActionItem] = Field(default_factory=list)

    # Analysis metadata
    agents_consulted: list[str] = Field(default_factory=list)
    analysis_results: list[AgentAnalysisResult] = Field(default_factory=list)
    total_analysis_time_ms: int = 0
    total_tokens_used: int = 0

    # QC Agent's overall assessment
    overall_assessment: str = ""
    risk_level: str = "low"  # "low" | "medium" | "high" | "critical"
    recommendation: str = "approve"  # "approve" | "review" | "block"
