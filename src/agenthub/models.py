from __future__ import annotations
"""Pydantic data models for AgentHub."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AgentCapability(str, Enum):
    """What an agent can do."""

    CODE_READ = "code_read"
    CODE_WRITE = "code_write"
    DB_QUERY = "db_query"
    DB_WRITE = "db_write"
    API_CALL = "api_call"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"


class RoutingConfig(BaseModel):
    """Per-agent routing preferences.

    Allows each agent to declare its own routing behavior — keyword weights,
    domain ownership, exclusions, priority, confidence thresholds, and
    fallback chains — instead of having these inferred externally.

    All fields default to permissive values so existing agents work unchanged.
    When a field is empty/zero, the routing system falls back to its existing
    heuristic logic.
    """

    keyword_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Keyword → weight mapping (1.0 = normal, 2.0 = high priority). "
        "Overrides builder heuristics when non-empty.",
    )
    domains: list[str] = Field(
        default_factory=list,
        description="Explicit domain tags (e.g., ['api', 'auth', 'backend']). "
        "Overrides inferred domain detection when non-empty.",
    )
    exclusions: list[str] = Field(
        default_factory=list,
        description="Domains/keywords this agent should never handle "
        "(e.g., ['frontend', 'css']). Queries matching these get penalized.",
    )
    priority: int = Field(
        default=0,
        description="Tiebreaker when scores are equal (higher = preferred).",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum normalized score [0.0-1.0] to accept a query. "
        "Below this threshold, the query is passed to fallback_agent_id.",
    )
    fallback_agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID to hand off to when confidence is below threshold.",
    )
    prefer_exact_match: bool = Field(
        default=False,
        description="Give higher bonus for whole-word keyword matches vs substring.",
    )


class AgentSpec(BaseModel):
    """Registration spec for an agent."""

    agent_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this agent does")
    capabilities: list[AgentCapability] = Field(default_factory=list)

    # Routing configuration (per-agent routing preferences)
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig,
        description="Per-agent routing preferences (weights, domains, exclusions, etc.)",
    )

    # Context management
    context_paths: list[str] = Field(
        default_factory=list,
        description="File/folder paths this agent knows about",
    )
    context_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that should route to this agent",
    )

    # Resource hints
    estimated_tokens: int = Field(default=2000, description="Typical token usage")
    max_context_size: int = Field(default=50000, description="Max context this agent uses")

    # Behavior
    system_prompt: str = Field(default="", description="Agent's system prompt")
    temperature: float = Field(default=0.7, ge=0, le=1)

    # Metadata for auto-generated agents
    metadata: dict[str, Any] = Field(default_factory=dict)

    # === Hierarchy fields (for sub-agent support) ===
    parent_agent_id: Optional[str] = Field(
        default=None,
        description="ID of parent agent (None for top-level Tier B agents)",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child sub-agents",
    )
    hierarchy_level: int = Field(
        default=0,
        description="0 = Tier B (team lead), 1+ = sub-agent depth",
    )
    is_team_lead: bool = Field(
        default=False,
        description="True when agent has sub-agents (children_ids non-empty)",
    )


class Message(BaseModel):
    """A single message in conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Conversation session with an agent."""

    session_id: str
    agent_id: str
    messages: list[Message] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Token tracking
    total_tokens_used: int = 0


class Artifact(BaseModel):
    """Structured output from agent."""

    artifact_type: Literal["code", "file", "sql", "json", "markdown"]
    content: str
    filename: Optional[str] = None
    language: Optional[str] = None
    description: str = ""


class AgentContextStatus(BaseModel):
    """Status of an agent's context awareness.

    Tracks whether an agent has seen recent changes in its domain.
    """

    agent_id: str = Field(..., description="Agent identifier")
    is_stale: bool = Field(
        default=False,
        description="True if files changed since last query",
    )
    changed_files: list[str] = Field(
        default_factory=list,
        description="List of changed file paths in agent's domain",
    )
    last_query_time: Optional[datetime] = Field(
        default=None,
        description="When this agent was last queried",
    )
    last_change_time: Optional[datetime] = Field(
        default=None,
        description="When files in agent's domain last changed",
    )
    status: Literal["fresh", "stale", "never_queried"] = Field(
        default="never_queried",
        description="Human-readable status",
    )


class TeamExecutionTrace(BaseModel):
    """Trace of a DAG team execution for observability."""

    dag_structure: dict[str, list[str]] = Field(
        default_factory=dict,
        description="agent_id -> list of dependency agent_ids",
    )
    execution_layers: list[list[str]] = Field(
        default_factory=list,
        description="Topological execution order - each layer runs in parallel",
    )
    sub_questions: dict[str, str] = Field(
        default_factory=dict,
        description="agent_id -> sub-question assigned to that agent",
    )
    agent_responses: dict[str, str] = Field(
        default_factory=dict,
        description="agent_id -> response content",
    )
    agent_tokens: dict[str, int] = Field(
        default_factory=dict,
        description="agent_id -> tokens used",
    )
    agent_times: dict[str, int] = Field(
        default_factory=dict,
        description="agent_id -> execution time in ms",
    )
    decomposition_tokens: int = Field(
        default=0,
        description="Tokens used for query decomposition",
    )
    synthesis_tokens: int = Field(
        default=0,
        description="Tokens used for response synthesis",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens across all operations",
    )
    total_time_ms: int = Field(
        default=0,
        description="Total wall-clock time in milliseconds",
    )
    parallel_speedup: float = Field(
        default=1.0,
        description="Speedup from parallel execution (sequential_time / actual_time)",
    )


class AgentResponse(BaseModel):
    """Standardized agent response."""

    content: str
    agent_id: str
    session_id: str
    tokens_used: int = 0
    artifacts: list[Artifact] = Field(default_factory=list)
    needs_followup: bool = False
    suggested_agent: Optional[str] = None  # For agent handoff
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., team_execution trace)",
    )


class SubAgentBoundary(BaseModel):
    """Proposed subdivision of a Tier B agent into focused sub-agents.

    Used by SubAgentManager to track how a large Tier B agent should be
    split into team members with more focused domains.
    """

    parent_agent_id: str = Field(
        ..., description="ID of the Tier B agent being subdivided"
    )
    sub_agent_id: str = Field(
        ..., description="Proposed ID for this sub-agent (e.g., 'backend_api')"
    )
    root_path: str = Field(
        ..., description="Root directory for this sub-agent's domain"
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns for files in this sub-agent's scope (e.g., 'api/**/*.py')",
    )
    estimated_context_kb: float = Field(
        default=0.0, description="Estimated context size in KB"
    )
    file_count: int = Field(default=0, description="Number of files in this sub-domain")
    role_description: str = Field(
        default="", description="Auto-generated description of this sub-agent's role"
    )
    key_modules: list[str] = Field(
        default_factory=list,
        description="Central files within this sub-domain (high in-degree)",
    )
    interfaces_with: list[str] = Field(
        default_factory=list,
        description="Other sub-agent IDs this sub-agent imports from",
    )
