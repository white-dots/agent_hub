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


class AgentSpec(BaseModel):
    """Registration spec for an agent."""

    agent_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this agent does")
    capabilities: list[AgentCapability] = Field(default_factory=list)

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
