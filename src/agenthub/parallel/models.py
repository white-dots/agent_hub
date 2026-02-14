from __future__ import annotations
"""Data models for Parallel Sessions.

This module defines all the dataclasses used by the parallel execution system:
- Task decomposition models
- Conflict analysis models
- Session execution models
- Merge and resolution models
- Configuration and result models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


# =============================================================================
# Enums
# =============================================================================


class RiskLevel(str, Enum):
    """Risk level for parallelization.

    Risk can only be upgraded by agents, never downgraded from static analysis.
    """

    NONE = "none"  # SAFE - auto-proceed
    LOW = "low"  # CAUTION - auto-proceed with monitoring
    MEDIUM = "medium"  # Ask CEO, PM recommends parallel with caution
    HIGH = "high"  # Ask CEO, PM recommends sequential + provides order


class OverlapType(str, Enum):
    """Type of file overlap detected between tasks."""

    DIRECT = "direct"  # Same file modified by both tasks
    SHARED_IMPORT = "shared_import"  # Transitive import overlap (depth=2)
    SHARED_TYPE = "shared_type"  # Both import same hub module (high in-degree)
    SHARED_CONFIG = "shared_config"  # Shared config/env files


class ConflictType(str, Enum):
    """Type of merge conflict detected."""

    TEXTUAL = "textual"  # Git merge conflict (same lines modified)
    SEMANTIC = "semantic"  # Tests fail but no textual conflict
    SCOPE_VIOLATION = "scope_violation"  # Session modified files outside scope


class CrossingResolutionType(str, Enum):
    """How a boundary crossing was resolved."""

    APPROVED_AS_IS = "approved_as_is"  # Owning agent approved the change
    APPROVED_WITH_MODIFICATION = "approved_with_modification"  # Approved with edits
    DEFERRED_TO_MERGE = "deferred_to_merge"  # Will handle during merge phase
    ESCALATED_TO_CEO = "escalated_to_ceo"  # Requires human decision
    REJECTED = "rejected"  # Owning agent rejected the change


# =============================================================================
# Task Decomposition Models
# =============================================================================


@dataclass
class ImplementationTask:
    """A discrete unit of work that produces code changes.

    Represents a single task extracted from a multi-part user request.
    Each task can be assigned to a parallel session.
    """

    task_id: str
    description: str
    estimated_files: list[str] = field(default_factory=list)
    """Files expected to be modified (from domain survey)."""

    estimated_new_files: list[str] = field(default_factory=list)
    """New files expected to be created."""

    domain_agents: list[str] = field(default_factory=list)
    """Agent IDs that claimed this task touches their domain."""

    complexity: Literal["trivial", "moderate", "complex"] = "moderate"
    """Estimated complexity for token budgeting."""

    estimated_tokens: int = 0
    """Estimated tokens needed to complete this task."""

    depends_on: list[str] = field(default_factory=list)
    """Task IDs this task depends on (for sequential ordering)."""


@dataclass
class DomainClaim:
    """A domain agent's claim that a request touches its domain.

    Generated during the domain survey phase of task decomposition.
    """

    agent_id: str
    agent_name: str
    is_involved: bool
    """True if the agent believes the request touches its domain."""

    description: str
    """What the agent needs to do for this request."""

    estimated_files: list[str] = field(default_factory=list)
    """Files the agent expects to touch."""

    confidence: float = 0.5
    """How sure the agent is that this request touches its domain (0-1).

    Threshold: claims with confidence < 0.3 are filtered out in _survey_domains().
    NOT used for risk decisions - only for filtering survey results.
    """


@dataclass
class DecompositionResult:
    """Result of decomposing a user request into tasks.

    Produced by TaskDecomposer.decompose().
    """

    tasks: list[ImplementationTask]
    original_request: str
    appears_simple: bool
    """True if the request looks simple on the surface."""

    actual_complexity: Literal["single", "multi_independent", "multi_dependent", "multi_mixed"]
    """Actual complexity after domain survey.

    - single: One task, no parallelization needed
    - multi_independent: Multiple tasks with no dependencies
    - multi_dependent: Multiple tasks with dependencies (sequential)
    - multi_mixed: Some parallel, some sequential
    """

    decomposition_reasoning: str
    """Explanation of how the request was broken down."""

    ceo_briefing: Optional[str] = None
    """Message for CEO when hidden complexity detected (appears_simple but complex)."""

    tokens_used: int = 0
    """Tokens used for decomposition."""


# =============================================================================
# Conflict Analysis Models
# =============================================================================


@dataclass
class FileOverlap:
    """Detected overlap between tasks for a specific file."""

    file_path: str
    tasks_touching: list[str]
    """Task IDs that may modify this file."""

    overlap_type: OverlapType
    risk_level: RiskLevel


@dataclass
class AgentConflictAssessment:
    """A domain agent's assessment of conflict between two tasks.

    Agents can only UPGRADE risk, never downgrade from static analysis.
    """

    agent_id: str
    agent_name: str
    task_pair: tuple[str, str]
    """The two task IDs being assessed."""

    has_concern: bool
    """True if the agent believes there's a conflict."""

    concern_description: str
    """Description of the concern."""

    severity: RiskLevel
    """Agent's assessment of risk level."""

    affected_files: list[str] = field(default_factory=list)
    """Files where the conflict may occur."""

    tokens_used: int = 0


@dataclass
class ParallelizationPlan:
    """Plan for executing tasks, either in parallel or sequential.

    The PM never silently falls back to sequential. Even at HIGH risk,
    the CEO gets a shot-call with the PM's recommendation.
    """

    parallel_groups: list[list[str]]
    """Groups of task_ids safe to run together in parallel."""

    sequential_order: list[str]
    """Recommended order for sequential execution (when HIGH risk)."""

    file_overlaps: list[FileOverlap] = field(default_factory=list)
    """All detected file overlaps."""

    agent_assessments: list[AgentConflictAssessment] = field(default_factory=list)
    """All agent conflict assessments."""

    overall_risk: RiskLevel = RiskLevel.NONE
    """Maximum risk across all analyses."""

    confidence: float = 0.9
    """PM's overall confidence in the plan's safety (0-1).

    NOT used for auto-fallback (no silent fallbacks).
    Displayed to CEO as context for their shot-call.
    """

    reasoning: str = ""
    """Explanation of the risk assessment."""

    pm_recommendation: Literal["parallel", "sequential"] = "parallel"
    """PM's recommendation based on risk analysis."""

    estimated_speedup: float = 1.0
    """Estimated speedup from parallel execution."""

    estimated_total_tokens: int = 0
    """Estimated total tokens across all sessions."""


# =============================================================================
# Session Execution Models
# =============================================================================


@dataclass
class SessionSpec:
    """Specification for a Claude Code session.

    Defines what a parallel session should do and its scope constraints.
    """

    task: ImplementationTask
    branch_name: str
    worktree_path: str
    """Git worktree path (NOT checkout - each session gets isolated filesystem)."""

    scoped_files: list[str] = field(default_factory=list)
    """Files this session is allowed to modify."""

    scoped_dirs: list[str] = field(default_factory=list)
    """Directories this session is allowed to work in."""

    context_from_agents: list[str] = field(default_factory=list)
    """Agent IDs whose context should be injected."""

    prompt: str = ""
    """The scoped prompt for the Claude Code session."""

    timeout_seconds: int = 300


@dataclass
class SessionResult:
    """Result from a Claude Code session."""

    task_id: str
    branch_name: str
    success: bool
    files_changed: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    stdout: str = ""
    tokens_used: int = 0
    time_seconds: float = 0.0
    test_results: Optional[dict] = None
    error: Optional[str] = None
    boundary_crossings: list[str] = field(default_factory=list)
    """Parsed [BOUNDARY_CROSSING: ...] tags from session output."""

    execution_backend: Literal["cli", "agent_teams"] = "cli"
    """Which backend executed this session."""


# =============================================================================
# Merge and Resolution Models
# =============================================================================


@dataclass
class MergeConflict:
    """A detected merge conflict between branches."""

    file_path: str
    conflict_type: ConflictType
    description: str
    branch_a: str
    branch_b: str
    diff_a: str = ""
    """Changes from branch A."""

    diff_b: str = ""
    """Changes from branch B."""

    owning_agent: Optional[str] = None
    """Agent ID that owns this file (most specific)."""

    auto_resolvable: bool = False
    """True if the conflict can be resolved automatically."""

    suggested_resolution: Optional[str] = None
    """Suggested resolution if auto_resolvable."""


@dataclass
class DomainResolutionProposal:
    """A domain agent's proposed resolution for a merge conflict.

    The agent that owns the conflicting file proposes how to resolve it.
    """

    agent_id: str
    agent_name: str
    conflict_file: str
    proposed_resolution: str
    """The proposed merged content."""

    reasoning: str
    """Why this resolution is correct."""

    confidence: float = 0.8
    """How confident the agent is in THIS resolution (0-1).

    < 0.6 triggers CEO escalation via _should_escalate_to_ceo().
    Different from DomainClaim.confidence (domain relevance).
    """

    side_effects: list[str] = field(default_factory=list)
    """Other files that may need updates due to this resolution."""

    needs_ceo: bool = False
    """True if the agent explicitly requests CEO involvement."""


@dataclass
class MergeResult:
    """Result of merging parallel sessions."""

    success: bool
    merged_branch: str
    conflicts: list[MergeConflict] = field(default_factory=list)
    resolutions: list[DomainResolutionProposal] = field(default_factory=list)
    files_merged: list[str] = field(default_factory=list)
    test_results: Optional[dict] = None
    needs_user_input: bool = False
    """True if there are unresolved conflicts requiring CEO decision."""

    escalation_reason: Optional[str] = None
    """Why CEO input is needed (if needs_user_input is True)."""

    summary: str = ""
    """Human-readable summary of the merge."""


# =============================================================================
# Boundary Crossing Models
# =============================================================================


@dataclass
class BoundaryCrossing:
    """Detected when a session needs files outside its scope.

    Can be detected either post-execution (CLI mode) or in real-time
    (Agent Teams mode via inbox monitoring).
    """

    session_task_id: str
    requesting_agent: str
    target_file: str
    owning_agent: Optional[str] = None
    """Agent that owns the target file (if identified)."""

    reason: str = ""
    """Why the session needs this file."""

    proposed_change: Optional[str] = None
    """What change the session wants to make."""

    blocking: bool = False
    """True if the crossing blocks task completion."""

    detected_at: Literal["post_execution", "real_time"] = "post_execution"
    """Which backend caught it."""


@dataclass
class BoundaryCrossingResolution:
    """Resolution for a boundary crossing.

    The owning agent decides whether to approve, modify, defer, or reject.
    """

    crossing: BoundaryCrossing
    approved: bool
    resolution_type: CrossingResolutionType
    modified_change: Optional[str] = None
    """Modified version of the proposed change (if APPROVED_WITH_MODIFICATION)."""

    reasoning: str = ""
    confidence: float = 0.8
    """How confident the owning agent is in this resolution (0-1).

    < 0.6 triggers escalation to CEO (same threshold as merge resolution).
    """


# =============================================================================
# Configuration and Result Models
# =============================================================================


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel execution."""

    max_parallel_sessions: int = 3
    """Maximum sessions to run in parallel."""

    session_timeout_seconds: int = 300
    """Timeout for each session in seconds."""

    claude_model: str = "claude-sonnet-4-20250514"
    """Model to use for Claude Code sessions."""

    auto_resolve_conflicts: bool = True
    """Whether to attempt automatic conflict resolution."""

    run_tests_after_merge: bool = True
    """Whether to run tests after merging."""

    test_command: Optional[str] = None
    """Test command to run. Auto-detects if None."""

    auto_proceed_threshold: Literal["safe", "caution"] = "caution"
    """Auto-proceed up to this risk level; above requires CEO shot-call.

    - "safe": Only auto-proceed on NONE risk
    - "caution": Auto-proceed on NONE or LOW risk
    """

    token_budget: int = 100000
    """Maximum total tokens across all sessions."""

    execution_backend: Literal["cli", "agent_teams"] = "cli"
    """Execution backend to use.

    - "cli": claude --print (stable, non-interactive)
    - "agent_teams": Claude Code Agent Teams (real-time inter-agent messaging)

    Falls back to "cli" if Agent Teams is unavailable.
    """


@dataclass
class ParallelExecutionTrace:
    """Comprehensive trace for debugging and benchmarking."""

    decomposition_time_ms: int = 0
    analysis_time_ms: int = 0
    session_times: dict = field(default_factory=dict)
    """task_id -> execution time in ms."""

    merge_time_ms: int = 0
    total_time_ms: int = 0

    decomposition_tokens: int = 0
    analysis_tokens: int = 0
    session_tokens: dict = field(default_factory=dict)
    """task_id -> tokens used."""

    merge_tokens: int = 0
    total_tokens: int = 0

    parallel_groups: list[list[str]] = field(default_factory=list)
    """Actual parallel groupings used."""

    conflicts_found: int = 0
    conflicts_auto_resolved: int = 0
    scope_violations: int = 0
    test_pass_rate: Optional[float] = None

    error_message: Optional[str] = None
    """Error message if execution failed."""

    error_traceback: Optional[str] = None
    """Full traceback if execution failed."""


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution."""

    success: bool
    tasks: list[ImplementationTask] = field(default_factory=list)
    plan: Optional[ParallelizationPlan] = None
    session_results: list[SessionResult] = field(default_factory=list)
    merge_result: Optional[MergeResult] = None

    total_time_seconds: float = 0.0
    sequential_estimate_seconds: float = 0.0
    speedup: float = 1.0
    """Actual speedup achieved (sequential_estimate / actual_time)."""

    total_tokens: int = 0
    trace: Optional[ParallelExecutionTrace] = None
    error_message: Optional[str] = None
    """Error message if execution failed."""
