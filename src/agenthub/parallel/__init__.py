from __future__ import annotations
"""Parallel Sessions module for AgentHub.

This module enables multiple Claude Code sessions to work simultaneously
on separate git branches when a user submits a multi-part request.

Key components:
- TaskDecomposer: Breaks requests into discrete implementation tasks
- ConflictRiskAnalyzer: Analyzes parallelization safety using ImportGraph + agents
- BranchOrchestrator: Manages git worktrees and spawns parallel sessions
- MergeCoordinator: Merges branches with domain-agent-assisted conflict resolution
- ParallelSessionManager: Top-level orchestrator

Example:
    >>> from agenthub.parallel import ParallelSessionManager, ParallelExecutionConfig
    >>> config = ParallelExecutionConfig(max_parallel_sessions=3)
    >>> manager = ParallelSessionManager(hub, project_root, config)
    >>> result = manager.execute("Add a save button and build a chart component")
"""

from agenthub.parallel.models import (
    AgentConflictAssessment,
    BoundaryCrossing,
    BoundaryCrossingResolution,
    ConflictType,
    CrossingResolutionType,
    DecompositionResult,
    DomainClaim,
    DomainResolutionProposal,
    FileOverlap,
    ImplementationTask,
    MergeConflict,
    MergeResult,
    OverlapType,
    ParallelExecutionConfig,
    ParallelExecutionResult,
    ParallelExecutionTrace,
    ParallelizationPlan,
    RiskLevel,
    SessionResult,
    SessionSpec,
)
from agenthub.parallel.orchestrator import BranchOrchestrator
from agenthub.parallel.teams_adapter import AgentTeamsAdapter, TeammateHandle, InboxMessage
from agenthub.parallel.merge import MergeCoordinator
from agenthub.parallel.escalation import (
    MidExecutionEscalationHandler,
    Escalation,
    EscalationType,
    EscalationResult,
)
from agenthub.parallel.manager import ParallelSessionManager
from agenthub.parallel.output import (
    EventType,
    ProgressEvent,
    ProgressReporter,
    format_progress_for_cli,
)
from agenthub.parallel.dashboard import (
    DashboardReporter,
    broadcast_parallel_event,
    create_dashboard_callback,
)

__all__ = [
    # Enums
    "RiskLevel",
    "OverlapType",
    "ConflictType",
    "CrossingResolutionType",
    # Task Decomposition
    "ImplementationTask",
    "DecompositionResult",
    "DomainClaim",
    # Conflict Analysis
    "FileOverlap",
    "AgentConflictAssessment",
    "ParallelizationPlan",
    # Session Execution
    "SessionSpec",
    "SessionResult",
    "BranchOrchestrator",
    # Agent Teams
    "AgentTeamsAdapter",
    "TeammateHandle",
    "InboxMessage",
    # Merge
    "MergeConflict",
    "DomainResolutionProposal",
    "MergeResult",
    "MergeCoordinator",
    # Boundary Crossing
    "BoundaryCrossing",
    "BoundaryCrossingResolution",
    # Escalation
    "MidExecutionEscalationHandler",
    "Escalation",
    "EscalationType",
    "EscalationResult",
    # Config and Results
    "ParallelExecutionConfig",
    "ParallelExecutionTrace",
    "ParallelExecutionResult",
    # Top-level Manager
    "ParallelSessionManager",
    # Progress Reporting
    "EventType",
    "ProgressEvent",
    "ProgressReporter",
    "format_progress_for_cli",
    # Dashboard Integration
    "DashboardReporter",
    "broadcast_parallel_event",
    "create_dashboard_callback",
]
