from __future__ import annotations
"""Progress reporting for parallel session execution.

The ProgressReporter provides real-time updates during parallel execution,
suitable for streaming to CLI, dashboard, or other consumers.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from agenthub.parallel.models import (
    BoundaryCrossing,
    DecompositionResult,
    DomainResolutionProposal,
    ImplementationTask,
    MergeConflict,
    ParallelExecutionResult,
    ParallelizationPlan,
    SessionResult,
)


class EventType(str, Enum):
    """Types of progress events."""

    # Lifecycle events
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"

    # Phase events
    DECOMPOSITION_STARTED = "decomposition_started"
    DECOMPOSITION_COMPLETE = "decomposition_complete"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETE = "analysis_complete"
    CEO_CONFIRMATION_NEEDED = "ceo_confirmation_needed"
    CEO_DECISION_RECEIVED = "ceo_decision_received"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETE = "execution_complete"
    MERGE_STARTED = "merge_started"
    MERGE_COMPLETE = "merge_complete"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETE = "verification_complete"

    # Session events
    SESSION_STARTED = "session_started"
    SESSION_PROGRESS = "session_progress"
    SESSION_COMPLETE = "session_complete"
    SESSION_FAILED = "session_failed"

    # Conflict events
    BOUNDARY_CROSSING = "boundary_crossing"
    CONFLICT_DETECTED = "conflict_detected"
    RESOLUTION_PROPOSED = "resolution_proposed"
    RESOLUTION_APPLIED = "resolution_applied"
    ESCALATION_NEEDED = "escalation_needed"

    # Test events
    TESTS_STARTED = "tests_started"
    TESTS_PASSED = "tests_passed"
    TESTS_FAILED = "tests_failed"


@dataclass
class ProgressEvent:
    """A progress event during parallel execution."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
        }


class ProgressReporter:
    """Reports progress during parallel session execution.

    Supports multiple output modes:
    - Callback: Custom function called for each event
    - Collect: Stores all events for later retrieval
    - Silent: No output (for testing)

    Example:
        >>> def on_event(event: ProgressEvent):
        ...     print(f"[{event.event_type.value}] {event.message}")
        >>>
        >>> reporter = ProgressReporter(on_event=on_event)
        >>> reporter.report_decomposition(decomposition)
        [decomposition_complete] Decomposed into 3 tasks
    """

    def __init__(
        self,
        on_event: Optional[Callable[[ProgressEvent], None]] = None,
        collect_events: bool = False,
    ):
        """Initialize ProgressReporter.

        Args:
            on_event: Optional callback for each event.
            collect_events: If True, store all events internally.
        """
        self._callback = on_event
        self._collect = collect_events
        self._events: list[ProgressEvent] = []
        self._start_time: float = 0

    def _emit(self, event: ProgressEvent) -> None:
        """Emit an event to callback and/or storage."""
        if self._collect:
            self._events.append(event)
        if self._callback:
            self._callback(event)

    def get_events(self) -> list[ProgressEvent]:
        """Get all collected events."""
        return list(self._events)

    def clear_events(self) -> None:
        """Clear collected events."""
        self._events.clear()

    # =========================================================================
    # Lifecycle Events
    # =========================================================================

    def report_started(self, request: str) -> None:
        """Report parallel execution started.

        Args:
            request: The original user request.
        """
        self._start_time = time.time()
        self._emit(ProgressEvent(
            event_type=EventType.STARTED,
            message=f"Starting parallel execution: {request[:100]}{'...' if len(request) > 100 else ''}",
            details={"request": request},
        ))

    def report_completed(self, result: ParallelExecutionResult) -> None:
        """Report parallel execution completed.

        Args:
            result: The execution result.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._emit(ProgressEvent(
            event_type=EventType.COMPLETED,
            message=(
                f"Parallel execution {'succeeded' if result.success else 'failed'} "
                f"in {elapsed:.1f}s ({result.speedup:.1f}x speedup)"
            ),
            details={
                "success": result.success,
                "tasks": len(result.tasks),
                "speedup": result.speedup,
                "total_tokens": result.total_tokens,
                "elapsed_seconds": elapsed,
            },
        ))

    def report_failed(self, error: str) -> None:
        """Report parallel execution failed.

        Args:
            error: Error message.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._emit(ProgressEvent(
            event_type=EventType.FAILED,
            message=f"Parallel execution failed: {error}",
            details={"error": error, "elapsed_seconds": elapsed},
        ))

    # =========================================================================
    # Phase Events
    # =========================================================================

    def report_decomposition_started(self, request: str) -> None:
        """Report decomposition phase started."""
        self._emit(ProgressEvent(
            event_type=EventType.DECOMPOSITION_STARTED,
            message="Decomposing request into tasks...",
            details={"request": request},
        ))

    def report_decomposition(self, result: DecompositionResult) -> None:
        """Report decomposition phase complete.

        Args:
            result: Decomposition result.
        """
        task_summaries = [
            {"task_id": t.task_id, "description": t.description[:50]}
            for t in result.tasks
        ]

        self._emit(ProgressEvent(
            event_type=EventType.DECOMPOSITION_COMPLETE,
            message=f"Decomposed into {len(result.tasks)} tasks",
            details={
                "task_count": len(result.tasks),
                "tasks": task_summaries,
                "appears_simple": result.appears_simple,
                "actual_complexity": result.actual_complexity,
                "tokens_used": result.tokens_used,
            },
        ))

    def report_analysis_started(self, tasks: list[ImplementationTask]) -> None:
        """Report analysis phase started."""
        self._emit(ProgressEvent(
            event_type=EventType.ANALYSIS_STARTED,
            message=f"Analyzing conflict risk for {len(tasks)} tasks...",
            details={"task_count": len(tasks)},
        ))

    def report_analysis(self, plan: ParallelizationPlan) -> None:
        """Report conflict analysis complete.

        Args:
            plan: Parallelization plan.
        """
        self._emit(ProgressEvent(
            event_type=EventType.ANALYSIS_COMPLETE,
            message=(
                f"Risk: {plan.overall_risk.value.upper()}, "
                f"Recommendation: {plan.pm_recommendation}"
            ),
            details={
                "overall_risk": plan.overall_risk.value,
                "pm_recommendation": plan.pm_recommendation,
                "confidence": plan.confidence,
                "parallel_groups": len(plan.parallel_groups),
                "file_overlaps": len(plan.file_overlaps),
                "agent_assessments": len(plan.agent_assessments),
                "estimated_speedup": plan.estimated_speedup,
            },
        ))

    def report_ceo_confirmation_needed(
        self,
        plan: ParallelizationPlan,
        briefing: str,
    ) -> None:
        """Report CEO confirmation needed."""
        self._emit(ProgressEvent(
            event_type=EventType.CEO_CONFIRMATION_NEEDED,
            message=f"CEO confirmation required ({plan.overall_risk.value.upper()} risk)",
            details={
                "risk": plan.overall_risk.value,
                "recommendation": plan.pm_recommendation,
                "briefing": briefing,
            },
        ))

    def report_ceo_decision(self, decision: str) -> None:
        """Report CEO decision received."""
        self._emit(ProgressEvent(
            event_type=EventType.CEO_DECISION_RECEIVED,
            message=f"CEO decision: {decision}",
            details={"decision": decision},
        ))

    def report_execution_started(
        self,
        plan: ParallelizationPlan,
        tasks: list[ImplementationTask],
    ) -> None:
        """Report execution phase started."""
        mode = "parallel" if plan.pm_recommendation == "parallel" else "sequential"
        self._emit(ProgressEvent(
            event_type=EventType.EXECUTION_STARTED,
            message=f"Executing {len(tasks)} tasks in {mode} mode",
            details={
                "mode": mode,
                "task_count": len(tasks),
                "parallel_groups": plan.parallel_groups,
            },
        ))

    def report_execution_complete(self, results: list[SessionResult]) -> None:
        """Report execution phase complete."""
        successful = sum(1 for r in results if r.success)
        self._emit(ProgressEvent(
            event_type=EventType.EXECUTION_COMPLETE,
            message=f"Execution complete: {successful}/{len(results)} sessions succeeded",
            details={
                "total_sessions": len(results),
                "successful": successful,
                "failed": len(results) - successful,
            },
        ))

    def report_merge_started(self, session_count: int) -> None:
        """Report merge phase started."""
        self._emit(ProgressEvent(
            event_type=EventType.MERGE_STARTED,
            message=f"Merging {session_count} session branches...",
            details={"session_count": session_count},
        ))

    def report_merge_progress(self, files_merged: int, total: int) -> None:
        """Report merge progress.

        Args:
            files_merged: Number of files merged so far.
            total: Total files to merge.
        """
        self._emit(ProgressEvent(
            event_type=EventType.MERGE_STARTED,  # Reuse for progress
            message=f"Merging files: {files_merged}/{total}",
            details={"files_merged": files_merged, "total": total},
        ))

    def report_merge_complete(
        self,
        success: bool,
        conflicts: int,
        auto_resolved: int,
    ) -> None:
        """Report merge phase complete."""
        self._emit(ProgressEvent(
            event_type=EventType.MERGE_COMPLETE,
            message=(
                f"Merge {'succeeded' if success else 'needs attention'}: "
                f"{conflicts} conflicts, {auto_resolved} auto-resolved"
            ),
            details={
                "success": success,
                "conflicts": conflicts,
                "auto_resolved": auto_resolved,
            },
        ))

    # =========================================================================
    # Session Events
    # =========================================================================

    def report_session_start(self, task_id: str, branch: str) -> None:
        """Report session started.

        Args:
            task_id: Task identifier.
            branch: Git branch name.
        """
        self._emit(ProgressEvent(
            event_type=EventType.SESSION_STARTED,
            message=f"Session started: {task_id} on {branch}",
            details={"task_id": task_id, "branch": branch},
        ))

    def report_session_progress(self, task_id: str, progress: float) -> None:
        """Report session progress.

        Args:
            task_id: Task identifier.
            progress: Progress (0.0 - 1.0).
        """
        self._emit(ProgressEvent(
            event_type=EventType.SESSION_PROGRESS,
            message=f"Session {task_id}: {progress:.0%}",
            details={"task_id": task_id, "progress": progress},
        ))

    def report_session_complete(self, result: SessionResult) -> None:
        """Report session completed.

        Args:
            result: Session result.
        """
        self._emit(ProgressEvent(
            event_type=EventType.SESSION_COMPLETE if result.success else EventType.SESSION_FAILED,
            message=(
                f"Session {result.task_id}: "
                f"{'completed' if result.success else 'failed'} "
                f"({result.time_seconds:.1f}s, {result.tokens_used} tokens)"
            ),
            details={
                "task_id": result.task_id,
                "success": result.success,
                "files_changed": result.files_changed,
                "tokens_used": result.tokens_used,
                "time_seconds": result.time_seconds,
                "error": result.error,
            },
        ))

    # =========================================================================
    # Conflict Events
    # =========================================================================

    def report_boundary_crossing(self, crossing: BoundaryCrossing) -> None:
        """Report boundary crossing detected.

        Args:
            crossing: The boundary crossing.
        """
        self._emit(ProgressEvent(
            event_type=EventType.BOUNDARY_CROSSING,
            message=f"Boundary crossing: {crossing.session_task_id} needs {crossing.target_file}",
            details={
                "session": crossing.session_task_id,
                "target_file": crossing.target_file,
                "owning_agent": crossing.owning_agent,
                "reason": crossing.reason,
                "blocking": crossing.blocking,
            },
        ))

    def report_conflict(self, conflict: MergeConflict) -> None:
        """Report conflict detected.

        Args:
            conflict: The merge conflict.
        """
        self._emit(ProgressEvent(
            event_type=EventType.CONFLICT_DETECTED,
            message=f"Conflict in {conflict.file_path} ({conflict.conflict_type.value})",
            details={
                "file_path": conflict.file_path,
                "conflict_type": conflict.conflict_type.value,
                "branch_a": conflict.branch_a,
                "branch_b": conflict.branch_b,
                "owning_agent": conflict.owning_agent,
                "auto_resolvable": conflict.auto_resolvable,
            },
        ))

    def report_resolution(self, proposal: DomainResolutionProposal) -> None:
        """Report agent resolution proposed.

        Args:
            proposal: Resolution proposal.
        """
        self._emit(ProgressEvent(
            event_type=EventType.RESOLUTION_PROPOSED,
            message=f"Resolution from {proposal.agent_name} ({proposal.confidence:.0%} confidence)",
            details={
                "agent_id": proposal.agent_id,
                "agent_name": proposal.agent_name,
                "conflict_file": proposal.conflict_file,
                "confidence": proposal.confidence,
                "needs_ceo": proposal.needs_ceo,
                "reasoning": proposal.reasoning[:200],
            },
        ))

    def report_resolution_applied(self, conflict_file: str, agent: str) -> None:
        """Report resolution applied to file."""
        self._emit(ProgressEvent(
            event_type=EventType.RESOLUTION_APPLIED,
            message=f"Applied resolution to {conflict_file} from {agent}",
            details={"conflict_file": conflict_file, "agent": agent},
        ))

    def report_escalation_needed(self, reason: str, details: dict) -> None:
        """Report escalation to CEO needed."""
        self._emit(ProgressEvent(
            event_type=EventType.ESCALATION_NEEDED,
            message=f"Escalation needed: {reason}",
            details=details,
        ))

    # =========================================================================
    # Test Events
    # =========================================================================

    def report_tests_started(self, command: str) -> None:
        """Report test suite started."""
        self._emit(ProgressEvent(
            event_type=EventType.TESTS_STARTED,
            message=f"Running tests: {command}",
            details={"command": command},
        ))

    def report_tests_passed(self) -> None:
        """Report tests passed."""
        self._emit(ProgressEvent(
            event_type=EventType.TESTS_PASSED,
            message="All tests passed",
        ))

    def report_tests_failed(self, output: str) -> None:
        """Report tests failed."""
        self._emit(ProgressEvent(
            event_type=EventType.TESTS_FAILED,
            message="Tests failed",
            details={"output": output[:1000]},
        ))


def format_progress_for_cli(event: ProgressEvent) -> str:
    """Format a progress event for CLI output.

    Args:
        event: The progress event.

    Returns:
        Formatted string for terminal output.
    """
    # Use emoji indicators based on event type
    indicators = {
        EventType.STARTED: "🚀",
        EventType.COMPLETED: "✅",
        EventType.FAILED: "❌",
        EventType.DECOMPOSITION_STARTED: "📋",
        EventType.DECOMPOSITION_COMPLETE: "📋",
        EventType.ANALYSIS_STARTED: "🔍",
        EventType.ANALYSIS_COMPLETE: "🔍",
        EventType.CEO_CONFIRMATION_NEEDED: "👤",
        EventType.CEO_DECISION_RECEIVED: "👤",
        EventType.EXECUTION_STARTED: "⚡",
        EventType.EXECUTION_COMPLETE: "⚡",
        EventType.MERGE_STARTED: "🔀",
        EventType.MERGE_COMPLETE: "🔀",
        EventType.SESSION_STARTED: "▶️",
        EventType.SESSION_PROGRESS: "⏳",
        EventType.SESSION_COMPLETE: "✓",
        EventType.SESSION_FAILED: "✗",
        EventType.BOUNDARY_CROSSING: "⚠️",
        EventType.CONFLICT_DETECTED: "⚠️",
        EventType.RESOLUTION_PROPOSED: "💡",
        EventType.RESOLUTION_APPLIED: "✓",
        EventType.ESCALATION_NEEDED: "🔔",
        EventType.TESTS_STARTED: "🧪",
        EventType.TESTS_PASSED: "✅",
        EventType.TESTS_FAILED: "❌",
    }

    indicator = indicators.get(event.event_type, "•")
    timestamp = event.timestamp.strftime("%H:%M:%S")

    return f"[{timestamp}] {indicator} {event.message}"
