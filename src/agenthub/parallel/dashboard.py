from __future__ import annotations
"""Dashboard integration for parallel session events.

Broadcasts parallel execution events to the AgentHub dashboard
for real-time visualization.
"""

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional

from agenthub.parallel.models import (
    BoundaryCrossing,
    DecompositionResult,
    DomainResolutionProposal,
    MergeConflict,
    MergeResult,
    ParallelExecutionResult,
    ParallelizationPlan,
    SessionResult,
)
from agenthub.parallel.output import ProgressEvent


# Default dashboard URL (can be overridden via environment)
import os
DASHBOARD_URL = os.environ.get("AGENTHUB_DASHBOARD_URL", "http://localhost:3001")


def broadcast_parallel_event(
    event_type: str,
    description: str,
    details: Optional[dict] = None,
    dashboard_url: Optional[str] = None,
) -> bool:
    """Broadcast a parallel execution event to the dashboard.

    This is fire-and-forget - errors are silently ignored since
    the dashboard may not be running.

    Args:
        event_type: Type of event (e.g., "parallel_started", "session_complete").
        description: Human-readable event description.
        details: Optional additional details.
        dashboard_url: Optional override for dashboard URL.

    Returns:
        True if broadcast succeeded, False otherwise.
    """
    url = dashboard_url or DASHBOARD_URL

    try:
        data = {
            "event_type": event_type,
            "description": description,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "source": "parallel_sessions",
        }

        req = urllib.request.Request(
            f"{url}/api/parallel/events",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        urllib.request.urlopen(req, timeout=1)
        return True

    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
        # Dashboard not running or unreachable
        return False
    except Exception:
        # Other errors - log to stderr but don't fail
        return False


class DashboardReporter:
    """Reports parallel execution events to the dashboard.

    Wraps the broadcast function with structured event helpers.

    Example:
        >>> reporter = DashboardReporter()
        >>> reporter.report_started("Add save button and chart")
        >>> reporter.report_session_complete(session_result)
    """

    def __init__(self, dashboard_url: Optional[str] = None):
        """Initialize DashboardReporter.

        Args:
            dashboard_url: Optional dashboard URL override.
        """
        self._url = dashboard_url or DASHBOARD_URL
        self._execution_id: Optional[str] = None

    def set_execution_id(self, execution_id: str) -> None:
        """Set the current execution ID for event correlation."""
        self._execution_id = execution_id

    def _broadcast(self, event_type: str, description: str, details: dict) -> bool:
        """Broadcast with execution ID attached."""
        if self._execution_id:
            details["execution_id"] = self._execution_id
        return broadcast_parallel_event(event_type, description, details, self._url)

    # =========================================================================
    # Lifecycle Events
    # =========================================================================

    def report_started(self, request: str) -> None:
        """Report parallel execution started."""
        self._broadcast(
            "parallel_started",
            f"Parallel execution started: {request[:100]}",
            {"request": request},
        )

    def report_completed(self, result: ParallelExecutionResult) -> None:
        """Report parallel execution completed."""
        self._broadcast(
            "parallel_completed",
            f"Parallel execution {'succeeded' if result.success else 'failed'}",
            {
                "success": result.success,
                "tasks": len(result.tasks),
                "speedup": result.speedup,
                "total_time": result.total_time_seconds,
                "total_tokens": result.total_tokens,
            },
        )

    def report_failed(self, error: str) -> None:
        """Report parallel execution failed."""
        self._broadcast(
            "parallel_failed",
            f"Parallel execution failed: {error}",
            {"error": error},
        )

    # =========================================================================
    # Phase Events
    # =========================================================================

    def report_decomposition(self, result: DecompositionResult) -> None:
        """Report task decomposition complete."""
        self._broadcast(
            "task_decomposed",
            f"Decomposed into {len(result.tasks)} tasks",
            {
                "task_count": len(result.tasks),
                "tasks": [
                    {"id": t.task_id, "description": t.description[:50]}
                    for t in result.tasks
                ],
                "complexity": result.actual_complexity,
                "tokens_used": result.tokens_used,
            },
        )

    def report_analysis(self, plan: ParallelizationPlan) -> None:
        """Report risk analysis complete."""
        self._broadcast(
            "risk_analyzed",
            f"Risk: {plan.overall_risk.value.upper()}, Recommendation: {plan.pm_recommendation}",
            {
                "risk": plan.overall_risk.value,
                "recommendation": plan.pm_recommendation,
                "confidence": plan.confidence,
                "parallel_groups": plan.parallel_groups,
                "overlaps": len(plan.file_overlaps),
            },
        )

    def report_ceo_needed(self, plan: ParallelizationPlan, briefing: str) -> None:
        """Report CEO confirmation needed."""
        self._broadcast(
            "ceo_confirmation_needed",
            f"CEO confirmation required ({plan.overall_risk.value.upper()} risk)",
            {
                "risk": plan.overall_risk.value,
                "recommendation": plan.pm_recommendation,
                "briefing": briefing,
            },
        )

    def report_ceo_decision(self, decision: str) -> None:
        """Report CEO decision received."""
        self._broadcast(
            "ceo_decision",
            f"CEO decision: {decision}",
            {"decision": decision},
        )

    # =========================================================================
    # Session Events
    # =========================================================================

    def report_session_started(self, task_id: str, branch: str) -> None:
        """Report session started."""
        self._broadcast(
            "session_started",
            f"Session started: {task_id}",
            {"task_id": task_id, "branch": branch},
        )

    def report_session_progress(self, task_id: str, progress: float) -> None:
        """Report session progress."""
        self._broadcast(
            "session_progress",
            f"Session {task_id}: {progress:.0%}",
            {"task_id": task_id, "progress": progress},
        )

    def report_session_complete(self, result: SessionResult) -> None:
        """Report session completed."""
        self._broadcast(
            "session_completed",
            f"Session {result.task_id}: {'success' if result.success else 'failed'}",
            {
                "task_id": result.task_id,
                "success": result.success,
                "files_changed": result.files_changed,
                "tokens_used": result.tokens_used,
                "time_seconds": result.time_seconds,
                "error": result.error,
            },
        )

    # =========================================================================
    # Merge Events
    # =========================================================================

    def report_merge_started(self, session_count: int) -> None:
        """Report merge phase started."""
        self._broadcast(
            "merge_started",
            f"Merging {session_count} branches",
            {"session_count": session_count},
        )

    def report_merge_complete(self, result: MergeResult) -> None:
        """Report merge completed."""
        self._broadcast(
            "merge_completed",
            f"Merge {'succeeded' if result.success else 'needs attention'}",
            {
                "success": result.success,
                "conflicts": len(result.conflicts),
                "resolutions": len(result.resolutions),
                "files_merged": len(result.files_merged),
                "needs_user_input": result.needs_user_input,
            },
        )

    # =========================================================================
    # Conflict Events
    # =========================================================================

    def report_boundary_crossing(self, crossing: BoundaryCrossing) -> None:
        """Report boundary crossing detected."""
        self._broadcast(
            "boundary_crossing",
            f"Boundary crossing: {crossing.target_file}",
            {
                "session": crossing.session_task_id,
                "target_file": crossing.target_file,
                "owning_agent": crossing.owning_agent,
                "reason": crossing.reason,
            },
        )

    def report_conflict(self, conflict: MergeConflict) -> None:
        """Report merge conflict detected."""
        self._broadcast(
            "conflict_detected",
            f"Conflict: {conflict.file_path}",
            {
                "file_path": conflict.file_path,
                "conflict_type": conflict.conflict_type.value,
                "branches": [conflict.branch_a, conflict.branch_b],
                "owning_agent": conflict.owning_agent,
            },
        )

    def report_resolution(self, proposal: DomainResolutionProposal) -> None:
        """Report resolution proposed."""
        self._broadcast(
            "conflict_resolved",
            f"Resolution from {proposal.agent_name}",
            {
                "agent": proposal.agent_name,
                "file": proposal.conflict_file,
                "confidence": proposal.confidence,
                "needs_ceo": proposal.needs_ceo,
            },
        )

    # =========================================================================
    # Generic Event Forwarding
    # =========================================================================

    def forward_progress_event(self, event: ProgressEvent) -> None:
        """Forward a ProgressEvent to the dashboard.

        Useful for integrating with ProgressReporter.

        Args:
            event: Progress event to forward.
        """
        self._broadcast(
            event.event_type.value,
            event.message,
            event.details,
        )


def create_dashboard_callback(
    dashboard_url: Optional[str] = None,
) -> callable:
    """Create a callback function for ProgressReporter that broadcasts to dashboard.

    Args:
        dashboard_url: Optional dashboard URL override.

    Returns:
        Callback function for ProgressReporter.

    Example:
        >>> callback = create_dashboard_callback()
        >>> reporter = ProgressReporter(on_event=callback)
    """
    reporter = DashboardReporter(dashboard_url)

    def callback(event: ProgressEvent) -> None:
        reporter.forward_progress_event(event)

    return callback
