from __future__ import annotations
"""Mid-execution escalation handling for parallel sessions.

The MidExecutionEscalationHandler manages escalations during parallel execution:
- Boundary crossing negotiations that need CEO approval
- Domain agent conflicts during execution
- Risk level upgrades that require confirmation
- Session failures that need user guidance

Escalations can happen at multiple points:
1. Pre-execution: Risk analysis returns MEDIUM/HIGH
2. During execution: Boundary crossing detected (Agent Teams mode)
3. Post-execution: Merge conflicts or test failures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

from agenthub.parallel.models import (
    BoundaryCrossing,
    BoundaryCrossingResolution,
    CrossingResolutionType,
    MergeConflict,
    ParallelizationPlan,
    RiskLevel,
    SessionResult,
)

if TYPE_CHECKING:
    from agenthub.hub import AgentHub


class EscalationType(str, Enum):
    """Type of escalation requiring CEO decision."""

    RISK_CONFIRMATION = "risk_confirmation"
    """Plan has MEDIUM/HIGH risk - need confirmation to proceed."""

    BOUNDARY_CROSSING = "boundary_crossing"
    """Session needs file outside scope - need approval."""

    MERGE_CONFLICT = "merge_conflict"
    """Unresolvable merge conflict - need resolution."""

    TEST_FAILURE = "test_failure"
    """Tests fail after merge - need guidance."""

    SESSION_FAILURE = "session_failure"
    """Session failed to complete - need guidance."""

    AGENT_DISAGREEMENT = "agent_disagreement"
    """Domain agents disagree on resolution - need tiebreaker."""


@dataclass
class Escalation:
    """An escalation requiring CEO decision.

    Tracks the full context needed for the CEO to make an informed decision.
    """

    escalation_id: str
    escalation_type: EscalationType
    summary: str
    """Brief one-line summary."""

    details: str
    """Full details for CEO review."""

    options: list[dict] = field(default_factory=list)
    """Available options for the CEO to choose from.

    Each option is a dict with:
    - id: str
    - label: str
    - description: str
    - recommended: bool
    """

    context: dict = field(default_factory=dict)
    """Additional context data."""

    resolved: bool = False
    resolution: Optional[str] = None
    """Chosen option ID after resolution."""


@dataclass
class EscalationResult:
    """Result of an escalation decision."""

    escalation_id: str
    chosen_option: str
    additional_input: Optional[str] = None
    """Free-form input from CEO if provided."""


class MidExecutionEscalationHandler:
    """Handles escalations during parallel session execution.

    Responsibilities:
    - Tracking pending escalations
    - Building CEO-friendly escalation prompts
    - Applying escalation decisions
    - Supporting both sync (CLI) and async (Agent Teams) escalation

    Example:
        >>> handler = MidExecutionEscalationHandler(hub)
        >>> escalation = handler.escalate_risk(plan)
        >>> # Show to user, get decision
        >>> handler.resolve_escalation(escalation.escalation_id, "proceed_parallel")
    """

    def __init__(
        self,
        hub: "AgentHub",
        on_escalation: Optional[Callable[[Escalation], None]] = None,
    ):
        """Initialize MidExecutionEscalationHandler.

        Args:
            hub: AgentHub for context.
            on_escalation: Optional callback when escalation is created.
        """
        self._hub = hub
        self._on_escalation = on_escalation
        self._escalations: dict[str, Escalation] = {}
        self._escalation_counter = 0

    def _next_id(self) -> str:
        """Generate next escalation ID."""
        self._escalation_counter += 1
        return f"esc_{self._escalation_counter:04d}"

    # =========================================================================
    # Risk Escalation
    # =========================================================================

    def escalate_risk(
        self,
        plan: ParallelizationPlan,
        request: str,
    ) -> Escalation:
        """Create escalation for MEDIUM/HIGH risk plan.

        This is the "CEO shot-call" - user decides whether to proceed
        with parallel execution or fall back to sequential.

        Args:
            plan: The parallelization plan.
            request: Original user request.

        Returns:
            Escalation for CEO decision.
        """
        escalation_id = self._next_id()

        # Build summary
        if plan.overall_risk == RiskLevel.HIGH:
            summary = "High risk detected - parallel execution may cause conflicts"
        else:
            summary = "Medium risk detected - some overlap between tasks"

        # Build details
        details = self._build_risk_details(plan, request)

        # Build options
        options = [
            {
                "id": "proceed_parallel",
                "label": "Proceed with Parallel",
                "description": (
                    f"Run tasks in parallel groups. "
                    f"Estimated speedup: {plan.estimated_speedup:.1f}x"
                ),
                "recommended": plan.pm_recommendation == "parallel",
            },
            {
                "id": "proceed_sequential",
                "label": "Proceed Sequentially",
                "description": (
                    "Run tasks one at a time in recommended order. "
                    "Slower but safer."
                ),
                "recommended": plan.pm_recommendation == "sequential",
            },
            {
                "id": "abort",
                "label": "Abort",
                "description": "Cancel the operation.",
                "recommended": False,
            },
        ]

        escalation = Escalation(
            escalation_id=escalation_id,
            escalation_type=EscalationType.RISK_CONFIRMATION,
            summary=summary,
            details=details,
            options=options,
            context={
                "plan": plan,
                "request": request,
            },
        )

        self._escalations[escalation_id] = escalation

        if self._on_escalation:
            self._on_escalation(escalation)

        return escalation

    def _build_risk_details(self, plan: ParallelizationPlan, request: str) -> str:
        """Build detailed risk explanation for CEO.

        Args:
            plan: The parallelization plan.
            request: Original request.

        Returns:
            Formatted details string.
        """
        lines = [
            f"**Your Request:** {request}",
            "",
            f"**Risk Level:** {plan.overall_risk.value.upper()}",
            f"**PM Recommendation:** {plan.pm_recommendation}",
            "",
            "**Analysis:**",
            plan.reasoning,
            "",
        ]

        if plan.file_overlaps:
            lines.append("**File Overlaps Detected:**")
            for overlap in plan.file_overlaps[:5]:
                tasks = ", ".join(overlap.tasks_touching)
                lines.append(f"- `{overlap.file_path}` ({overlap.overlap_type.value}) - Tasks: {tasks}")
            if len(plan.file_overlaps) > 5:
                lines.append(f"- ... and {len(plan.file_overlaps) - 5} more")
            lines.append("")

        if plan.agent_assessments:
            concerns = [a for a in plan.agent_assessments if a.has_concern]
            if concerns:
                lines.append("**Agent Concerns:**")
                for assessment in concerns[:3]:
                    lines.append(f"- **{assessment.agent_name}**: {assessment.concern_description}")
                lines.append("")

        lines.append(f"**Parallel Groups:** {len(plan.parallel_groups)}")
        lines.append(f"**Estimated Speedup:** {plan.estimated_speedup:.1f}x")
        lines.append(f"**Estimated Tokens:** {plan.estimated_total_tokens:,}")

        return "\n".join(lines)

    # =========================================================================
    # Boundary Crossing Escalation
    # =========================================================================

    def escalate_boundary_crossing(
        self,
        crossing: BoundaryCrossing,
        owning_agent_response: Optional[str] = None,
    ) -> Escalation:
        """Create escalation for boundary crossing needing approval.

        Called when:
        - Owning agent rejects the crossing
        - Owning agent has low confidence
        - No owning agent identified

        Args:
            crossing: The boundary crossing.
            owning_agent_response: Optional response from owning agent.

        Returns:
            Escalation for CEO decision.
        """
        escalation_id = self._next_id()

        summary = f"Session needs access to {crossing.target_file}"

        details = self._build_crossing_details(crossing, owning_agent_response)

        options = [
            {
                "id": "approve",
                "label": "Approve Access",
                "description": "Allow the session to modify this file.",
                "recommended": not crossing.blocking,
            },
            {
                "id": "reject",
                "label": "Reject Access",
                "description": "Deny access; session should find alternative.",
                "recommended": False,
            },
            {
                "id": "defer",
                "label": "Defer to Merge",
                "description": "Let session proceed; handle during merge phase.",
                "recommended": crossing.blocking,
            },
        ]

        escalation = Escalation(
            escalation_id=escalation_id,
            escalation_type=EscalationType.BOUNDARY_CROSSING,
            summary=summary,
            details=details,
            options=options,
            context={
                "crossing": crossing,
                "owning_agent_response": owning_agent_response,
            },
        )

        self._escalations[escalation_id] = escalation

        if self._on_escalation:
            self._on_escalation(escalation)

        return escalation

    def _build_crossing_details(
        self,
        crossing: BoundaryCrossing,
        owning_agent_response: Optional[str],
    ) -> str:
        """Build details for boundary crossing escalation.

        Args:
            crossing: The boundary crossing.
            owning_agent_response: Owning agent's response.

        Returns:
            Formatted details string.
        """
        lines = [
            f"**Requesting Task:** {crossing.session_task_id}",
            f"**Target File:** `{crossing.target_file}`",
            f"**Reason:** {crossing.reason}",
            "",
        ]

        if crossing.owning_agent:
            lines.append(f"**Owning Agent:** {crossing.owning_agent}")
        else:
            lines.append("**Owning Agent:** None identified")

        if owning_agent_response:
            lines.append("")
            lines.append(f"**Agent Response:** {owning_agent_response}")

        if crossing.proposed_change:
            lines.append("")
            lines.append("**Proposed Change:**")
            lines.append(f"```\n{crossing.proposed_change[:500]}\n```")

        if crossing.blocking:
            lines.append("")
            lines.append("**Note:** This is a blocking request - the session cannot complete without resolution.")

        return "\n".join(lines)

    # =========================================================================
    # Merge Conflict Escalation
    # =========================================================================

    def escalate_merge_conflict(
        self,
        conflict: MergeConflict,
        resolution_attempts: list[dict],
    ) -> Escalation:
        """Create escalation for unresolvable merge conflict.

        Args:
            conflict: The merge conflict.
            resolution_attempts: Previous resolution attempts by agents.

        Returns:
            Escalation for CEO decision.
        """
        escalation_id = self._next_id()

        summary = f"Merge conflict in {conflict.file_path}"

        details = self._build_conflict_details(conflict, resolution_attempts)

        options = [
            {
                "id": "use_branch_a",
                "label": f"Use {conflict.branch_a}",
                "description": "Accept changes from branch A only.",
                "recommended": False,
            },
            {
                "id": "use_branch_b",
                "label": f"Use {conflict.branch_b}",
                "description": "Accept changes from branch B only.",
                "recommended": False,
            },
            {
                "id": "manual_edit",
                "label": "Manual Edit",
                "description": "I'll provide the correct resolution.",
                "recommended": True,
            },
            {
                "id": "skip_file",
                "label": "Skip File",
                "description": "Exclude this file from the merge (may break build).",
                "recommended": False,
            },
        ]

        escalation = Escalation(
            escalation_id=escalation_id,
            escalation_type=EscalationType.MERGE_CONFLICT,
            summary=summary,
            details=details,
            options=options,
            context={
                "conflict": conflict,
                "resolution_attempts": resolution_attempts,
            },
        )

        self._escalations[escalation_id] = escalation

        if self._on_escalation:
            self._on_escalation(escalation)

        return escalation

    def _build_conflict_details(
        self,
        conflict: MergeConflict,
        resolution_attempts: list[dict],
    ) -> str:
        """Build details for merge conflict escalation.

        Args:
            conflict: The merge conflict.
            resolution_attempts: Previous attempts.

        Returns:
            Formatted details string.
        """
        lines = [
            f"**File:** `{conflict.file_path}`",
            f"**Conflict Type:** {conflict.conflict_type.value}",
            "",
            f"**Branch A ({conflict.branch_a}):**",
            f"```\n{conflict.diff_a[:800]}\n```",
            "",
            f"**Branch B ({conflict.branch_b}):**",
            f"```\n{conflict.diff_b[:800]}\n```",
        ]

        if resolution_attempts:
            lines.append("")
            lines.append("**Resolution Attempts:**")
            for attempt in resolution_attempts:
                agent = attempt.get("agent", "Unknown")
                confidence = attempt.get("confidence", 0)
                reasoning = attempt.get("reasoning", "")
                lines.append(f"- **{agent}** (confidence: {confidence:.0%}): {reasoning}")

        return "\n".join(lines)

    # =========================================================================
    # Test Failure Escalation
    # =========================================================================

    def escalate_test_failure(
        self,
        test_results: dict,
        session_results: list[SessionResult],
    ) -> Escalation:
        """Create escalation for test failures after merge.

        Args:
            test_results: Test execution results.
            session_results: Results from merged sessions.

        Returns:
            Escalation for CEO decision.
        """
        escalation_id = self._next_id()

        summary = "Tests fail after merging parallel sessions"

        details = self._build_test_failure_details(test_results, session_results)

        options = [
            {
                "id": "investigate",
                "label": "Investigate",
                "description": "Let me look into the failures and try to fix them.",
                "recommended": True,
            },
            {
                "id": "accept_failures",
                "label": "Accept & Proceed",
                "description": "Proceed despite test failures (use with caution).",
                "recommended": False,
            },
            {
                "id": "rollback",
                "label": "Rollback",
                "description": "Abort the merge and rollback all changes.",
                "recommended": False,
            },
            {
                "id": "sequential_redo",
                "label": "Redo Sequentially",
                "description": "Discard parallel work and redo one at a time.",
                "recommended": False,
            },
        ]

        escalation = Escalation(
            escalation_id=escalation_id,
            escalation_type=EscalationType.TEST_FAILURE,
            summary=summary,
            details=details,
            options=options,
            context={
                "test_results": test_results,
                "session_results": session_results,
            },
        )

        self._escalations[escalation_id] = escalation

        if self._on_escalation:
            self._on_escalation(escalation)

        return escalation

    def _build_test_failure_details(
        self,
        test_results: dict,
        session_results: list[SessionResult],
    ) -> str:
        """Build details for test failure escalation.

        Args:
            test_results: Test results.
            session_results: Session results.

        Returns:
            Formatted details string.
        """
        lines = [
            "**Test Results:**",
            f"```\n{test_results.get('stdout', '')[:1500]}\n```",
        ]

        if test_results.get("stderr"):
            lines.append("")
            lines.append("**Errors:**")
            lines.append(f"```\n{test_results['stderr'][:500]}\n```")

        lines.append("")
        lines.append("**Merged Sessions:**")
        for session in session_results:
            files = ", ".join(session.files_changed[:3])
            if len(session.files_changed) > 3:
                files += f" (+{len(session.files_changed) - 3} more)"
            status = "success" if session.success else "failed"
            lines.append(f"- {session.task_id}: {status} - Files: {files}")

        return "\n".join(lines)

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve_escalation(
        self,
        escalation_id: str,
        chosen_option: str,
        additional_input: Optional[str] = None,
    ) -> EscalationResult:
        """Record resolution of an escalation.

        Args:
            escalation_id: The escalation to resolve.
            chosen_option: ID of the chosen option.
            additional_input: Optional free-form input.

        Returns:
            EscalationResult with the decision.

        Raises:
            KeyError: If escalation not found.
        """
        if escalation_id not in self._escalations:
            raise KeyError(f"Escalation not found: {escalation_id}")

        escalation = self._escalations[escalation_id]
        escalation.resolved = True
        escalation.resolution = chosen_option

        return EscalationResult(
            escalation_id=escalation_id,
            chosen_option=chosen_option,
            additional_input=additional_input,
        )

    def apply_risk_decision(
        self,
        result: EscalationResult,
        plan: ParallelizationPlan,
    ) -> ParallelizationPlan:
        """Apply CEO's risk decision to the plan.

        Args:
            result: The escalation result.
            plan: The original plan.

        Returns:
            Modified plan based on decision.
        """
        if result.chosen_option == "proceed_sequential":
            # Override to sequential
            plan.pm_recommendation = "sequential"
            plan.parallel_groups = [[tid] for tid in plan.sequential_order]

        elif result.chosen_option == "abort":
            # Return empty plan
            plan.parallel_groups = []
            plan.sequential_order = []

        # "proceed_parallel" keeps the plan as-is

        return plan

    def apply_crossing_decision(
        self,
        result: EscalationResult,
        crossing: BoundaryCrossing,
    ) -> BoundaryCrossingResolution:
        """Apply CEO's crossing decision.

        Args:
            result: The escalation result.
            crossing: The original crossing.

        Returns:
            BoundaryCrossingResolution with the decision.
        """
        if result.chosen_option == "approve":
            return BoundaryCrossingResolution(
                crossing=crossing,
                approved=True,
                resolution_type=CrossingResolutionType.APPROVED_AS_IS,
                reasoning="Approved by user",
                confidence=1.0,
            )

        elif result.chosen_option == "reject":
            return BoundaryCrossingResolution(
                crossing=crossing,
                approved=False,
                resolution_type=CrossingResolutionType.REJECTED,
                reasoning="Rejected by user",
                confidence=1.0,
            )

        else:  # defer
            return BoundaryCrossingResolution(
                crossing=crossing,
                approved=True,
                resolution_type=CrossingResolutionType.DEFERRED_TO_MERGE,
                reasoning="Deferred to merge phase by user",
                confidence=1.0,
            )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_pending_escalations(self) -> list[Escalation]:
        """Get all unresolved escalations.

        Returns:
            List of pending escalations.
        """
        return [e for e in self._escalations.values() if not e.resolved]

    def get_escalation(self, escalation_id: str) -> Optional[Escalation]:
        """Get an escalation by ID.

        Args:
            escalation_id: The escalation ID.

        Returns:
            Escalation or None if not found.
        """
        return self._escalations.get(escalation_id)

    def has_pending_escalations(self) -> bool:
        """Check if there are any pending escalations.

        Returns:
            True if there are unresolved escalations.
        """
        return any(not e.resolved for e in self._escalations.values())

    def clear_resolved(self) -> None:
        """Remove all resolved escalations from memory."""
        self._escalations = {
            eid: e for eid, e in self._escalations.items()
            if not e.resolved
        }
