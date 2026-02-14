from __future__ import annotations
"""Top-level manager for parallel sessions.

The ParallelSessionManager is the main entry point for parallel session execution.
It coordinates all the components:
- TaskDecomposer: Breaks requests into tasks
- ConflictRiskAnalyzer: Assesses parallelization safety
- BranchOrchestrator: Manages git and spawns sessions
- MergeCoordinator: Merges branches with conflict resolution
- MidExecutionEscalationHandler: Handles CEO escalations

Example:
    >>> manager = ParallelSessionManager(hub, project_root)
    >>> result = manager.execute("Add save button and chart component")
    >>> print(f"Speedup: {result.speedup:.1f}x")
"""

import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger(__name__)

from agenthub.parallel.analyzer import ConflictRiskAnalyzer
from agenthub.parallel.decomposer import TaskDecomposer
from agenthub.parallel.escalation import (
    Escalation,
    EscalationResult,
    MidExecutionEscalationHandler,
)
from agenthub.parallel.merge import MergeCoordinator
from agenthub.parallel.models import (
    DecompositionResult,
    ParallelExecutionConfig,
    ParallelExecutionResult,
    ParallelExecutionTrace,
    ParallelizationPlan,
    RiskLevel,
)
from agenthub.parallel.orchestrator import BranchOrchestrator

if TYPE_CHECKING:
    import anthropic

    from agenthub.auto.import_graph import ImportGraph
    from agenthub.hub import AgentHub


class ParallelSessionManager:
    """Top-level orchestrator for parallel session execution.

    This is the main API for users to execute multi-part requests
    using parallel Claude Code sessions.

    Flow:
    1. Decompose request into tasks
    2. Analyze parallelization risk
    3. If MEDIUM/HIGH risk: escalate to CEO for confirmation
    4. Execute sessions (parallel or sequential per plan)
    5. Merge results with conflict resolution
    6. Return comprehensive result

    Example:
        >>> from agenthub.parallel import ParallelSessionManager, ParallelExecutionConfig
        >>> config = ParallelExecutionConfig(max_parallel_sessions=3)
        >>> manager = ParallelSessionManager(hub, "/path/to/project", config)
        >>> result = manager.execute("Add a save button and build a chart component")
        >>> if result.success:
        ...     print(f"Completed in {result.total_time_seconds:.1f}s")
        ...     print(f"Speedup: {result.speedup:.1f}x")
    """

    def __init__(
        self,
        hub: "AgentHub",
        project_root: str,
        config: Optional[ParallelExecutionConfig] = None,
        import_graph: Optional["ImportGraph"] = None,
        on_escalation: Optional[Callable[[Escalation], EscalationResult]] = None,
    ):
        """Initialize ParallelSessionManager.

        Args:
            hub: AgentHub for accessing domain agents.
            project_root: Path to the project root (must be git repo).
            config: Configuration for parallel execution.
            import_graph: Optional pre-built ImportGraph for the codebase.
            on_escalation: Callback for handling escalations synchronously.
                          If None, escalations will block and require manual resolution.
        """
        self._hub = hub
        self._root = Path(project_root).resolve()
        self._config = config or ParallelExecutionConfig()
        self._graph = import_graph
        self._on_escalation = on_escalation

        # Get Anthropic client from hub
        self._client: "anthropic.Anthropic" = hub.client

        # Initialize components
        self._decomposer = TaskDecomposer(self._client, hub, import_graph)
        self._analyzer = ConflictRiskAnalyzer(import_graph, hub)
        self._orchestrator = BranchOrchestrator(
            str(self._root),
            hub,
            max_parallel=self._config.max_parallel_sessions,
            claude_model=self._config.claude_model,
            session_timeout=self._config.session_timeout_seconds,
            execution_backend=self._config.execution_backend,
        )
        self._merger = MergeCoordinator(
            str(self._root),
            hub,
            self._client,
            run_tests=self._config.run_tests_after_merge,
            test_command=self._config.test_command,
        )
        self._escalation_handler = MidExecutionEscalationHandler(hub)

    def execute(
        self,
        request: str,
        base_branch: str = "main",
        precomputed: Optional[tuple["DecompositionResult", "ParallelizationPlan"]] = None,
    ) -> ParallelExecutionResult:
        """Execute a multi-part request using parallel sessions.

        This is the main entry point. It:
        1. Decomposes the request into tasks
        2. Analyzes risk and creates execution plan
        3. Handles CEO escalations if needed
        4. Executes sessions (parallel or sequential)
        5. Merges results
        6. Returns comprehensive result

        Args:
            request: The user's multi-part request.
            base_branch: Git branch to work from.
            precomputed: Optional (DecompositionResult, ParallelizationPlan) from
                        a prior preview_parallel() call.  Skips the expensive
                        decomposition + analysis phases when provided.

        Returns:
            ParallelExecutionResult with full execution details.
        """
        start_time = time.time()
        trace = ParallelExecutionTrace()

        try:
            if precomputed:
                decomposition, plan = precomputed
                trace.decomposition_time_ms = 0
                trace.decomposition_tokens = decomposition.tokens_used
                trace.analysis_time_ms = 0
                trace.analysis_tokens = sum(a.tokens_used for a in plan.agent_assessments)
            else:
                # Phase 1: Decomposition
                decomposition_start = time.time()
                decomposition = self._decomposer.decompose(request)
                trace.decomposition_time_ms = int((time.time() - decomposition_start) * 1000)
                trace.decomposition_tokens = decomposition.tokens_used

            # Quick exit for single task
            if len(decomposition.tasks) <= 1:
                return self._execute_single_task(decomposition, base_branch, trace, start_time)

            if not precomputed:
                # Phase 2: Risk Analysis
                analysis_start = time.time()
                plan = self._analyzer.analyze(decomposition.tasks)
                trace.analysis_time_ms = int((time.time() - analysis_start) * 1000)
            trace.analysis_tokens = sum(a.tokens_used for a in plan.agent_assessments)

            # Phase 3: CEO Escalation (if needed)
            plan = self._handle_risk_escalation(plan, request, decomposition)

            if not plan.parallel_groups and not plan.sequential_order:
                # User aborted
                return ParallelExecutionResult(
                    success=False,
                    tasks=decomposition.tasks,
                    plan=plan,
                    total_time_seconds=time.time() - start_time,
                    trace=trace,
                )

            # Phase 4: Session Execution
            session_results = self._orchestrator.execute_plan(plan, decomposition.tasks)

            # Record session metrics
            for result in session_results:
                trace.session_times[result.task_id] = int(result.time_seconds * 1000)
                trace.session_tokens[result.task_id] = result.tokens_used

            trace.parallel_groups = plan.parallel_groups

            # Phase 5: Merge
            merge_start = time.time()
            merge_result = self._merger.merge_sessions(session_results, base_branch)
            trace.merge_time_ms = int((time.time() - merge_start) * 1000)

            # Handle merge escalations if needed
            if merge_result.needs_user_input:
                merge_result = self._handle_merge_escalation(
                    merge_result,
                    session_results,
                )

            # Calculate metrics
            total_time = time.time() - start_time
            sequential_estimate = self._estimate_sequential_time(decomposition.tasks)
            speedup = sequential_estimate / total_time if total_time > 0 else 1.0

            # Build trace
            trace.total_time_ms = int(total_time * 1000)
            trace.total_tokens = (
                trace.decomposition_tokens +
                trace.analysis_tokens +
                sum(trace.session_tokens.values()) +
                trace.merge_tokens
            )
            trace.conflicts_found = len(merge_result.conflicts)
            trace.conflicts_auto_resolved = sum(
                1 for c in merge_result.conflicts if c.auto_resolvable
            )
            if merge_result.test_results:
                trace.test_pass_rate = 1.0 if merge_result.test_results.get("passed") else 0.0

            return ParallelExecutionResult(
                success=merge_result.success,
                tasks=decomposition.tasks,
                plan=plan,
                session_results=session_results,
                merge_result=merge_result,
                total_time_seconds=total_time,
                sequential_estimate_seconds=sequential_estimate,
                speedup=speedup,
                total_tokens=trace.total_tokens,
                trace=trace,
            )

        except Exception as e:
            # Log the full error for debugging
            error_msg = f"Parallel execution failed: {type(e).__name__}: {e}"
            error_tb = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_tb}")

            # Cleanup on failure
            try:
                self._orchestrator.rollback()
            except Exception as cleanup_err:
                logger.warning(f"Rollback failed: {cleanup_err}")
            try:
                self._merger.abort_merge()
            except Exception as cleanup_err:
                logger.warning(f"Abort merge failed: {cleanup_err}")

            # Preserve error details in trace
            trace.error_message = error_msg
            trace.error_traceback = error_tb

            return ParallelExecutionResult(
                success=False,
                total_time_seconds=time.time() - start_time,
                trace=trace,
                error_message=error_msg,
            )

    def _execute_single_task(
        self,
        decomposition: DecompositionResult,
        base_branch: str,
        trace: ParallelExecutionTrace,
        start_time: float,
    ) -> ParallelExecutionResult:
        """Execute a single-task decomposition.

        For simple requests that don't need parallel execution.

        Args:
            decomposition: Decomposition with single task.
            base_branch: Base git branch.
            trace: Trace for metrics.
            start_time: Execution start time.

        Returns:
            ParallelExecutionResult.
        """
        if not decomposition.tasks:
            return ParallelExecutionResult(
                success=False,
                total_time_seconds=time.time() - start_time,
                trace=trace,
            )

        task = decomposition.tasks[0]

        # Create minimal plan
        plan = ParallelizationPlan(
            parallel_groups=[[task.task_id]],
            sequential_order=[task.task_id],
            overall_risk=RiskLevel.NONE,
            pm_recommendation="parallel",  # Single task, doesn't matter
        )

        # Execute
        session_results = self._orchestrator.execute_plan(plan, decomposition.tasks)

        # Merge (trivial for single task)
        merge_result = self._merger.merge_sessions(session_results, base_branch)

        total_time = time.time() - start_time

        return ParallelExecutionResult(
            success=merge_result.success if merge_result else len(session_results) > 0 and session_results[0].success,
            tasks=decomposition.tasks,
            plan=plan,
            session_results=session_results,
            merge_result=merge_result,
            total_time_seconds=total_time,
            sequential_estimate_seconds=total_time,  # No speedup for single task
            speedup=1.0,
            total_tokens=trace.decomposition_tokens + sum(r.tokens_used for r in session_results),
            trace=trace,
        )

    def _handle_risk_escalation(
        self,
        plan: ParallelizationPlan,
        request: str,
        decomposition: DecompositionResult,
    ) -> ParallelizationPlan:
        """Handle CEO escalation for MEDIUM/HIGH risk plans.

        Args:
            plan: The parallelization plan.
            request: Original user request.
            decomposition: Task decomposition for context.

        Returns:
            Potentially modified plan based on CEO decision.
        """
        # Check if escalation needed
        auto_proceed_levels = {
            "safe": [RiskLevel.NONE],
            "caution": [RiskLevel.NONE, RiskLevel.LOW],
        }

        if plan.overall_risk in auto_proceed_levels.get(
            self._config.auto_proceed_threshold, []
        ):
            # No escalation needed
            return plan

        # Create escalation
        escalation = self._escalation_handler.escalate_risk(plan, request)

        # Get CEO decision
        if self._on_escalation:
            result = self._on_escalation(escalation)
        else:
            # No callback - wait for manual resolution
            result = self._wait_for_escalation_resolution(escalation)

        # Apply decision
        return self._escalation_handler.apply_risk_decision(result, plan)

    def _handle_merge_escalation(
        self,
        merge_result,
        session_results,
    ):
        """Handle escalations during merge phase.

        Args:
            merge_result: Current merge result.
            session_results: Session results.

        Returns:
            Updated merge result.
        """
        # For now, we just log the escalation need
        # Full implementation would handle each unresolved conflict
        if merge_result.escalation_reason:
            escalation = self._escalation_handler.escalate_test_failure(
                merge_result.test_results or {},
                session_results,
            )

            if self._on_escalation:
                result = self._on_escalation(escalation)
                if result.chosen_option == "rollback":
                    self._orchestrator.rollback()
                    self._merger.abort_merge()
                    merge_result.success = False

        return merge_result

    def _wait_for_escalation_resolution(
        self,
        escalation: Escalation,
    ) -> EscalationResult:
        """Wait for manual escalation resolution.

        This is a blocking call that should be overridden in
        interactive implementations.

        Args:
            escalation: The pending escalation.

        Returns:
            EscalationResult with user decision.
        """
        # Default: use recommended option
        recommended = next(
            (o for o in escalation.options if o.get("recommended")),
            escalation.options[0] if escalation.options else {"id": "abort"},
        )

        return EscalationResult(
            escalation_id=escalation.escalation_id,
            chosen_option=recommended["id"],
        )

    def _estimate_sequential_time(self, tasks) -> float:
        """Estimate how long sequential execution would take.

        Uses complexity estimates to approximate time.

        Args:
            tasks: List of tasks.

        Returns:
            Estimated time in seconds.
        """
        complexity_times = {
            "trivial": 30,
            "moderate": 120,
            "complex": 300,
        }

        total = sum(
            complexity_times.get(t.complexity, 120)
            for t in tasks
        )

        return float(total)

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def decompose(self, request: str) -> DecompositionResult:
        """Decompose a request without executing.

        Useful for previewing what tasks would be created.

        Args:
            request: The user request.

        Returns:
            DecompositionResult with tasks.
        """
        return self._decomposer.decompose(request)

    def analyze(self, request: str) -> tuple[DecompositionResult, ParallelizationPlan]:
        """Decompose and analyze a request without executing.

        Useful for previewing the execution plan.

        Args:
            request: The user request.

        Returns:
            Tuple of (DecompositionResult, ParallelizationPlan).
        """
        decomposition = self._decomposer.decompose(request)
        plan = self._analyzer.analyze(decomposition.tasks)
        return decomposition, plan

    def get_pending_escalations(self) -> list[Escalation]:
        """Get all pending escalations.

        Returns:
            List of unresolved escalations.
        """
        return self._escalation_handler.get_pending_escalations()

    def resolve_escalation(
        self,
        escalation_id: str,
        chosen_option: str,
        additional_input: Optional[str] = None,
    ) -> EscalationResult:
        """Resolve a pending escalation.

        Args:
            escalation_id: ID of the escalation.
            chosen_option: Chosen option ID.
            additional_input: Optional additional input.

        Returns:
            EscalationResult.
        """
        return self._escalation_handler.resolve_escalation(
            escalation_id,
            chosen_option,
            additional_input,
        )

    def rollback(self) -> None:
        """Rollback all changes from parallel execution.

        Cleans up worktrees, deletes parallel branches.
        """
        self._orchestrator.rollback()
        self._merger.delete_integration_branch()

    @property
    def config(self) -> ParallelExecutionConfig:
        """Get the current configuration."""
        return self._config

    @config.setter
    def config(self, value: ParallelExecutionConfig) -> None:
        """Update configuration.

        Note: Some settings (like execution_backend) may require
        reinitializing components.
        """
        self._config = value
        # Update orchestrator settings
        self._orchestrator._max_parallel = value.max_parallel_sessions
        self._orchestrator._model = value.claude_model
        self._orchestrator._timeout = value.session_timeout_seconds
        # Merger settings
        self._merger._run_tests = value.run_tests_after_merge
        self._merger._test_command = value.test_command
