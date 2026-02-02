"""Change analysis pipeline orchestrating Tier B agents and QC Agent.

This module provides the ChangeAnalysisPipeline class that:
1. Receives file changes (from FileWatcher or manual trigger)
2. Determines affected Tier B agents
3. Runs analysis in parallel across affected agents
4. Collects concerns from all agents
5. Passes to QC Agent for synthesis
6. Produces final ConcernReport
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Optional

from agenthub.qc.models import (
    AgentAnalysisResult,
    ChangeSet,
    Concern,
    ConcernReport,
)

if TYPE_CHECKING:
    from agenthub.hub import AgentHub
    from agenthub.qc.qc_agent import QCAgent


class ChangeAnalysisPipeline:
    """Orchestrates the change analysis process.

    Flow:
    1. Receives file changes (from FileWatcher or manual trigger)
    2. Determines affected Tier B agents
    3. Runs analysis in parallel across affected agents
    4. Collects concerns from all agents
    5. Passes to QC Agent for synthesis
    6. Produces final ConcernReport

    Example:
        >>> pipeline = ChangeAnalysisPipeline(hub)
        >>> change_set = ChangeSet(...)
        >>> report = pipeline.analyze(change_set)
        >>> print(f"Found {report.total_concerns} concerns")
    """

    def __init__(
        self,
        hub: "AgentHub",
        qc_agent: Optional["QCAgent"] = None,
        on_concern_raised: Optional[Callable[[Concern], None]] = None,
        on_report_complete: Optional[Callable[[ConcernReport], None]] = None,
        max_parallel_agents: int = 5,
        timeout_per_agent_seconds: float = 60.0,
    ):
        """Initialize the pipeline.

        Args:
            hub: AgentHub instance.
            qc_agent: QC Agent (Tier C). If None, creates default.
            on_concern_raised: Callback when a concern is raised.
            on_report_complete: Callback when report is complete.
            max_parallel_agents: Maximum agents to analyze in parallel.
            timeout_per_agent_seconds: Timeout for each agent's analysis.
        """
        self.hub = hub
        self.qc_agent = qc_agent
        self.on_concern_raised = on_concern_raised
        self.on_report_complete = on_report_complete
        self.max_parallel_agents = max_parallel_agents
        self.timeout_per_agent = timeout_per_agent_seconds

        # History of analyses
        self._analysis_history: list[ConcernReport] = []
        self._max_history = 100

    def analyze(self, change_set: ChangeSet) -> ConcernReport:
        """Run the full analysis pipeline.

        Args:
            change_set: Set of file changes to analyze.

        Returns:
            ConcernReport with all concerns and action items.
        """
        start_time = time.time()

        # Step 1: Find affected Tier B agents
        affected_agents = self._find_affected_agents(change_set)

        # Step 2: Run analysis in parallel
        analysis_results = self._run_parallel_analysis(change_set, affected_agents)

        # Step 3: Collect all concerns
        all_concerns: list[Concern] = []
        for result in analysis_results:
            for concern in result.concerns:
                all_concerns.append(concern)
                if self.on_concern_raised:
                    self.on_concern_raised(concern)

        # Step 4: Synthesize with QC Agent
        report = self._synthesize_with_qc_agent(
            change_set, analysis_results, all_concerns
        )

        # Update timing
        total_time = int((time.time() - start_time) * 1000)
        report.total_analysis_time_ms = total_time

        # Store in history
        self._analysis_history.append(report)
        if len(self._analysis_history) > self._max_history:
            self._analysis_history = self._analysis_history[-self._max_history :]

        # Callback
        if self.on_report_complete:
            self.on_report_complete(report)

        return report

    def _find_affected_agents(self, change_set: ChangeSet) -> list[str]:
        """Find Tier B agents affected by the changes."""
        affected = []

        # Get all agents and filter for Tier B
        for spec in self.hub.list_agents():
            if not spec.metadata.get("auto_generated", False):
                continue  # Skip Tier A agents

            if spec.metadata.get("tier") == "C":
                continue  # Skip Tier C (QC) agent

            agent = self.hub.get_agent(spec.agent_id)
            if agent and agent.can_analyze_changes():
                # Check if any files affect this agent
                affected_files = agent.get_affected_files(change_set)
                if affected_files:
                    affected.append(spec.agent_id)

        return affected

    def _run_parallel_analysis(
        self,
        change_set: ChangeSet,
        agent_ids: list[str],
    ) -> list[AgentAnalysisResult]:
        """Run analysis across multiple agents in parallel."""
        results: list[AgentAnalysisResult] = []

        if not agent_ids:
            return results

        def analyze_with_agent(agent_id: str) -> AgentAnalysisResult:
            agent = self.hub.get_agent(agent_id)
            if not agent:
                return AgentAnalysisResult(
                    agent_id=agent_id,
                    domain="unknown",
                    analyzed_files=[],
                    concerns=[],
                    analysis_time_ms=0,
                    tokens_used=0,
                    skipped_reason="Agent not found",
                )

            try:
                return agent.analyze_changes(change_set)
            except Exception as e:
                return AgentAnalysisResult(
                    agent_id=agent_id,
                    domain=agent.spec.name,
                    analyzed_files=[],
                    concerns=[],
                    analysis_time_ms=0,
                    tokens_used=0,
                    skipped_reason=f"Error: {str(e)}",
                )

        # Run in parallel with thread pool
        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            futures = {
                executor.submit(analyze_with_agent, agent_id): agent_id
                for agent_id in agent_ids
            }

            for future in as_completed(futures, timeout=self.timeout_per_agent * len(agent_ids)):
                try:
                    result = future.result(timeout=self.timeout_per_agent)
                    results.append(result)
                except Exception as e:
                    agent_id = futures[future]
                    results.append(
                        AgentAnalysisResult(
                            agent_id=agent_id,
                            domain="unknown",
                            analyzed_files=[],
                            concerns=[],
                            analysis_time_ms=int(self.timeout_per_agent * 1000),
                            tokens_used=0,
                            skipped_reason=f"Timeout or error: {str(e)}",
                        )
                    )

        return results

    def _synthesize_with_qc_agent(
        self,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
        all_concerns: list[Concern],
    ) -> ConcernReport:
        """Use QC Agent to synthesize concerns into a report."""
        if self.qc_agent:
            return self.qc_agent.synthesize_concerns(
                change_set, analysis_results, all_concerns
            )
        else:
            # Fallback: Basic synthesis without QC Agent
            return self._basic_synthesis(change_set, analysis_results, all_concerns)

    def _basic_synthesis(
        self,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
        all_concerns: list[Concern],
    ) -> ConcernReport:
        """Basic synthesis without QC Agent."""
        from agenthub.qc.models import ActionItem, ConcernSeverity

        # Count by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for concern in all_concerns:
            severity_counts[concern.severity.value] += 1

        # Group by category
        by_category: dict[str, list[Concern]] = {}
        for concern in all_concerns:
            cat = concern.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(concern)

        # Sort all concerns by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_concerns = sorted(
            all_concerns,
            key=lambda c: severity_order.get(c.severity.value, 5),
        )

        # Generate basic action items from critical/high concerns
        action_items = []
        priority = 1
        for concern in sorted_concerns:
            if concern.severity in [ConcernSeverity.CRITICAL, ConcernSeverity.HIGH]:
                action_items.append(
                    ActionItem(
                        action_id=str(uuid.uuid4())[:8],
                        priority=priority,
                        title=f"Address: {concern.title}",
                        description=concern.suggestion or concern.description,
                        related_concerns=[concern.concern_id],
                        assignee_hint=concern.domain,
                    )
                )
                priority += 1
                if priority > 10:
                    break

        # Determine risk level
        if severity_counts["critical"] > 0:
            risk_level = "critical"
            recommendation = "block"
        elif severity_counts["high"] > 2:
            risk_level = "high"
            recommendation = "review"
        elif severity_counts["high"] > 0 or severity_counts["medium"] > 3:
            risk_level = "medium"
            recommendation = "review"
        else:
            risk_level = "low"
            recommendation = "approve"

        # Total tokens
        total_tokens = sum(r.tokens_used for r in analysis_results)

        # Generate assessment
        if all_concerns:
            assessment = f"Found {len(all_concerns)} concern(s) across {len(analysis_results)} domain(s). "
            if severity_counts["critical"] > 0:
                assessment += f"Critical issues require immediate attention. "
            elif severity_counts["high"] > 0:
                assessment += f"High-priority issues should be addressed before merge. "
            else:
                assessment += "Most issues are medium to low severity."
        else:
            assessment = "No concerns were identified in the analyzed changes."

        return ConcernReport(
            report_id=str(uuid.uuid4())[:8],
            change_set_id=change_set.change_id,
            total_concerns=len(all_concerns),
            critical_count=severity_counts["critical"],
            high_count=severity_counts["high"],
            medium_count=severity_counts["medium"],
            low_count=severity_counts["low"],
            concerns_by_category=by_category,
            all_concerns=sorted_concerns,
            action_items=action_items,
            agents_consulted=[r.agent_id for r in analysis_results],
            analysis_results=analysis_results,
            total_analysis_time_ms=0,  # Will be updated by caller
            total_tokens_used=total_tokens,
            overall_assessment=assessment,
            risk_level=risk_level,
            recommendation=recommendation,
        )

    def get_history(self, limit: int = 20) -> list[ConcernReport]:
        """Get recent analysis reports.

        Args:
            limit: Maximum number of reports to return.

        Returns:
            List of recent ConcernReports, newest first.
        """
        return list(reversed(self._analysis_history[-limit:]))

    def get_report(self, report_id: str) -> Optional[ConcernReport]:
        """Get a specific report by ID.

        Args:
            report_id: Report ID to find.

        Returns:
            ConcernReport if found, None otherwise.
        """
        for report in self._analysis_history:
            if report.report_id == report_id:
                return report
        return None

    def clear_history(self) -> None:
        """Clear the analysis history."""
        self._analysis_history = []
