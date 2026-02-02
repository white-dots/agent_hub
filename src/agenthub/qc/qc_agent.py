"""QC Agent - Tier C meta-agent for concern synthesis.

The QC Agent collects concerns from all Tier B agents and synthesizes them
into a cohesive quality report with prioritized action items.
"""

import json
import re
import uuid
from typing import TYPE_CHECKING, Optional

from agenthub.agents.base import BaseAgent
from agenthub.models import AgentSpec
from agenthub.qc.models import (
    ActionItem,
    AgentAnalysisResult,
    ChangeSet,
    Concern,
    ConcernReport,
    ConcernSeverity,
)

if TYPE_CHECKING:
    import anthropic


class QCAgent(BaseAgent):
    """Quality Control Agent - Tier C meta-agent.

    This agent:
    1. Collects concerns from all Tier B agents
    2. Synthesizes them into a cohesive report
    3. Prioritizes issues across domains
    4. Identifies cross-cutting concerns
    5. Suggests action items
    6. Makes overall recommendation (approve/review/block)

    Example:
        >>> qc = QCAgent(client)
        >>> report = qc.synthesize_concerns(change_set, results, concerns)
        >>> print(report.recommendation)
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        spec: Optional[AgentSpec] = None,
    ):
        """Initialize QC Agent.

        Args:
            client: Anthropic API client.
            spec: Optional custom spec. Uses default if not provided.
        """
        if spec is None:
            spec = AgentSpec(
                agent_id="qc_agent",
                name="Quality Control Agent",
                description="Meta-agent that synthesizes concerns from all Tier B agents and produces QC reports",
                context_keywords=[
                    "quality",
                    "qc",
                    "review",
                    "concern",
                    "issue",
                    "breaking",
                    "security",
                    "performance",
                    "test",
                    "code quality",
                    "assessment",
                    "report",
                ],
                system_prompt=self._default_system_prompt(),
                metadata={
                    "tier": "C",
                    "auto_generated": False,
                    "role": "meta_agent",
                },
            )

        super().__init__(spec, client)

    def _default_system_prompt(self) -> str:
        return """# Quality Control Agent (Tier C)

You are the QC Agent for this codebase. Your role is to:

1. **Collect** concerns raised by domain-specific Tier B agents
2. **Synthesize** them into a cohesive quality report
3. **Prioritize** issues across all domains
4. **Identify** cross-cutting concerns that affect multiple areas
5. **Deduplicate** similar or related concerns
6. **Suggest** concrete action items
7. **Recommend** whether the changes should be approved, reviewed, or blocked

## Assessment Guidelines

### Risk Levels
- **critical**: Security vulnerabilities, data loss risks, production-breaking changes
- **high**: Breaking API changes, significant performance regressions, missing critical tests
- **medium**: Code quality issues, minor performance concerns, documentation gaps
- **low**: Style issues, minor improvements, informational notes

### Recommendations
- **approve**: Low risk, changes look good
- **review**: Medium risk, human review recommended before merge
- **block**: High/Critical risk, must address before merge

## Output Format
Always respond with structured JSON containing:
- Overall assessment (2-3 sentences)
- Risk level (critical/high/medium/low)
- Recommendation (approve/review/block)
- Prioritized action items
- Cross-cutting concerns identified"""

    def build_context(self) -> str:
        """QC Agent context is built dynamically from concerns."""
        return "QC Agent synthesizes concerns from Tier B agents."

    def can_analyze_changes(self) -> bool:
        """QC Agent doesn't analyze changes directly."""
        return False

    def synthesize_concerns(
        self,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
        all_concerns: list[Concern],
    ) -> ConcernReport:
        """Synthesize concerns from all agents into a QC report.

        Args:
            change_set: Original change set being analyzed.
            analysis_results: Results from each Tier B agent.
            all_concerns: All concerns raised by agents.

        Returns:
            Complete ConcernReport with synthesis and recommendations.
        """
        import time

        start_time = time.time()

        # If no concerns, return a clean report
        if not all_concerns:
            return self._create_clean_report(change_set, analysis_results)

        # Build prompt for synthesis
        prompt = self._build_synthesis_prompt(change_set, analysis_results, all_concerns)

        # Call LLM
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self.spec.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        synthesis_time = int((time.time() - start_time) * 1000)

        # Parse response
        report = self._parse_synthesis_response(
            response.content[0].text,
            change_set,
            analysis_results,
            all_concerns,
        )

        # Add synthesis tokens to total
        total_tokens = sum(r.tokens_used for r in analysis_results) + tokens_used
        report.total_tokens_used = total_tokens
        report.total_analysis_time_ms = synthesis_time

        return report

    def _create_clean_report(
        self,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
    ) -> ConcernReport:
        """Create a report when no concerns were found."""
        return ConcernReport(
            report_id=str(uuid.uuid4())[:8],
            change_set_id=change_set.change_id,
            total_concerns=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            concerns_by_category={},
            all_concerns=[],
            action_items=[],
            agents_consulted=[r.agent_id for r in analysis_results],
            analysis_results=analysis_results,
            total_analysis_time_ms=0,
            total_tokens_used=sum(r.tokens_used for r in analysis_results),
            overall_assessment="No concerns were identified in the analyzed changes. The code appears to follow best practices.",
            risk_level="low",
            recommendation="approve",
        )

    def _build_synthesis_prompt(
        self,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
        all_concerns: list[Concern],
    ) -> str:
        """Build prompt for concern synthesis."""
        # Format concerns by domain
        concerns_text = []
        for result in analysis_results:
            if result.concerns:
                domain_text = f"\n## {result.domain} ({result.agent_id})\n"
                for concern in result.concerns:
                    domain_text += f"""
### [{concern.severity.value.upper()}] {concern.title}
- Category: {concern.category.value}
- Affected: {', '.join(concern.affected_files[:3])}
- Description: {concern.description}
- Suggestion: {concern.suggestion or 'None provided'}
- Concern ID: {concern.concern_id}
"""
                concerns_text.append(domain_text)

        if not concerns_text:
            concerns_text = ["No concerns were raised by any agent."]

        # Count summary
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for c in all_concerns:
            severity_counts[c.severity.value] += 1

        return f"""## Change Set Analysis

**Change ID:** {change_set.change_id}
**Files Changed:** {len(change_set.files)}
**Agents Consulted:** {len(analysis_results)}
**Total Concerns:** {len(all_concerns)}

### Concern Summary
- Critical: {severity_counts['critical']}
- High: {severity_counts['high']}
- Medium: {severity_counts['medium']}
- Low: {severity_counts['low']}
- Info: {severity_counts['info']}

### Concerns by Domain
{chr(10).join(concerns_text)}

---

## Your Tasks

1. **Synthesize** the concerns above into a cohesive assessment
2. **Identify** any cross-cutting concerns (issues that affect multiple domains)
3. **Prioritize** create a ranked list of action items (use concern_id in related_concerns)
4. **Determine** overall risk level and recommendation

Respond in JSON format:
```json
{{
    "overall_assessment": "2-3 sentence summary...",
    "risk_level": "low|medium|high|critical",
    "recommendation": "approve|review|block",
    "cross_cutting_concerns": [
        {{
            "title": "...",
            "description": "...",
            "affected_domains": ["...", "..."]
        }}
    ],
    "action_items": [
        {{
            "priority": 1,
            "title": "...",
            "description": "...",
            "related_concerns": ["concern_id1", "concern_id2"],
            "estimated_effort": "small|medium|large"
        }}
    ]
}}
```"""

    def _parse_synthesis_response(
        self,
        response_text: str,
        change_set: ChangeSet,
        analysis_results: list[AgentAnalysisResult],
        all_concerns: list[Concern],
    ) -> ConcernReport:
        """Parse QC Agent's synthesis response."""
        # Try to extract JSON
        json_match = re.search(r"```json\n?(.*?)```", response_text, re.DOTALL)

        # Default values
        overall_assessment = "Analysis complete."
        risk_level = "low"
        recommendation = "approve"
        action_items = []

        if json_match:
            try:
                data = json.loads(json_match.group(1))
                overall_assessment = data.get("overall_assessment", overall_assessment)
                risk_level = data.get("risk_level", risk_level)
                recommendation = data.get("recommendation", recommendation)

                # Parse action items
                for i, item in enumerate(data.get("action_items", [])[:10]):
                    action_items.append(
                        ActionItem(
                            action_id=str(uuid.uuid4())[:8],
                            priority=item.get("priority", i + 1),
                            title=item.get("title", f"Action item {i + 1}"),
                            description=item.get("description", ""),
                            related_concerns=item.get("related_concerns", []),
                            estimated_effort=item.get("estimated_effort"),
                        )
                    )

            except json.JSONDecodeError:
                pass

        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for c in all_concerns:
            severity_counts[c.severity.value] += 1

        # Group by category
        by_category: dict[str, list[Concern]] = {}
        for concern in all_concerns:
            cat = concern.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(concern)

        # Sort concerns by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_concerns = sorted(
            all_concerns,
            key=lambda c: severity_order.get(c.severity.value, 5),
        )

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
            total_analysis_time_ms=0,  # Will be set by caller
            total_tokens_used=0,  # Will be set by caller
            overall_assessment=overall_assessment,
            risk_level=risk_level,
            recommendation=recommendation,
        )
