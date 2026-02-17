from __future__ import annotations
"""Complexity classification for query routing.

This module determines whether a query should be handled by a single agent
(simple) or requires multi-agent collaboration (complex).
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthub.hub import AgentHub


@dataclass
class ComplexityResult:
    """Result of complexity classification."""

    is_complex: bool
    matched_agents: list[str] = field(default_factory=list)  # agent_ids with score > 0
    agent_scores: dict[str, int] = field(default_factory=dict)  # agent_id -> score
    confidence: float = 0.0  # 0.0 to 1.0
    trigger_reason: str = "simple"  # "multi_agent_match" | "cross_cutting_signal" | "simple"


class ComplexityClassifier:
    """Classifies query complexity to decide single-agent vs DAG team execution.

    Classification logic:
    1. Run keyword scoring against ALL agents
    2. Count how many agents scored > 0 (matched_count)
    3. Check for cross-cutting signal words in the query
    4. Decision:
       - matched_count <= 1 -> simple (single agent)
       - matched_count >= 3 -> complex (DAG team)
       - matched_count == 2 AND cross-cutting signal present -> complex

    Example:
        >>> classifier = ComplexityClassifier(hub, threshold=0.4)
        >>> result = classifier.classify("how does data flow from API to database?")
        >>> if result.is_complex:
        ...     # Use DAG team execution
        ...     pass
    """

    # Regex patterns that indicate cross-cutting concerns.
    # Split into two tiers:
    #  - STRONG: explicitly asks about relationships/comparisons between domains.
    #    Triggers Tier A expansion (add all domain experts to team).
    #  - WEAK: indicates complexity but not necessarily cross-domain.
    #    Lowers keyword threshold but does NOT expand the team.
    CROSS_CUTTING_SIGNALS_STRONG = [
        r"end\s+to\s+end",
        r"from\s+.+\s+to\s+",
        r"across",
        r"between",
        r"relationship",
        r"full\s+picture",
        r"all\s+the\s+way",
        r"throughout",
        # Comparison / impact signals — explicitly multi-domain
        r"difference",
        r"differ",
        r"compare",
        r"affect",
        r"impact",
        r"influence",
    ]

    CROSS_CUTTING_SIGNALS_WEAK = [
        r"flow",
        r"how\s+does\s+.+\s+work",
        r"trace",
        r"lifecycle",
        r"architecture",
        r"overview",
        r"entire",
        r"whole",
        r"complete",
        r"integration",
        r"interact",
        r"connect",
        r"communicate",
    ]

    # Combined for backward compatibility
    CROSS_CUTTING_SIGNALS = CROSS_CUTTING_SIGNALS_STRONG + CROSS_CUTTING_SIGNALS_WEAK

    def __init__(
        self,
        hub: "AgentHub",
        threshold: float = 0.6,  # Increased from 0.5 to be more conservative
        min_agents_for_team: int = 2,
        max_agents_per_team: int = 5,
        min_score_to_match: int = 2,
        min_total_score_for_team: int = 6,  # New: require total score >= 6 across agents
    ):
        """Initialize the classifier.

        Args:
            hub: AgentHub instance for access to agents and router.
            threshold: Minimum confidence score to trigger DAG execution.
                       Lower = more queries go to DAG. Default 0.6.
            min_agents_for_team: Minimum matched agents to consider team execution.
                       Default 2 (require at least 2 distinct agents).
            max_agents_per_team: Maximum agents to include in a team.
            min_score_to_match: Minimum keyword score to consider an agent "matched".
                       Default 2 (require at least 2 keyword matches).
            min_total_score_for_team: Minimum combined score across all matched agents.
                       Prevents team execution when agents barely qualify.
                       Default 6 (e.g., two agents with score 3 each).
        """
        self.hub = hub
        self.threshold = threshold
        self.min_agents_for_team = min_agents_for_team
        self.max_agents_per_team = max_agents_per_team
        self.min_score_to_match = min_score_to_match
        self.min_total_score_for_team = min_total_score_for_team

        # Compile cross-cutting patterns (both tiers)
        self._cross_cutting_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CROSS_CUTTING_SIGNALS
        ]
        self._strong_cross_cutting_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CROSS_CUTTING_SIGNALS_STRONG
        ]

    def classify(self, query: str) -> ComplexityResult:
        """Classify a query as simple or complex.

        Args:
            query: The user's query.

        Returns:
            ComplexityResult with classification details.
        """
        # Get all agent scores (excluding sub-agents from team consideration)
        agent_scores = self._get_all_agent_scores(query, exclude_sub_agents=True)

        # Check for cross-cutting signals early — affects matching threshold
        has_cross_cutting = self._has_cross_cutting_signal(query)

        # When cross-cutting signals are present, lower the threshold so
        # marginally relevant agents are included.  Cross-domain questions
        # like "difference between Tier A and Tier B tenants" need agents
        # that each cover part of the answer, even if they only match on
        # one keyword.
        effective_min_score = self.min_score_to_match
        if has_cross_cutting:
            effective_min_score = max(1, self.min_score_to_match - 1)

        # Count agents with score >= effective threshold
        matched_agents = [
            agent_id
            for agent_id, score in agent_scores.items()
            if score >= effective_min_score
        ]
        matched_count = len(matched_agents)

        # Calculate total score across matched agents
        total_score = sum(agent_scores.get(a, 0) for a in matched_agents)

        # Sort matched agents by score (highest first)
        matched_agents.sort(key=lambda a: agent_scores.get(a, 0), reverse=True)

        # Only expand with Tier A domain experts for STRONG cross-cutting
        # signals (comparisons, relationships, "from X to Y").  Weak signals
        # like "how does X work" just lower the keyword threshold above.
        has_strong_cross_cutting = self._has_strong_cross_cutting_signal(query)
        if has_strong_cross_cutting and matched_agents:
            matched_agents = self._expand_with_tier_a_agents(
                matched_agents, agent_scores,
            )
            matched_count = len(matched_agents)
            total_score = sum(agent_scores.get(a, 0) for a in matched_agents)

        # Cap at max_agents_per_team
        if len(matched_agents) > self.max_agents_per_team:
            matched_agents = matched_agents[: self.max_agents_per_team]

        # Determine complexity
        is_complex = False
        trigger_reason = "simple"
        confidence = 0.0

        if matched_count < self.min_agents_for_team:
            # Not enough agents match -> simple (single agent)
            is_complex = False
            trigger_reason = "simple"
            confidence = 1.0 - (matched_count * 0.2)

        elif total_score < self.min_total_score_for_team and not has_cross_cutting:
            # Agents barely qualify - not enough combined relevance
            # This prevents team execution when multiple agents have low scores.
            # Exception: cross-cutting queries bypass this check because they
            # naturally include agents with lower individual scores that each
            # cover a different facet of the answer.
            is_complex = False
            trigger_reason = "low_total_score"
            confidence = 0.3

        elif matched_count >= self.min_agents_for_team + 2:
            # 4+ strong matches -> definitely complex (team execution)
            is_complex = True
            trigger_reason = "multi_agent_match"
            # Higher confidence for more matches
            confidence = min(0.6 + (matched_count * 0.1), 0.95)

        elif matched_count == self.min_agents_for_team:
            # Exactly 2 agents match -> team ONLY if cross-cutting signal present
            # AND agents have reasonable scores.
            # With cross-cutting signals we lower the bar: at least one agent
            # must score >= 3 (the "anchor") so we're not teaming on noise.
            top_two_scores = sorted([agent_scores.get(a, 0) for a in matched_agents], reverse=True)[:2]

            if has_cross_cutting and top_two_scores[0] >= 3:
                is_complex = True
                trigger_reason = "cross_cutting_signal"
                confidence = 0.65
            elif not has_cross_cutting and all(s >= 3 for s in top_two_scores):
                is_complex = True
                trigger_reason = "cross_cutting_signal"
                confidence = 0.65
            else:
                is_complex = False
                trigger_reason = "simple"
                confidence = 0.45

        elif matched_count == self.min_agents_for_team + 1:
            # 3 agents match -> complex only if cross-cutting OR very strong scores
            avg_score = total_score / matched_count
            if has_cross_cutting or avg_score >= 4:
                is_complex = True
                trigger_reason = "multi_agent_match"
                confidence = 0.7 if has_cross_cutting else 0.6
            else:
                is_complex = False
                trigger_reason = "simple"
                confidence = 0.5

        # Apply threshold check
        if is_complex and confidence < self.threshold:
            is_complex = False
            trigger_reason = "below_threshold"

        return ComplexityResult(
            is_complex=is_complex,
            matched_agents=matched_agents,
            agent_scores=agent_scores,
            confidence=confidence,
            trigger_reason=trigger_reason,
        )

    def _get_all_agent_scores(
        self, query: str, exclude_sub_agents: bool = False
    ) -> dict[str, int]:
        """Get keyword match scores for all agents.

        Args:
            query: The user's query.
            exclude_sub_agents: If True, exclude sub-agents from scoring.
                Sub-agents should be delegated by their parent, not directly matched.

        Returns:
            Dict mapping agent_id to match score.
        """
        from agenthub.routing import KeywordRouter

        # Get all agents from hub (list_agents returns list of AgentSpec)
        agents = self.hub.list_agents()

        # Optionally filter out sub-agents (they're managed by their parent agent)
        if exclude_sub_agents:
            agents = [a for a in agents if not a.metadata.get("is_sub_agent", False)]

        # Use KeywordRouter's scoring logic with KG boost if available
        kg = getattr(self.hub, "_knowledge_graph", None)
        router = KeywordRouter(case_sensitive=False, knowledge_graph=kg)
        return router.get_all_scores(query, agents)

    def _has_cross_cutting_signal(self, query: str) -> bool:
        """Check if query contains any cross-cutting signal (strong or weak).

        Args:
            query: The user's query.

        Returns:
            True if any cross-cutting signal is found.
        """
        for pattern in self._cross_cutting_patterns:
            if pattern.search(query):
                return True
        return False

    def _has_strong_cross_cutting_signal(self, query: str) -> bool:
        """Check if query contains a STRONG cross-cutting signal.

        Strong signals explicitly indicate multi-domain questions:
        comparisons ("difference between"), causal chains ("affect",
        "impact"), traversals ("from X to Y", "end to end").

        Args:
            query: The user's query.

        Returns:
            True if a strong cross-cutting signal is found.
        """
        for pattern in self._strong_cross_cutting_patterns:
            if pattern.search(query):
                return True
        return False

    def _expand_with_tier_a_agents(
        self,
        matched_agents: list[str],
        agent_scores: dict[str, float],
    ) -> list[str]:
        """Expand matched agents with Tier A domain experts.

        For strong cross-cutting queries, Tier A agents that weren't matched
        by keywords may still cover crucial parts of the answer.  For example,
        "difference between Tier A and Tier B tenants" matches
        ``database_agent`` (via "tenant", "tier") but needs
        ``naver_api_agent`` (owns ``ingestion.py`` which guards tier
        features) even though it has no keyword overlap.

        Strategy: include all Tier A agents that have context_paths.  The
        team decomposer will generate focused sub-questions for each agent,
        and the synthesizer will ignore agents with nothing to contribute.
        Token budgets (15K per agent in team mode) prevent cost explosion.

        Only Tier A agents are expanded — Tier B (auto-generated) agents
        are too numerous and would dilute the team.

        Args:
            matched_agents: Already matched agent IDs (sorted by score).
            agent_scores: All agent scores from keyword routing.

        Returns:
            Expanded list of agent IDs, sorted by score (highest first).
        """
        matched_set = set(matched_agents)

        # Add Tier A agents that have context_paths (they're domain experts)
        candidates: list[tuple[str, float]] = []
        for spec in self.hub.list_agents():
            if spec.agent_id in matched_set:
                continue
            if spec.metadata.get("is_sub_agent"):
                continue
            # Only expand with Tier A agents (hand-crafted domain experts)
            tier = spec.metadata.get("tier")
            if tier != "A":
                continue
            if not spec.context_paths:
                continue

            candidates.append((spec.agent_id, agent_scores.get(spec.agent_id, 0)))

        if not candidates:
            return matched_agents

        # Sort by keyword score — agents with higher relevance first
        candidates.sort(key=lambda x: x[1], reverse=True)
        expanded = list(matched_agents)
        for agent_id, _ in candidates:
            if len(expanded) < self.max_agents_per_team:
                expanded.append(agent_id)

        # Re-sort by score
        expanded.sort(key=lambda a: agent_scores.get(a, 0), reverse=True)
        return expanded
