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

    # Regex patterns that indicate cross-cutting concerns
    CROSS_CUTTING_SIGNALS = [
        r"end\s+to\s+end",
        r"flow",
        r"how\s+does\s+.+\s+work",
        r"trace",
        r"lifecycle",
        r"from\s+.+\s+to\s+",
        r"across",
        r"between",
        r"relationship",
        r"architecture",
        r"overview",
        r"full\s+picture",
        r"entire",
        r"whole",
        r"complete",
        r"all\s+the\s+way",
        r"throughout",
        r"integration",
        r"interact",
        r"connect",
        r"communicate",
    ]

    def __init__(
        self,
        hub: "AgentHub",
        threshold: float = 0.4,
        min_agents_for_team: int = 2,
        max_agents_per_team: int = 6,
    ):
        """Initialize the classifier.

        Args:
            hub: AgentHub instance for access to agents and router.
            threshold: Minimum confidence score to trigger DAG execution.
                       Lower = more queries go to DAG.
            min_agents_for_team: Minimum matched agents to consider team execution.
            max_agents_per_team: Maximum agents to include in a team.
        """
        self.hub = hub
        self.threshold = threshold
        self.min_agents_for_team = min_agents_for_team
        self.max_agents_per_team = max_agents_per_team

        # Compile cross-cutting patterns
        self._cross_cutting_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CROSS_CUTTING_SIGNALS
        ]

    def classify(self, query: str) -> ComplexityResult:
        """Classify a query as simple or complex.

        Args:
            query: The user's query.

        Returns:
            ComplexityResult with classification details.
        """
        # Get all agent scores
        agent_scores = self._get_all_agent_scores(query)

        # Count agents with score > 0
        matched_agents = [
            agent_id
            for agent_id, score in agent_scores.items()
            if score > 0
        ]
        matched_count = len(matched_agents)

        # Sort matched agents by score (highest first)
        matched_agents.sort(key=lambda a: agent_scores.get(a, 0), reverse=True)

        # Cap at max_agents_per_team
        if len(matched_agents) > self.max_agents_per_team:
            matched_agents = matched_agents[: self.max_agents_per_team]

        # Check for cross-cutting signals
        has_cross_cutting = self._has_cross_cutting_signal(query)

        # Determine complexity
        is_complex = False
        trigger_reason = "simple"
        confidence = 0.0

        if matched_count <= 1:
            # Only one agent matches -> simple
            is_complex = False
            trigger_reason = "simple"
            confidence = 1.0 - (matched_count * 0.3)  # Higher confidence with fewer matches

        elif matched_count >= 3:
            # Three or more agents match -> complex
            is_complex = True
            trigger_reason = "multi_agent_match"
            # Confidence based on how many agents matched
            confidence = min(0.5 + (matched_count * 0.1), 1.0)

        elif matched_count == 2:
            # Two agents match -> depends on cross-cutting signals
            if has_cross_cutting:
                is_complex = True
                trigger_reason = "cross_cutting_signal"
                confidence = 0.6
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

    def _get_all_agent_scores(self, query: str) -> dict[str, int]:
        """Get keyword match scores for all agents.

        Args:
            query: The user's query.

        Returns:
            Dict mapping agent_id to match score.
        """
        from agenthub.routing import KeywordRouter

        # Get all agents from hub (list_agents returns list of AgentSpec)
        agents = self.hub.list_agents()

        # Use KeywordRouter's scoring logic
        router = KeywordRouter(case_sensitive=False)
        return router.get_all_scores(query, agents)

    def _has_cross_cutting_signal(self, query: str) -> bool:
        """Check if query contains cross-cutting signal words.

        Args:
            query: The user's query.

        Returns:
            True if any cross-cutting signal is found.
        """
        for pattern in self._cross_cutting_patterns:
            if pattern.search(query):
                return True
        return False
