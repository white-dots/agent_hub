"""Query routing strategies for AgentHub."""

from typing import TYPE_CHECKING, Optional, Protocol

from agenthub.models import AgentSpec

if TYPE_CHECKING:
    import anthropic


class Router(Protocol):
    """Protocol for router implementations."""

    def route(self, query: str, agents: list[AgentSpec]) -> Optional[str]:
        """Route a query to an agent.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Agent ID to route to, or None if no match.
        """
        ...


class KeywordRouter:
    """Simple keyword-based routing.

    Routes queries to agents based on keyword matching. Fast and
    deterministic, suitable for most use cases.

    Example:
        >>> router = KeywordRouter()
        >>> agent_id = router.route("fix the database query", agents)
        >>> print(agent_id)  # "db_agent" if "database" is in its keywords
    """

    def __init__(self, case_sensitive: bool = False):
        """Initialize KeywordRouter.

        Args:
            case_sensitive: Whether keyword matching is case-sensitive.
        """
        self.case_sensitive = case_sensitive

    def route(self, query: str, agents: list[AgentSpec]) -> Optional[str]:
        """Route query based on keyword matching.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Agent ID of best match, or None if no match.
        """
        scores = self.get_all_scores(query, agents)
        if scores:
            # Return agent with highest score
            best = max(scores.items(), key=lambda x: x[1])
            if best[1] > 0:
                return best[0]
        return None

    def get_all_scores(self, query: str, agents: list[AgentSpec]) -> dict[str, int]:
        """Get keyword match scores for ALL agents.

        This is useful for determining how many agents are relevant to a query,
        which helps decide whether to use single-agent or multi-agent execution.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Dict mapping agent_id -> score (0 if no match).
        """
        if not agents:
            return {}

        query_text = query if self.case_sensitive else query.lower()
        scores: dict[str, int] = {}

        for agent in agents:
            score = 0
            for keyword in agent.context_keywords:
                keyword_text = keyword if self.case_sensitive else keyword.lower()
                if keyword_text in query_text:
                    score += 1
                    # Bonus for exact word match
                    if f" {keyword_text} " in f" {query_text} ":
                        score += 1

            scores[agent.agent_id] = score

        return scores


class TierAwareRouter:
    """Routes queries considering both Tier A (business) and Tier B (code) agents.

    Tier A agents are manually created business/domain agents.
    Tier B agents are auto-generated code agents.

    Routing strategy:
    1. If query mentions specific files/paths → prefer Tier B
    2. If query mentions business terms → prefer Tier A
    3. If ambiguous → use tier preference setting
    """

    CODE_INDICATORS = [
        ".py",
        ".sql",
        ".js",
        ".ts",
        "file",
        "folder",
        "module",
        "function",
        "class",
        "import",
        "src/",
        "tests/",
        "def ",
        "async ",
    ]

    BUSINESS_INDICATORS = [
        "pricing",
        "customer",
        "order",
        "revenue",
        "campaign",
        "strategy",
        "report",
        "analytics",
        "user",
        "product",
    ]

    def __init__(self, prefer_tier_a: bool = True, case_sensitive: bool = False):
        """Initialize TierAwareRouter.

        Args:
            prefer_tier_a: If True, prefer business agents for ambiguous queries.
            case_sensitive: Whether matching is case-sensitive.
        """
        self.prefer_tier_a = prefer_tier_a
        self.case_sensitive = case_sensitive
        self._keyword_router = KeywordRouter(case_sensitive=case_sensitive)

    def route(self, query: str, agents: list[AgentSpec]) -> Optional[str]:
        """Route with tier awareness.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Agent ID to route to, or None.
        """
        if not agents:
            return None

        # Split agents by tier
        tier_a = [a for a in agents if not a.metadata.get("auto_generated")]
        tier_b = [a for a in agents if a.metadata.get("auto_generated")]

        query_lower = query.lower()

        # Check for code path indicators → Tier B
        if self._mentions_code_paths(query_lower):
            result = self._keyword_router.route(query, tier_b)
            if result:
                return result
            return self._keyword_router.route(query, tier_a)

        # Check for business terms → Tier A
        if self._mentions_business_terms(query_lower):
            result = self._keyword_router.route(query, tier_a)
            if result:
                return result
            return self._keyword_router.route(query, tier_b)

        # Ambiguous - use preference
        if self.prefer_tier_a:
            result = self._keyword_router.route(query, tier_a)
            if result:
                return result
            return self._keyword_router.route(query, tier_b)
        else:
            result = self._keyword_router.route(query, tier_b)
            if result:
                return result
            return self._keyword_router.route(query, tier_a)

    def _mentions_code_paths(self, query: str) -> bool:
        """Check if query mentions file paths, extensions, etc."""
        return any(ind in query for ind in self.CODE_INDICATORS)

    def _mentions_business_terms(self, query: str) -> bool:
        """Check if query is business/product focused."""
        return any(ind in query for ind in self.BUSINESS_INDICATORS)


class LLMRouter:
    """LLM-based routing using Claude Haiku for fast classification.

    Use this when keyword routing isn't sufficient and you need
    semantic understanding of queries.

    Example:
        >>> router = LLMRouter(client)
        >>> agent_id = router.route("how do we handle edge cases?", agents)
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        model: str = "claude-haiku-3-5-20241022",
    ):
        """Initialize LLMRouter.

        Args:
            client: Anthropic client for API calls.
            model: Model to use for routing (default: Haiku for speed).
        """
        self.client = client
        self.model = model

    def route(self, query: str, agents: list[AgentSpec]) -> Optional[str]:
        """Route query using LLM classification.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Agent ID from LLM decision, or None.
        """
        if not agents:
            return None

        # Build agent descriptions
        agent_info = "\n".join(
            [f"- {a.agent_id}: {a.description}" for a in agents]
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Given this query, which agent should handle it?

Query: {query}

Available agents:
{agent_info}

Respond with ONLY the agent_id, nothing else.""",
                    }
                ],
            )

            agent_id = response.content[0].text.strip()

            # Validate
            valid_ids = [a.agent_id for a in agents]
            if agent_id in valid_ids:
                return agent_id

            # Try to fuzzy match
            for valid_id in valid_ids:
                if valid_id in agent_id or agent_id in valid_id:
                    return valid_id

            return None

        except Exception:
            # Fall back to keyword routing on error
            return KeywordRouter().route(query, agents)


def create_llm_router(client: "anthropic.Anthropic") -> LLMRouter:
    """Create an LLM-based router.

    Convenience function for creating an LLM router with default settings.

    Args:
        client: Anthropic client for API calls.

    Returns:
        Configured LLMRouter instance.
    """
    return LLMRouter(client)
