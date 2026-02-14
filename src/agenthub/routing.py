from __future__ import annotations
"""Query routing strategies for AgentHub."""

import math
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
    """Keyword-based routing with whole-word matching and IDF weighting.

    Routes queries to agents based on keyword matching. Uses whole-word
    matching as the primary signal (full weight) and substring matching
    as a weak fallback (0.3x weight). Applies IDF weighting to penalize
    keywords shared across many agents.

    Example:
        >>> router = KeywordRouter()
        >>> agent_id = router.route("fix the database query", agents)
        >>> print(agent_id)  # "db_agent" if "database" is in its keywords
    """

    # Keywords that indicate query is about a specific domain - used for exclusion
    DOMAIN_EXCLUSION_KEYWORDS = {
        "frontend": ["react", "component", "tsx", "jsx", "css", "styled", "hook", "usestate", "useeffect", "redux"],
        "backend": ["api", "endpoint", "route", "controller", "fastapi", "flask", "django", "express"],
        "database": ["sql", "query", "migration", "schema", "model", "orm", "postgres", "mysql"],
        "test": ["test", "spec", "mock", "jest", "pytest", "unittest", "coverage"],
    }

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

    def get_all_scores(self, query: str, agents: list[AgentSpec]) -> dict[str, float]:
        """Get keyword match scores for ALL agents.

        Uses whole-word matching as the primary signal and substring matching
        as a weak fallback (0.3x weight). Applies IDF weighting so keywords
        shared across many agents contribute less than unique keywords.

        Respects per-agent routing settings: keyword_weights, exclusions,
        priority tiebreakers, and exact-match preferences.

        Args:
            query: The user's query.
            agents: List of available agent specs.

        Returns:
            Dict mapping agent_id -> score (0 if no match).
        """
        if not agents:
            return {}

        query_text = query if self.case_sensitive else query.lower()
        query_words = set(query_text.split())
        scores: dict[str, float] = {}

        # Pre-compute IDF for each keyword across all agents.
        # Keywords appearing in many agents get lower weight.
        keyword_doc_freq: dict[str, int] = {}
        for agent in agents:
            for keyword in agent.context_keywords:
                kw = keyword if self.case_sensitive else keyword.lower()
                keyword_doc_freq[kw] = keyword_doc_freq.get(kw, 0) + 1

        num_agents = len(agents)

        def idf(kw: str) -> float:
            df = keyword_doc_freq.get(kw, 1)
            # +1.0 floor so single-agent keywords score > 1.0
            return math.log(num_agents / df) + 1.0

        # Detect query domain for exclusion scoring
        query_domains = self._detect_query_domains(query_text)

        for agent in agents:
            score = 0.0
            routing = getattr(agent, "routing", None)
            has_declared_weights = routing and routing.keyword_weights

            for keyword in agent.context_keywords:
                keyword_text = keyword if self.case_sensitive else keyword.lower()
                idf_factor = idf(keyword_text)

                # Get base weight from declared weights or default 1.0
                if has_declared_weights and keyword_text in routing.keyword_weights:
                    base_weight = routing.keyword_weights[keyword_text]
                else:
                    base_weight = 1.0

                # PRIMARY: whole-word match (strongest signal)
                if keyword_text in query_words:
                    exact_bonus = 1.5 if (routing and routing.prefer_exact_match) else 1.0
                    score += (base_weight + exact_bonus) * idf_factor
                # SECONDARY: stemmed word match — keyword is a prefix/stem of a query word
                # (e.g., "tenant" matches "tenants", "tier" matches "tier-based")
                # Gets 0.8x weight since it's nearly as good as exact match
                elif any(qw.startswith(keyword_text) for qw in query_words if len(keyword_text) >= 3):
                    score += base_weight * 0.8 * idf_factor
                # TERTIARY: substring match (weak fallback, 0.3x weight)
                elif keyword_text in query_text:
                    score += base_weight * 0.3 * idf_factor

            # Apply domain exclusion penalty (heuristic-based)
            agent_domain = self._detect_agent_domain(agent)
            if query_domains and agent_domain and agent_domain not in query_domains:
                score = max(0, score - 5)

            # Apply per-agent exclusion penalty
            if routing and routing.exclusions:
                for excl in routing.exclusions:
                    if excl.lower() in query_text:
                        score = max(0, score - 5)
                        break

            # Apply priority tiebreaker
            if routing and routing.priority != 0:
                score += routing.priority * 0.1

            scores[agent.agent_id] = score

        return scores

    def _detect_query_domains(self, query: str) -> set[str]:
        """Detect which domain(s) a query is about."""
        domains = set()
        for domain, keywords in self.DOMAIN_EXCLUSION_KEYWORDS.items():
            matching = sum(1 for kw in keywords if kw in query)
            if matching >= 2:  # Strong signal
                domains.add(domain)
        return domains

    def _detect_agent_domain(self, agent: AgentSpec) -> Optional[str]:
        """Detect which domain an agent belongs to."""
        agent_name = agent.name.lower()
        agent_id = agent.agent_id.lower()

        for domain, keywords in self.DOMAIN_EXCLUSION_KEYWORDS.items():
            if domain in agent_name or domain in agent_id:
                return domain
            # Check if agent keywords overlap with domain keywords
            agent_kws = set(kw.lower() for kw in agent.context_keywords)
            domain_kws = set(keywords)
            if len(agent_kws & domain_kws) >= 2:
                return domain
        return None


class TierAwareRouter:
    """Routes queries considering both Tier A (business) and Tier B (code) agents.

    Tier A agents are manually created business/domain agents.
    Tier B agents are auto-generated code agents.

    Routing strategy:
    1. If query mentions specific files/paths → prefer Tier B
    2. If query mentions business terms → prefer Tier A
    3. If ambiguous → use tier preference setting
    """

    # Unambiguous code indicators — file extensions, path patterns, syntax tokens.
    # Removed ambiguous words: "file", "module", "function", "class", "import"
    # which also appear in business contexts.
    CODE_INDICATORS = [
        ".py",
        ".sql",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        "src/",
        "tests/",
        "lib/",
        "def ",
        "async ",
        "const ",
        "function(",
    ]

    # Strong business signals only — removed "user", "product", "order", "report"
    # which are too ambiguous (also common in code contexts).
    BUSINESS_INDICATORS = [
        "pricing",
        "customer",
        "revenue",
        "campaign",
        "strategy",
        "analytics",
        "marketing",
        "onboarding",
        "subscription",
        "churn",
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
        """Check if query is business/product focused.

        Requires 2+ indicator matches to avoid false positives from
        ambiguous terms.
        """
        matches = sum(1 for ind in self.BUSINESS_INDICATORS if ind in query)
        return matches >= 2


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
        model: str = "claude-opus-4-20250514",
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
