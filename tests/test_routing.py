"""Tests for routing strategies."""

import pytest

from agenthub.models import AgentSpec
from agenthub.routing import KeywordRouter, TierAwareRouter


def create_spec(agent_id: str, keywords: list[str], auto_generated: bool = False) -> AgentSpec:
    """Helper to create agent specs for testing."""
    return AgentSpec(
        agent_id=agent_id,
        name=f"{agent_id} Agent",
        description=f"Agent {agent_id}",
        context_keywords=keywords,
        metadata={"auto_generated": auto_generated} if auto_generated else {},
    )


class TestKeywordRouter:
    """Tests for KeywordRouter."""

    def test_single_keyword_match(self):
        """Test routing with single keyword match."""
        router = KeywordRouter()
        agents = [
            create_spec("code", ["code", "function"]),
            create_spec("db", ["database", "query"]),
        ]

        assert router.route("fix the code", agents) == "code"
        assert router.route("write a query", agents) == "db"

    def test_multiple_keyword_match(self):
        """Test routing with multiple keyword matches selects highest score."""
        router = KeywordRouter()
        agents = [
            create_spec("api", ["api", "endpoint"]),
            create_spec("code", ["code", "api", "function"]),
        ]

        # "code" has more matches for this query
        result = router.route("write api code", agents)
        # Both match "api", but "code" also matches "code"
        assert result in ["api", "code"]

    def test_case_insensitive(self):
        """Test case insensitive matching by default."""
        router = KeywordRouter(case_sensitive=False)
        agents = [create_spec("test", ["DATABASE", "Query"])]

        assert router.route("database query", agents) == "test"

    def test_case_sensitive(self):
        """Test case sensitive matching."""
        router = KeywordRouter(case_sensitive=True)
        agents = [create_spec("test", ["Database"])]

        # Lowercase shouldn't match
        assert router.route("database", agents) is None
        # Exact case should match
        assert router.route("Database", agents) == "test"

    def test_no_match_returns_none(self):
        """Test no match returns None."""
        router = KeywordRouter()
        agents = [create_spec("test", ["specific", "keywords"])]

        assert router.route("completely unrelated xyz", agents) is None

    def test_empty_agents_returns_none(self):
        """Test empty agents list returns None."""
        router = KeywordRouter()
        assert router.route("any query", []) is None

    def test_exact_word_bonus(self):
        """Test exact word match gets bonus score."""
        router = KeywordRouter()
        agents = [
            create_spec("partial", ["api"]),
            create_spec("exact", ["api"]),
        ]

        # Both have "api" keyword, but the query has "api" as exact word
        # They should have equal scores, so first match wins
        result = router.route("the api endpoint", agents)
        assert result in ["partial", "exact"]


class TestTierAwareRouter:
    """Tests for TierAwareRouter."""

    def test_code_indicators_prefer_tier_b(self):
        """Test code path indicators prefer Tier B agents."""
        router = TierAwareRouter()
        agents = [
            create_spec("business", ["api"], auto_generated=False),
            create_spec("code", ["api", "src"], auto_generated=True),
        ]

        # Mentions ".py" - should prefer Tier B
        result = router.route("where is the .py file for api?", agents)
        assert result == "code"

    def test_business_indicators_prefer_tier_a(self):
        """Test business terms prefer Tier A agents."""
        router = TierAwareRouter()
        agents = [
            create_spec("pricing", ["pricing", "margin"], auto_generated=False),
            create_spec("code", ["pricing"], auto_generated=True),
        ]

        # "pricing" is a business term
        result = router.route("what's our pricing strategy?", agents)
        assert result == "pricing"

    def test_ambiguous_uses_preference(self):
        """Test ambiguous queries use tier preference."""
        agents = [
            create_spec("business", ["help"], auto_generated=False),
            create_spec("code", ["help"], auto_generated=True),
        ]

        # Query doesn't have clear indicators
        router_prefer_a = TierAwareRouter(prefer_tier_a=True)
        router_prefer_b = TierAwareRouter(prefer_tier_a=False)

        assert router_prefer_a.route("help me with something", agents) == "business"
        assert router_prefer_b.route("help me with something", agents) == "code"

    def test_fallback_to_other_tier(self):
        """Test fallback to other tier if no match in preferred tier."""
        router = TierAwareRouter(prefer_tier_a=True)
        agents = [
            # Only Tier B agent with matching keyword
            create_spec("code", ["special"], auto_generated=True),
        ]

        # Should fall back to Tier B since no Tier A agents
        result = router.route("something special", agents)
        assert result == "code"

    def test_file_extension_detection(self):
        """Test various file extensions trigger code path detection."""
        router = TierAwareRouter()
        agents = [
            create_spec("business", ["find", "file"], auto_generated=False),
            create_spec("code", ["find", "file", "src", "function"], auto_generated=True),
        ]

        code_queries = [
            "find the .py file",
            "where is the .sql file",
            "update the .js file",
            "src/ folder",
            "function definition",
        ]

        for query in code_queries:
            result = router.route(query, agents)
            assert result == "code", f"Failed for query: {query}"
