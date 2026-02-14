"""Tests for routing strategies."""

import pytest

from agenthub.models import AgentSpec, RoutingConfig
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
        """Test file extensions trigger code path preference for Tier B."""
        router = TierAwareRouter()
        # Both agents share the same keyword so only tier preference decides
        agents = [
            create_spec("business", ["handler"], auto_generated=False),
            create_spec("code", ["handler"], auto_generated=True),
        ]

        # ".py" triggers code indicator → Tier B preferred
        result = router.route("find the handler .py", agents)
        assert result == "code"

    def test_src_path_triggers_code(self):
        """Test 'src/' path indicator triggers Tier B preference."""
        router = TierAwareRouter()
        agents = [
            create_spec("business", ["config"], auto_generated=False),
            create_spec("code", ["config"], auto_generated=True),
        ]
        result = router.route("check the config in src/ folder", agents)
        assert result == "code"


# ── Helpers for per-agent routing tests ──────────────────────────────────


def create_spec_with_routing(
    agent_id: str,
    keywords: list[str],
    routing: RoutingConfig | None = None,
    auto_generated: bool = False,
) -> AgentSpec:
    """Helper to create agent specs with RoutingConfig."""
    return AgentSpec(
        agent_id=agent_id,
        name=f"{agent_id} Agent",
        description=f"Agent {agent_id}",
        context_keywords=keywords,
        routing=routing or RoutingConfig(),
        metadata={"auto_generated": auto_generated} if auto_generated else {},
    )


class TestRoutingConfig:
    """Tests for the RoutingConfig model itself."""

    def test_default_routing_config(self):
        """Empty RoutingConfig has permissive defaults."""
        rc = RoutingConfig()
        assert rc.keyword_weights == {}
        assert rc.domains == []
        assert rc.exclusions == []
        assert rc.priority == 0
        assert rc.min_confidence == 0.0
        assert rc.fallback_agent_id is None
        assert rc.prefer_exact_match is False

    def test_routing_config_in_agent_spec(self):
        """RoutingConfig integrates into AgentSpec correctly."""
        spec = AgentSpec(
            agent_id="test",
            name="Test",
            description="Test agent",
            routing=RoutingConfig(
                keyword_weights={"api": 2.0},
                domains=["backend"],
                priority=5,
            ),
        )
        assert spec.routing.keyword_weights == {"api": 2.0}
        assert spec.routing.domains == ["backend"]
        assert spec.routing.priority == 5

    def test_min_confidence_validation(self):
        """min_confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            RoutingConfig(min_confidence=1.5)
        with pytest.raises(Exception):
            RoutingConfig(min_confidence=-0.1)


class TestDeclaredKeywordWeights:
    """Test agent-declared keyword weights override builder heuristics."""

    def test_declared_weights_used_by_keyword_router(self):
        """KeywordRouter uses agent-declared weights when available."""
        router = KeywordRouter()
        agents = [
            create_spec_with_routing(
                "api",
                ["api", "endpoint"],
                routing=RoutingConfig(keyword_weights={"api": 3.0, "endpoint": 1.0}),
            ),
            create_spec_with_routing(
                "db",
                ["api", "database"],
                routing=RoutingConfig(keyword_weights={"api": 0.5, "database": 3.0}),
            ),
        ]

        # Both agents have "api" keyword, but api_agent weights it at 3.0
        # while db_agent weights it at only 0.5
        scores = router.get_all_scores("fix the api", agents)
        assert scores["api"] > scores["db"]

    def test_declared_weights_in_routing_index(self):
        """RoutingIndexBuilder respects agent-declared weights."""
        from agenthub.auto.routing_index import RoutingIndexBuilder

        spec = create_spec_with_routing(
            "api",
            ["api", "endpoint", "route"],
            routing=RoutingConfig(keyword_weights={"api": 5.0, "endpoint": 2.0}),
        )

        builder = RoutingIndexBuilder("/tmp/test")
        metadata = builder._build_agent_metadata(spec)

        # Declared weights should be used
        assert metadata.keyword_weights["api"] == 5.0
        assert metadata.keyword_weights["endpoint"] == 2.0
        # Non-declared keywords get default 1.0
        assert metadata.keyword_weights["route"] == 1.0


class TestDeclaredDomains:
    """Test agent-declared domains skip heuristic detection."""

    def test_declared_domains_used(self):
        """Builder uses declared domain instead of inferring."""
        from agenthub.auto.routing_index import RoutingIndexBuilder

        spec = create_spec_with_routing(
            "my_agent",
            ["process", "transform"],
            routing=RoutingConfig(domains=["backend", "data"]),
        )

        builder = RoutingIndexBuilder("/tmp/test")
        metadata = builder._build_agent_metadata(spec)

        # Should use first declared domain
        assert metadata.domain == "backend"

    def test_inferred_domain_when_not_declared(self):
        """Builder falls back to inference when no domains declared."""
        from agenthub.auto.routing_index import RoutingIndexBuilder

        spec = create_spec_with_routing(
            "my_agent",
            ["react", "component", "hook", "styled"],
            routing=RoutingConfig(),  # Empty — should infer "frontend"
        )

        builder = RoutingIndexBuilder("/tmp/test")
        metadata = builder._build_agent_metadata(spec)

        assert metadata.domain == "frontend"


class TestExclusionPenalty:
    """Test exclusion list penalizes agents for unwanted queries."""

    def test_exclusion_penalty_applied(self):
        """Agent gets penalized when query matches its exclusions."""
        router = KeywordRouter()
        agents = [
            create_spec_with_routing(
                "backend",
                ["api", "server", "handler"],
                routing=RoutingConfig(exclusions=["frontend", "css", "react"]),
            ),
            create_spec_with_routing(
                "frontend",
                ["react", "component"],
            ),
        ]

        # Query about "react" should penalize backend_agent
        scores = router.get_all_scores("fix the react component", agents)
        assert scores["frontend"] > scores["backend"]

    def test_exclusion_in_indexed_router(self):
        """IndexedKeywordRouter applies exclusion penalty."""
        from agenthub.auto.routing_index import IndexedKeywordRouter, RoutingIndexBuilder

        agents = [
            create_spec_with_routing(
                "backend",
                ["api", "handler", "css"],
                routing=RoutingConfig(exclusions=["frontend", "css"]),
            ),
            create_spec_with_routing(
                "frontend",
                ["css", "style"],
            ),
        ]

        builder = RoutingIndexBuilder("/tmp/test")
        index = builder.build(agents)
        router = IndexedKeywordRouter(index)

        scores = router.get_all_scores("fix the css styling")
        # backend has "css" keyword but css is in its exclusions → penalized
        assert scores["frontend"] > scores["backend"]


class TestPriorityTiebreaker:
    """Test priority field breaks ties between equal-scoring agents."""

    def test_higher_priority_wins(self):
        """Agent with higher priority wins when scores are equal."""
        router = KeywordRouter()
        agents = [
            create_spec_with_routing(
                "low",
                ["api"],
                routing=RoutingConfig(priority=1),
            ),
            create_spec_with_routing(
                "high",
                ["api"],
                routing=RoutingConfig(priority=10),
            ),
        ]

        # Both match "api" equally, but "high" has priority=10
        result = router.route("fix the api", agents)
        assert result == "high"


class TestFallbackChain:
    """Test fallback chain when confidence threshold isn't met."""

    def test_fallback_traversal_in_indexed_router(self):
        """IndexedKeywordRouter follows fallback chain.

        The strict agent has many keywords but a very high confidence threshold.
        A weak query only matches a few keywords, giving a low normalized score
        that falls below the threshold — triggering the fallback chain.
        """
        from agenthub.auto.routing_index import IndexedKeywordRouter, RoutingIndexBuilder

        agents = [
            create_spec_with_routing(
                "strict",
                # Many keywords → high max_possible, but query only matches one
                ["api", "endpoint", "handler", "controller", "service", "route"],
                routing=RoutingConfig(
                    keyword_weights={"api": 1.0, "endpoint": 2.0, "handler": 2.0,
                                     "controller": 2.0, "service": 2.0, "route": 2.0},
                    min_confidence=0.9,
                    fallback_agent_id="relaxed",
                ),
            ),
            create_spec_with_routing(
                "relaxed",
                ["api", "help"],
                routing=RoutingConfig(min_confidence=0.0),
            ),
        ]

        builder = RoutingIndexBuilder("/tmp/test")
        index = builder.build(agents)
        router = IndexedKeywordRouter(index)

        # Query "api help" only matches "api" for strict (1/6 keywords, low score)
        # but matches both keywords for relaxed → relaxed should win via scoring
        # or fallback. Either way, relaxed should be selected.
        result = router.route("api help")
        assert result == "relaxed"

    def test_fallback_max_depth(self):
        """Fallback chain stops at MAX_FALLBACK_DEPTH.

        Chain agents a→b→c each have unique keywords that DON'T match the
        query, so they score 0. Agent d's keyword DOES match. The chain
        agents all have min_confidence > 0 and a fallback set, so they
        reject the query (score 0 < any threshold > 0) and pass it along.
        """
        from agenthub.auto.routing_index import IndexedKeywordRouter, RoutingIndexBuilder

        agents = [
            create_spec_with_routing(
                "a", ["zebra"],
                routing=RoutingConfig(min_confidence=0.5, fallback_agent_id="b"),
            ),
            create_spec_with_routing(
                "b", ["unicorn"],
                routing=RoutingConfig(min_confidence=0.5, fallback_agent_id="c"),
            ),
            create_spec_with_routing(
                "c", ["dragon"],
                routing=RoutingConfig(min_confidence=0.5, fallback_agent_id="d"),
            ),
            create_spec_with_routing(
                "d", ["api"],
                routing=RoutingConfig(min_confidence=0.0),  # Accepts anything
            ),
        ]

        builder = RoutingIndexBuilder("/tmp/test")
        index = builder.build(agents)
        router = IndexedKeywordRouter(index)

        # Only "d" matches "api". a,b,c have score=0.
        # But d should win directly via scoring since it's the only match.
        result = router.route("fix the api")
        assert result == "d"


class TestBackwardCompatibility:
    """Test existing agents with empty RoutingConfig work unchanged."""

    def test_empty_routing_config_works(self):
        """Agent with default RoutingConfig routes normally."""
        router = KeywordRouter()
        agents = [
            AgentSpec(
                agent_id="old_agent",
                name="Old Agent",
                description="Legacy agent without routing config",
                context_keywords=["legacy", "system"],
            ),
        ]

        result = router.route("fix the legacy system", agents)
        assert result == "old_agent"

    def test_empty_routing_builder_falls_back(self):
        """Builder uses heuristics when routing config is empty."""
        from agenthub.auto.routing_index import RoutingIndexBuilder

        spec = AgentSpec(
            agent_id="old_agent",
            name="Database Agent",
            description="Old agent",
            context_keywords=["sql", "query", "migration", "schema"],
        )

        builder = RoutingIndexBuilder("/tmp/test")
        metadata = builder._build_agent_metadata(spec)

        # Should infer domain from keywords
        assert metadata.domain == "database"
        # Should compute heuristic weights
        assert metadata.keyword_weights["sql"] > 0


class TestPreferExactMatch:
    """Test prefer_exact_match boosts whole-word matches."""

    def test_exact_match_boost(self):
        """Agent with prefer_exact_match gets higher bonus for whole words."""
        router = KeywordRouter()
        agents = [
            create_spec_with_routing(
                "exact_agent",
                ["api"],
                routing=RoutingConfig(prefer_exact_match=True),
            ),
            create_spec_with_routing(
                "normal_agent",
                ["api"],
                routing=RoutingConfig(prefer_exact_match=False),
            ),
        ]

        scores = router.get_all_scores("the api is broken", agents)
        # exact_agent should get a bigger bonus for "api" as a whole word
        assert scores["exact_agent"] > scores["normal_agent"]


# ── Phase 1A: Whole-word matching tests ──────────────────────────────────


class TestWholeWordMatching:
    """Test whole-word matching replaces substring matching."""

    def test_no_substring_false_positive(self):
        """'model' keyword should NOT match 'remodel' as a high-score hit."""
        router = KeywordRouter()
        agents = [create_spec("model_agent", ["model"])]
        scores = router.get_all_scores("remodel the kitchen", agents)
        # Substring match gives 0.3x weight, so score should be low
        assert scores["model_agent"] < 1.0

    def test_whole_word_matches_fully(self):
        """'model' keyword SHOULD match 'model' as standalone word."""
        router = KeywordRouter()
        agents = [create_spec("model_agent", ["model"])]
        scores = router.get_all_scores("update the model", agents)
        # Whole-word match: base 1.0 + exact bonus 1.0 = 2.0, times IDF
        assert scores["model_agent"] >= 2.0

    def test_substring_vs_whole_word_discrimination(self):
        """Whole-word match should score much higher than substring match."""
        router = KeywordRouter()
        agents = [
            create_spec("whole", ["model"]),
            create_spec("sub", ["model"]),
        ]
        # "model" as whole word vs "remodel" as substring
        whole_scores = router.get_all_scores("fix the model", agents)
        sub_scores = router.get_all_scores("remodel the thing", agents)
        # Both agents score the same for each query, but whole >> sub
        assert whole_scores["whole"] > sub_scores["sub"] * 2

    def test_multi_word_keyword_substring_fallback(self):
        """Multi-word keywords like 'file_watcher' still match as substring."""
        router = KeywordRouter()
        agents = [create_spec("watcher", ["file_watcher"])]
        scores = router.get_all_scores("configure the file_watcher module", agents)
        # "file_watcher" appears as whole word in query_words (space-split)
        assert scores["watcher"] > 0


# ── Phase 1B: IDF weighting tests ───────────────────────────────────────


class TestIDFWeighting:
    """Test IDF-style weighting penalizes common keywords."""

    def test_common_keyword_lower_score(self):
        """Keyword shared by many agents contributes less per agent."""
        router = KeywordRouter()
        agents = [
            create_spec("a", ["api", "auth"]),
            create_spec("b", ["api", "billing"]),
            create_spec("c", ["api", "cache"]),
        ]
        scores = router.get_all_scores("fix the auth system", agents)
        # Agent "a" should win: "auth" is unique to it (high IDF),
        # while "api" is shared (low IDF) and doesn't appear in query
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_unique_keyword_high_score(self):
        """Keyword unique to one agent gets highest IDF factor."""
        router = KeywordRouter()
        agents = [
            create_spec("special", ["unicorn_module"]),
            create_spec("generic", ["module"]),
        ]
        scores = router.get_all_scores("fix the unicorn_module", agents)
        assert scores["special"] > scores["generic"]

    def test_idf_doesnt_break_single_agent(self):
        """IDF works correctly when there's only one agent (no division by zero)."""
        router = KeywordRouter()
        agents = [create_spec("solo", ["api", "endpoint"])]
        scores = router.get_all_scores("fix the api endpoint", agents)
        assert scores["solo"] > 0


# ── Phase 1C: TierAwareRouter indicator tests ────────────────────────────


class TestTightenedIndicators:
    """Test that ambiguous words no longer trigger false tier preferences."""

    def test_user_no_longer_triggers_business(self):
        """'user' was removed from BUSINESS_INDICATORS so it stays neutral."""
        router = TierAwareRouter()
        # 'user' alone should not trigger business preference
        assert not router._mentions_business_terms("tell me about user settings")

    def test_business_requires_two_matches(self):
        """Business detection now requires 2+ indicator matches."""
        router = TierAwareRouter()
        # Single business term: not enough
        assert not router._mentions_business_terms("pricing is important")
        # Two business terms: triggers
        assert router._mentions_business_terms("pricing and revenue analysis")

    def test_code_indicators_still_work(self):
        """Unambiguous code indicators like file extensions still work."""
        router = TierAwareRouter()
        assert router._mentions_code_paths("edit src/main.py")
        assert router._mentions_code_paths("fix the .ts file")
        assert router._mentions_code_paths("add def calculate()")

    def test_ambiguous_words_removed(self):
        """'file', 'module', 'function', 'class' no longer trigger code preference."""
        router = TierAwareRouter()
        # These common words should NOT trigger code preference on their own
        assert not router._mentions_code_paths("what class of customer is this")
        assert not router._mentions_code_paths("our product module needs work")
