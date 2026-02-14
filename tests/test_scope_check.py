"""Tests for enhanced heuristic scope check and LLM pre-screen."""

from unittest.mock import MagicMock, patch

import pytest

from agenthub.agents.base import BaseAgent, heuristic_scope_check
from agenthub.models import AgentSpec, RoutingConfig


# ── Heuristic scope check tests ─────────────────────────────────────────


class TestHeuristicScopeCheck:
    """Test the shared heuristic_scope_check function."""

    def test_zero_keyword_overlap_rejects(self):
        """Query with zero overlap with agent keywords is rejected."""
        spec = AgentSpec(
            agent_id="db_agent",
            name="Database Expert",
            description="Database agent",
            context_keywords=["sql", "query", "migration", "schema"],
        )
        result = heuristic_scope_check(spec, "how does the react component render")
        assert result["in_scope"] is False
        assert "sql" in result["message"] or "domain" in result["message"].lower()

    def test_single_keyword_overlap_passes(self):
        """Query with at least one keyword overlap passes."""
        spec = AgentSpec(
            agent_id="db_agent",
            name="Database Expert",
            description="Database agent",
            context_keywords=["sql", "query", "migration", "schema"],
        )
        result = heuristic_scope_check(spec, "write a sql query for users")
        assert result["in_scope"] is True

    def test_short_query_not_rejected(self):
        """Very short queries (< 3 words) are not rejected by keyword check."""
        spec = AgentSpec(
            agent_id="db_agent",
            name="Database Expert",
            description="Database agent",
            context_keywords=["sql", "query"],
        )
        # Only 2 words, neither matches — but too short to confidently reject
        result = heuristic_scope_check(spec, "help me")
        assert result["in_scope"] is True

    def test_exclusion_threshold_rejects(self):
        """Agent rejects when 2+ exclusion terms match the query."""
        spec = AgentSpec(
            agent_id="backend_agent",
            name="Backend Expert",
            description="Backend agent",
            context_keywords=["api", "server"],
            routing=RoutingConfig(exclusions=["frontend", "css", "react"]),
        )
        result = heuristic_scope_check(spec, "fix the frontend react component styling")
        assert result["in_scope"] is False

    def test_single_exclusion_not_enough(self):
        """Single exclusion match is not enough to reject."""
        spec = AgentSpec(
            agent_id="backend_agent",
            name="Backend Expert",
            description="Backend agent",
            context_keywords=["api", "server"],
            routing=RoutingConfig(exclusions=["frontend", "css", "react"]),
        )
        result = heuristic_scope_check(spec, "fix the frontend api endpoint")
        assert result["in_scope"] is True

    def test_no_exclusions_passes(self):
        """Agent without exclusions doesn't reject on that check."""
        spec = AgentSpec(
            agent_id="agent",
            name="Agent",
            description="Agent",
            context_keywords=["api"],
        )
        result = heuristic_scope_check(spec, "fix the api")
        assert result["in_scope"] is True


# ── LLM pre-screen tests ────────────────────────────────────────────────


class _ConcreteAgent(BaseAgent):
    """Concrete subclass of BaseAgent for testing."""

    def build_context(self) -> str:
        return "test context"


class TestLLMPrescreen:
    """Test the _llm_scope_prescreen method."""

    def test_prescreen_skipped_for_tier_a(self):
        """LLM pre-screen is NOT called for manually created agents."""
        client = MagicMock()
        spec = AgentSpec(
            agent_id="manual_agent",
            name="Manual Agent",
            description="Manually created",
            context_keywords=["test"],
            metadata={},  # No auto_generated flag
        )
        agent = _ConcreteAgent(spec, client)

        result = agent._llm_scope_prescreen("any query")
        assert result["in_scope"] is True
        # Ensure no LLM call was made
        client.messages.create.assert_not_called()

    def test_prescreen_rejects_out_of_scope(self):
        """LLM pre-screen rejects when Haiku says 'no'."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="no")]
        mock_response.usage.input_tokens = 400
        mock_response.usage.output_tokens = 5
        client.messages.create.return_value = mock_response

        spec = AgentSpec(
            agent_id="auto_agent",
            name="Auto Agent",
            description="Auto generated",
            context_keywords=["database"],
            metadata={"auto_generated": True, "tier": "B"},
        )
        agent = _ConcreteAgent(spec, client)

        result = agent._llm_scope_prescreen("how does react rendering work")
        assert result["in_scope"] is False
        assert result["tokens_used"] == 405

    def test_prescreen_accepts_in_scope(self):
        """LLM pre-screen accepts when Haiku says 'yes'."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="yes")]
        mock_response.usage.input_tokens = 400
        mock_response.usage.output_tokens = 5
        client.messages.create.return_value = mock_response

        spec = AgentSpec(
            agent_id="auto_agent",
            name="Auto Agent",
            description="Auto generated",
            context_keywords=["database"],
            metadata={"auto_generated": True, "tier": "B"},
        )
        agent = _ConcreteAgent(spec, client)

        result = agent._llm_scope_prescreen("write a sql migration")
        assert result["in_scope"] is True

    def test_prescreen_fail_open_on_error(self):
        """On API error, pre-screen assumes in-scope (fail open)."""
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")

        spec = AgentSpec(
            agent_id="auto_agent",
            name="Auto Agent",
            description="Auto generated",
            context_keywords=["database"],
            metadata={"auto_generated": True, "tier": "B"},
        )
        agent = _ConcreteAgent(spec, client)

        result = agent._llm_scope_prescreen("any query")
        assert result["in_scope"] is True

    def test_prescreen_uses_haiku_model(self):
        """Pre-screen should use Haiku for cost efficiency."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="yes")]
        mock_response.usage.input_tokens = 400
        mock_response.usage.output_tokens = 5
        client.messages.create.return_value = mock_response

        spec = AgentSpec(
            agent_id="auto_agent",
            name="Auto Agent",
            description="Auto generated",
            context_keywords=["database"],
            metadata={"auto_generated": True, "tier": "B"},
        )
        agent = _ConcreteAgent(spec, client)
        agent._llm_scope_prescreen("test query")

        # Verify Haiku model was used
        call_kwargs = client.messages.create.call_args[1]
        assert "haiku" in call_kwargs["model"]
        assert call_kwargs["max_tokens"] == 10

    def test_prescreen_includes_rnr_metadata(self):
        """Pre-screen system prompt includes R&R in_scope/out_of_scope."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="yes")]
        mock_response.usage.input_tokens = 400
        mock_response.usage.output_tokens = 5
        client.messages.create.return_value = mock_response

        spec = AgentSpec(
            agent_id="auto_agent",
            name="Auto Agent",
            description="Auto generated",
            context_keywords=["database"],
            metadata={
                "auto_generated": True,
                "tier": "B",
                "rnr": {
                    "in_scope": ["SQL queries", "Schema management"],
                    "out_of_scope": ["Frontend rendering"],
                },
            },
        )
        agent = _ConcreteAgent(spec, client)
        agent._llm_scope_prescreen("test query")

        call_kwargs = client.messages.create.call_args[1]
        system_prompt = call_kwargs["system"]
        assert "SQL queries" in system_prompt
        assert "Frontend rendering" in system_prompt
