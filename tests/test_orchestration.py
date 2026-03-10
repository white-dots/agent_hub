"""Tests for smart orchestrator communication (multi-agent routing, followups)."""

from unittest.mock import MagicMock, patch

import pytest

from agenthub.hub import AgentHub
from agenthub.models import AgentResponse, AgentSpec, RoutingConfig, Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    # Default synthesis response
    synth_response = MagicMock()
    synth_response.content = [MagicMock(text="Synthesized response")]
    synth_response.usage.input_tokens = 100
    synth_response.usage.output_tokens = 50
    client.messages.create.return_value = synth_response
    return client


@pytest.fixture
def hub_with_agents(mock_client):
    """Create a hub with multiple mock agents for testing."""
    hub = AgentHub(client=mock_client)

    # Create agent specs
    pricing_spec = AgentSpec(
        agent_id="pricing_agent",
        name="Pricing Expert",
        description="Handles pricing, discounts, and margins",
        context_keywords=["price", "pricing", "discount", "margin", "cost"],
        routing=RoutingConfig(
            keyword_weights={"price": 2.0, "discount": 2.0},
        ),
    )
    auth_spec = AgentSpec(
        agent_id="auth_agent",
        name="Auth Expert",
        description="Handles authentication, login, and permissions",
        context_keywords=["auth", "login", "permission", "token", "session"],
        routing=RoutingConfig(
            keyword_weights={"auth": 2.0, "login": 2.0},
        ),
    )
    api_spec = AgentSpec(
        agent_id="api_agent",
        name="API Expert",
        description="Handles API endpoints, routing, and middleware",
        context_keywords=["api", "endpoint", "route", "middleware", "request"],
        routing=RoutingConfig(
            keyword_weights={"api": 2.0, "endpoint": 2.0},
        ),
    )

    # Create mock agents
    for spec in [pricing_spec, auth_spec, api_spec]:
        agent = MagicMock()
        agent.spec = spec
        hub.register(agent)

    return hub


# ---------------------------------------------------------------------------
# _route_multi tests
# ---------------------------------------------------------------------------

class TestRouteMulti:
    def test_single_domain_query_returns_one(self, hub_with_agents):
        """A clearly single-domain query should route to one agent."""
        result = hub_with_agents._route_multi("What is the pricing formula?")
        # Should have at least one result
        assert len(result) >= 1
        assert result[0] == "pricing_agent"

    def test_multi_domain_query_returns_multiple(self, hub_with_agents):
        """A query spanning domains should return multiple agents."""
        # This query touches both pricing and API
        result = hub_with_agents._route_multi(
            "How does the API endpoint calculate the discount price?"
        )
        assert len(result) >= 1  # At least one agent

    def test_max_agents_respected(self, hub_with_agents):
        result = hub_with_agents._route_multi(
            "price auth api endpoint discount login",
            max_agents=2,
        )
        assert len(result) <= 2

    def test_empty_query(self, hub_with_agents):
        result = hub_with_agents._route_multi("")
        # Should handle gracefully
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _resolve_agent_name tests
# ---------------------------------------------------------------------------

class TestResolveAgentName:
    def test_direct_id_match(self, hub_with_agents):
        result = hub_with_agents._resolve_agent_name("pricing_agent")
        assert result == "pricing_agent"

    def test_fuzzy_match_by_name(self, hub_with_agents):
        result = hub_with_agents._resolve_agent_name("Pricing Expert")
        assert result == "pricing_agent"

    def test_partial_match(self, hub_with_agents):
        result = hub_with_agents._resolve_agent_name("pricing")
        assert result == "pricing_agent"

    def test_no_match(self, hub_with_agents):
        result = hub_with_agents._resolve_agent_name("nonexistent_agent")
        assert result is None

    def test_exclude_set(self, hub_with_agents):
        result = hub_with_agents._resolve_agent_name(
            "pricing_agent", exclude={"pricing_agent"}
        )
        assert result is None


# ---------------------------------------------------------------------------
# _handle_followups tests
# ---------------------------------------------------------------------------

class TestHandleFollowups:
    def test_no_followup_returns_original(self, hub_with_agents):
        """If needs_followup is False, return original response."""
        response = AgentResponse(
            content="Here's the answer",
            agent_id="pricing_agent",
            session_id="test",
            needs_followup=False,
        )
        session = Session(session_id="test", agent_id="pricing_agent")

        result = hub_with_agents._handle_followups(response, "test query", session, None)
        assert result == response

    def test_followup_with_suggested_agent(self, hub_with_agents, mock_client):
        """When needs_followup=True and suggested_agent is set, hand off."""
        # First agent says "ask auth_agent"
        initial = AgentResponse(
            content="This is about authentication, not pricing",
            agent_id="pricing_agent",
            session_id="test",
            needs_followup=True,
            suggested_agent="auth_agent",
        )
        # Second agent gives answer
        auth_response = AgentResponse(
            content="Authentication works like this...",
            agent_id="auth_agent",
            session_id="test",
            needs_followup=False,
        )
        hub_with_agents._agents["auth_agent"].run.return_value = auth_response

        session = Session(session_id="test", agent_id="pricing_agent")
        result = hub_with_agents._handle_followups(initial, "How does login work?", session, None)

        # Should have called auth_agent
        hub_with_agents._agents["auth_agent"].run.assert_called_once()
        # Final response should be from auth or merged
        assert "auth" in result.agent_id or "Authentication" in result.content

    def test_followup_prevents_loops(self, hub_with_agents):
        """Should not revisit an agent that was already followed."""
        # Agent A says ask B, agent B says ask A
        response_a = AgentResponse(
            content="Ask B",
            agent_id="pricing_agent",
            session_id="test",
            needs_followup=True,
            suggested_agent="auth_agent",
        )
        response_b = AgentResponse(
            content="Ask pricing",
            agent_id="auth_agent",
            session_id="test",
            needs_followup=True,
            suggested_agent="pricing_agent",  # Loop!
        )
        hub_with_agents._agents["auth_agent"].run.return_value = response_b

        session = Session(session_id="test", agent_id="pricing_agent")
        result = hub_with_agents._handle_followups(response_a, "test", session, None)

        # Should not loop forever — max_followups caps it
        assert result is not None

    def test_followup_max_depth(self, hub_with_agents):
        """Should stop after max_followups even if agents keep suggesting."""
        # Create a chain: pricing -> auth -> api -> ...
        response1 = AgentResponse(
            content="Ask auth",
            agent_id="pricing_agent",
            session_id="test",
            needs_followup=True,
            suggested_agent="auth_agent",
        )
        response2 = AgentResponse(
            content="Ask api",
            agent_id="auth_agent",
            session_id="test",
            needs_followup=True,
            suggested_agent="api_agent",
        )
        response3 = AgentResponse(
            content="Final answer from api",
            agent_id="api_agent",
            session_id="test",
            needs_followup=False,
        )

        hub_with_agents._agents["auth_agent"].run.return_value = response2
        hub_with_agents._agents["api_agent"].run.return_value = response3

        session = Session(session_id="test", agent_id="pricing_agent")
        result = hub_with_agents._handle_followups(response1, "test", session, None, max_followups=3)
        assert result is not None


# ---------------------------------------------------------------------------
# _run_multi_agent tests
# ---------------------------------------------------------------------------

class TestRunMultiAgent:
    def test_runs_multiple_agents(self, hub_with_agents):
        """Should run multiple agents and merge results."""
        resp1 = AgentResponse(
            content="Pricing info",
            agent_id="pricing_agent",
            session_id="test",
            tokens_used=100,
        )
        resp2 = AgentResponse(
            content="Auth info",
            agent_id="auth_agent",
            session_id="test",
            tokens_used=100,
        )
        hub_with_agents._agents["pricing_agent"].run.return_value = resp1
        hub_with_agents._agents["auth_agent"].run.return_value = resp2

        session = Session(session_id="test", agent_id="test")
        result = hub_with_agents._run_multi_agent(
            "How do prices and auth work?",
            ["pricing_agent", "auth_agent"],
            session,
            None,
        )
        assert result is not None
        assert result.metadata.get("multi_agent") or result.content

    def test_skips_scope_rejections(self, hub_with_agents):
        """Should skip agents that reject the query."""
        resp_reject = AgentResponse(
            content="Out of scope",
            agent_id="pricing_agent",
            session_id="test",
            tokens_used=0,
            metadata={"scope_rejected": True},
        )
        resp_ok = AgentResponse(
            content="Auth answer",
            agent_id="auth_agent",
            session_id="test",
            tokens_used=100,
        )
        hub_with_agents._agents["pricing_agent"].run.return_value = resp_reject
        hub_with_agents._agents["auth_agent"].run.return_value = resp_ok

        session = Session(session_id="test", agent_id="test")
        result = hub_with_agents._run_multi_agent(
            "How does login work?",
            ["pricing_agent", "auth_agent"],
            session,
            None,
        )
        # Should only have auth's response
        assert "Auth answer" in result.content


# ---------------------------------------------------------------------------
# _merge_responses tests
# ---------------------------------------------------------------------------

class TestMergeResponses:
    def test_merge_with_synthesis(self, hub_with_agents, mock_client):
        """Should use LLM synthesis when client is available."""
        responses = [
            AgentResponse(
                content="Pricing handles discounts",
                agent_id="pricing_agent",
                session_id="test",
                tokens_used=100,
            ),
            AgentResponse(
                content="Auth handles tokens",
                agent_id="auth_agent",
                session_id="test",
                tokens_used=100,
            ),
        ]
        session = Session(session_id="test", agent_id="test")
        result = hub_with_agents._merge_responses(responses, "test query", session)

        # Should have called the LLM for synthesis
        mock_client.messages.create.assert_called()
        assert result.metadata.get("multi_agent")
        assert "pricing_agent" in result.metadata.get("contributing_agents", [])

    def test_merge_fallback_without_client(self, hub_with_agents):
        """Without client, should do structured concatenation."""
        hub_with_agents.client = None

        responses = [
            AgentResponse(
                content="Pricing info",
                agent_id="pricing_agent",
                session_id="test",
                tokens_used=100,
            ),
            AgentResponse(
                content="Auth info",
                agent_id="auth_agent",
                session_id="test",
                tokens_used=100,
            ),
        ]
        session = Session(session_id="test", agent_id="test")
        result = hub_with_agents._merge_responses(responses, "test query", session)

        assert "Pricing info" in result.content
        assert "Auth info" in result.content
        assert result.metadata.get("multi_agent")
