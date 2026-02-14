"""Tests for hub.run() scope-rejection retry logic."""

from unittest.mock import MagicMock, patch

import pytest

from agenthub.hub import AgentHub
from agenthub.models import AgentResponse, AgentSpec, RoutingConfig, Session
from agenthub.routing import KeywordRouter


def _make_agent(spec, run_return):
    """Create a mock agent with a given spec and run() return value."""
    agent = MagicMock()
    agent.spec = spec
    agent.run = MagicMock(return_value=run_return)
    return agent


def _make_rejection(agent_id, session_id, suggested_agent=None):
    """Create a scope-rejected AgentResponse."""
    return AgentResponse(
        content="Outside my scope.",
        agent_id=agent_id,
        session_id=session_id,
        tokens_used=0,
        artifacts=[],
        metadata={
            "scope_rejected": True,
            "suggested_agent": suggested_agent,
        },
    )


def _make_success(agent_id, session_id, content="Here is the answer."):
    """Create a successful AgentResponse."""
    return AgentResponse(
        content=content,
        agent_id=agent_id,
        session_id=session_id,
        tokens_used=100,
        artifacts=[],
    )


class TestScopeRejectionRetry:
    """Test retry on scope rejection in hub.run()."""

    def test_retry_with_suggested_agent(self):
        """Hub retries with the suggested agent when scope_rejected is True."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        spec_a = AgentSpec(
            agent_id="agent_a", name="Agent A", description="A",
            context_keywords=["query"],
        )
        spec_b = AgentSpec(
            agent_id="agent_b", name="Agent B", description="B",
            context_keywords=["query"],
        )

        agent_a = _make_agent(spec_a, _make_rejection("agent_a", "s1", suggested_agent="agent_b"))
        agent_b = _make_agent(spec_b, _make_success("agent_b", "s1"))

        hub._agents = {"agent_a": agent_a, "agent_b": agent_b}

        response = hub.run("test query", agent_id="agent_a")

        assert response.agent_id == "agent_b"
        assert response.content == "Here is the answer."
        assert response.metadata.get("retry_history") == ["agent_a", "agent_b"]
        assert response.metadata.get("retries_used") == 1

    def test_retry_cap_at_5(self):
        """Hub stops retrying after 5 retries (6 total attempts)."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        specs = {}
        agents = {}
        # Create enough agents to exceed the retry cap
        for i, name in enumerate("abcdefgh"):
            aid = f"agent_{name}"
            specs[aid] = AgentSpec(
                agent_id=aid, name=f"Agent {name.upper()}", description=name,
                context_keywords=["shared"],
            )
            # All reject and suggest the next one
            next_name = chr(ord("a") + i + 1) if i + 1 < 8 else None
            next_agent = f"agent_{next_name}" if next_name else None
            agents[aid] = _make_agent(
                specs[aid],
                _make_rejection(aid, "s1", suggested_agent=next_agent),
            )

        hub._agents = agents

        response = hub.run("test query", agent_id="agent_a")

        # Should have tried a→b→c→d→e→f (6 attempts = initial + 5 retries)
        assert response.metadata.get("scope_rejected") is True
        assert len(response.metadata.get("retry_history", [])) == 6

    def test_circular_suggestion_breaks(self):
        """Hub breaks circular suggestions (A->B->A)."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        spec_a = AgentSpec(
            agent_id="agent_a", name="Agent A", description="A",
            context_keywords=["test"],
        )
        spec_b = AgentSpec(
            agent_id="agent_b", name="Agent B", description="B",
            context_keywords=["test"],
        )

        agent_a = _make_agent(spec_a, _make_rejection("agent_a", "s1", suggested_agent="agent_b"))
        agent_b = _make_agent(spec_b, _make_rejection("agent_b", "s1", suggested_agent="agent_a"))

        hub._agents = {"agent_a": agent_a, "agent_b": agent_b}

        response = hub.run("test query", agent_id="agent_a")

        # Should try a -> b, then detect cycle and stop
        assert response.metadata.get("scope_rejected") is True
        assert response.metadata.get("retry_history") == ["agent_a", "agent_b"]

    def test_no_retry_when_not_rejected(self):
        """Hub doesn't retry when the response is successful."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        spec = AgentSpec(
            agent_id="agent_a", name="Agent A", description="A",
            context_keywords=["test"],
        )
        agent = _make_agent(spec, _make_success("agent_a", "s1"))

        hub._agents = {"agent_a": agent}

        response = hub.run("test query", agent_id="agent_a")

        assert response.content == "Here is the answer."
        assert "retry_history" not in response.metadata

    def test_retry_falls_back_to_next_best_score(self):
        """When no suggested_agent, hub tries the next-best-scoring agent."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        spec_a = AgentSpec(
            agent_id="agent_a", name="Agent A", description="A",
            context_keywords=["query", "data"],
        )
        spec_b = AgentSpec(
            agent_id="agent_b", name="Agent B", description="B",
            context_keywords=["query"],
        )

        # agent_a rejects without suggesting anyone
        agent_a = _make_agent(spec_a, _make_rejection("agent_a", "s1", suggested_agent=None))
        agent_b = _make_agent(spec_b, _make_success("agent_b", "s1"))

        hub._agents = {"agent_a": agent_a, "agent_b": agent_b}

        response = hub.run("test query", agent_id="agent_a")

        # Should fall back to agent_b via keyword scoring
        assert response.agent_id == "agent_b"
        assert response.metadata.get("retries_used") == 1

    def test_fuzzy_match_suggested_agent_name(self):
        """Hub fuzzy-matches suggested_agent when it's a display name, not an ID."""
        hub = AgentHub(client=MagicMock())
        hub._router = KeywordRouter()

        spec_a = AgentSpec(
            agent_id="agent_a", name="Agent A", description="A",
            context_keywords=["query"],
        )
        spec_b = AgentSpec(
            agent_id="backend_expert", name="Backend Expert", description="B",
            context_keywords=["query"],
        )

        # Suggests by display name, not ID
        agent_a = _make_agent(spec_a, _make_rejection("agent_a", "s1", suggested_agent="Backend Expert"))
        agent_b = _make_agent(spec_b, _make_success("backend_expert", "s1"))

        hub._agents = {"agent_a": agent_a, "backend_expert": agent_b}

        response = hub.run("test query", agent_id="agent_a")

        assert response.agent_id == "backend_expert"
