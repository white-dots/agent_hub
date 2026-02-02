"""Tests for AgentHub."""

import pytest

from agenthub.hub import AgentHub
from agenthub.models import AgentSpec

from tests.conftest import MockAgent


class TestAgentRegistration:
    """Tests for agent registration."""

    def test_register_agent(self, mock_client):
        """Test registering an agent."""
        hub = AgentHub(client=mock_client)
        agent = MockAgent("my_agent")

        hub.register(agent)

        assert "my_agent" in [a.agent_id for a in hub.list_agents()]

    def test_register_duplicate_raises(self, mock_client):
        """Test registering duplicate agent raises error."""
        hub = AgentHub(client=mock_client)
        agent1 = MockAgent("same_id")
        agent2 = MockAgent("same_id")

        hub.register(agent1)

        with pytest.raises(ValueError, match="already registered"):
            hub.register(agent2)

    def test_unregister_agent(self, mock_client):
        """Test unregistering an agent."""
        hub = AgentHub(client=mock_client)
        agent = MockAgent("to_remove")

        hub.register(agent)
        hub.unregister("to_remove")

        assert "to_remove" not in [a.agent_id for a in hub.list_agents()]

    def test_get_agent(self, mock_client):
        """Test getting an agent by ID."""
        hub = AgentHub(client=mock_client)
        agent = MockAgent("findme")

        hub.register(agent)
        found = hub.get_agent("findme")

        assert found is agent

    def test_get_nonexistent_agent(self, mock_client):
        """Test getting nonexistent agent returns None."""
        hub = AgentHub(client=mock_client)
        assert hub.get_agent("nonexistent") is None


class TestSessionManagement:
    """Tests for session management."""

    def test_create_session(self, mock_client):
        """Test creating a session."""
        hub = AgentHub(client=mock_client)
        session = hub.create_session()

        assert session.session_id is not None
        assert session.agent_id == "router"

    def test_create_session_with_agent(self, mock_client):
        """Test creating session with specific agent."""
        hub = AgentHub(client=mock_client)
        session = hub.create_session(agent_id="code_agent")

        assert session.agent_id == "code_agent"

    def test_get_session(self, mock_client):
        """Test retrieving a session."""
        hub = AgentHub(client=mock_client)
        session = hub.create_session()

        retrieved = hub.get_session(session.session_id)

        assert retrieved is session

    def test_delete_session(self, mock_client):
        """Test deleting a session."""
        hub = AgentHub(client=mock_client)
        session = hub.create_session()
        session_id = session.session_id

        hub.delete_session(session_id)

        assert hub.get_session(session_id) is None


class TestRouting:
    """Tests for query routing."""

    def test_route_by_keyword(self, mock_client):
        """Test routing based on keywords."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("code_agent", keywords=["code", "function"]))
        hub.register(MockAgent("db_agent", keywords=["database", "query"]))

        assert hub.route("fix the code") == "code_agent"
        assert hub.route("write a database query") == "db_agent"

    def test_route_no_agents_raises(self, mock_client):
        """Test routing with no agents raises error."""
        hub = AgentHub(client=mock_client)

        with pytest.raises(ValueError, match="No agents registered"):
            hub.route("any query")

    def test_route_fallback(self, mock_client):
        """Test routing falls back to first agent."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("first"))
        hub.register(MockAgent("second"))

        # Query doesn't match any keywords
        result = hub.route("something unrelated xyz")
        assert result == "first"


class TestExecution:
    """Tests for query execution."""

    def test_run_creates_session(self, mock_client):
        """Test run creates a new session if none provided."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("test_agent", keywords=["test"]))

        response = hub.run("test query")

        assert response.session_id is not None
        assert len(hub.list_sessions()) == 1

    def test_run_with_existing_session(self, mock_client):
        """Test run with existing session."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("test_agent", keywords=["test"]))
        session = hub.create_session()

        response = hub.run("test query", session_id=session.session_id)

        assert response.session_id == session.session_id

    def test_run_force_agent(self, mock_client):
        """Test forcing a specific agent."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("agent_a", keywords=["aaa"]))
        hub.register(MockAgent("agent_b", keywords=["bbb"]))

        # Query matches agent_a, but we force agent_b
        response = hub.run("aaa query", agent_id="agent_b")

        assert response.agent_id == "agent_b"

    def test_run_invalid_session_raises(self, mock_client):
        """Test run with invalid session raises error."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("test_agent"))

        with pytest.raises(ValueError, match="not found"):
            hub.run("query", session_id="nonexistent")

    def test_run_updates_session(self, mock_client):
        """Test run updates session with messages."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("test_agent", keywords=["test"]))
        session = hub.create_session()

        hub.run("test query", session_id=session.session_id)

        assert len(session.messages) == 2  # user + assistant
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"


class TestListAgents:
    """Tests for listing agents by tier."""

    def test_list_all_agents(self, mock_client):
        """Test listing all agents."""
        hub = AgentHub(client=mock_client)
        hub.register(MockAgent("agent1"))
        hub.register(MockAgent("agent2"))

        agents = hub.list_agents()

        assert len(agents) == 2

    def test_list_tier_a_agents(self, mock_client):
        """Test listing Tier A (business) agents."""
        hub = AgentHub(client=mock_client)

        # Regular agent (Tier A)
        hub.register(MockAgent("business"))

        # Auto-generated agent (Tier B)
        auto_agent = MockAgent("auto")
        auto_agent.spec.metadata["auto_generated"] = True
        hub.register(auto_agent)

        tier_a = hub.list_agents(tier="A")

        assert len(tier_a) == 1
        assert tier_a[0].agent_id == "business"

    def test_list_tier_b_agents(self, mock_client):
        """Test listing Tier B (auto-generated) agents."""
        hub = AgentHub(client=mock_client)

        # Regular agent (Tier A)
        hub.register(MockAgent("business"))

        # Auto-generated agent (Tier B)
        auto_agent = MockAgent("auto")
        auto_agent.spec.metadata["auto_generated"] = True
        hub.register(auto_agent)

        tier_b = hub.list_agents(tier="B")

        assert len(tier_b) == 1
        assert tier_b[0].agent_id == "auto"
