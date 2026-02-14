"""Shared test fixtures."""

from unittest.mock import MagicMock

import pytest

from agenthub.agents.base import BaseAgent
from agenthub.models import AgentResponse, AgentSpec, Session


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, agent_id: str = "test_agent", keywords: list[str] | None = None):
        spec = AgentSpec(
            agent_id=agent_id,
            name="Test Agent",
            description="A test agent",
            context_keywords=keywords or ["test", "mock"],
        )
        # Use a mock client
        client = MagicMock()
        super().__init__(spec, client)

    def build_context(self) -> str:
        return "Mock context for testing"

    def run(self, query: str, session: Session, model: str | None = None, injected_context: str = "", max_tool_tokens: int = 30000) -> AgentResponse:
        return AgentResponse(
            content=f"Mock response for: {query}",
            agent_id=self.spec.agent_id,
            session_id=session.session_id,
            tokens_used=100,
        )


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_client():
    """Create a mock Anthropic client."""
    client = MagicMock()
    # Mock the messages.create response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock LLM response")]
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 50
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def sample_session():
    """Create a sample session."""
    return Session(
        session_id="test-session-123",
        agent_id="test_agent",
    )


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create directories
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    api_dir = src_dir / "api"
    api_dir.mkdir()
    models_dir = src_dir / "models"
    models_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # Create some Python files
    (src_dir / "__init__.py").write_text("# src package")
    (api_dir / "__init__.py").write_text("# api package")
    (api_dir / "routes.py").write_text(
        '''"""API routes."""
def get_users():
    return []

def create_user(data):
    return {"id": 1, **data}
'''
    )
    (api_dir / "auth.py").write_text(
        '''"""Authentication module."""
def authenticate(token):
    return True
'''
    )
    (models_dir / "__init__.py").write_text("# models package")
    (models_dir / "user.py").write_text(
        '''"""User model."""
class User:
    def __init__(self, name):
        self.name = name
'''
    )
    (tests_dir / "__init__.py").write_text("# tests package")
    (tests_dir / "test_api.py").write_text(
        '''"""API tests."""
def test_get_users():
    pass
'''
    )

    return tmp_path
