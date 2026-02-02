"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from agenthub.models import (
    AgentCapability,
    AgentResponse,
    AgentSpec,
    Artifact,
    Message,
    Session,
)


class TestAgentSpec:
    """Tests for AgentSpec model."""

    def test_create_minimal(self):
        """Test creating spec with minimal fields."""
        spec = AgentSpec(
            agent_id="test",
            name="Test Agent",
            description="A test agent",
        )
        assert spec.agent_id == "test"
        assert spec.name == "Test Agent"
        assert spec.capabilities == []
        assert spec.temperature == 0.7

    def test_create_full(self):
        """Test creating spec with all fields."""
        spec = AgentSpec(
            agent_id="code_agent",
            name="Code Expert",
            description="Knows code",
            capabilities=[AgentCapability.CODE_READ, AgentCapability.CODE_WRITE],
            context_keywords=["code", "function"],
            context_paths=["src/**/*.py"],
            estimated_tokens=5000,
            max_context_size=100000,
            system_prompt="You are a code expert.",
            temperature=0.5,
            metadata={"auto_generated": True},
        )
        assert len(spec.capabilities) == 2
        assert "code" in spec.context_keywords
        assert spec.temperature == 0.5
        assert spec.metadata["auto_generated"] is True

    def test_temperature_validation(self):
        """Test temperature must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AgentSpec(
                agent_id="test",
                name="Test",
                description="Test",
                temperature=1.5,
            )

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            AgentSpec(agent_id="test")


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_roles(self):
        """Test valid message roles."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_invalid_role(self):
        """Test invalid role raises error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating a session."""
        session = Session(session_id="123", agent_id="test_agent")
        assert session.session_id == "123"
        assert session.agent_id == "test_agent"
        assert session.messages == []
        assert session.total_tokens_used == 0

    def test_session_with_messages(self):
        """Test session with message history."""
        session = Session(
            session_id="123",
            agent_id="test_agent",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )
        assert len(session.messages) == 2


class TestArtifact:
    """Tests for Artifact model."""

    def test_create_code_artifact(self):
        """Test creating a code artifact."""
        artifact = Artifact(
            artifact_type="code",
            content="print('hello')",
            language="python",
        )
        assert artifact.artifact_type == "code"
        assert artifact.language == "python"

    def test_artifact_types(self):
        """Test valid artifact types."""
        for type_ in ["code", "file", "sql", "json", "markdown"]:
            artifact = Artifact(artifact_type=type_, content="test")
            assert artifact.artifact_type == type_


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_create_response(self):
        """Test creating an agent response."""
        response = AgentResponse(
            content="Here's my answer",
            agent_id="test_agent",
            session_id="session-123",
            tokens_used=150,
        )
        assert response.content == "Here's my answer"
        assert response.tokens_used == 150
        assert response.needs_followup is False

    def test_response_with_artifacts(self):
        """Test response with artifacts."""
        response = AgentResponse(
            content="Here's code:",
            agent_id="test_agent",
            session_id="session-123",
            artifacts=[
                Artifact(artifact_type="code", content="print('hi')", language="python")
            ],
        )
        assert len(response.artifacts) == 1
        assert response.artifacts[0].language == "python"
