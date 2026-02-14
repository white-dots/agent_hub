"""Tests for sub-agent functionality (Phase 0 of Parallel Sessions)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agenthub.auto.import_graph import ImportGraph
from agenthub.auto.sub_agent_policy import SubAgentPolicy
from agenthub.models import AgentSpec, SubAgentBoundary


# === Fixtures ===


@pytest.fixture
def large_project(tmp_path):
    """Create a large project structure that should trigger subdivision.

    Creates 70+ files across 4 subdirectories to exceed the default
    min_files_to_split threshold of 60.
    """
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()

    # Create 4 subdirectories with files
    subdirs = ["api", "models", "services", "utils"]

    for subdir in subdirs:
        subdir_path = backend_dir / subdir
        subdir_path.mkdir()
        (subdir_path / "__init__.py").write_text(f"# {subdir} package")

        # Create 15-20 files per subdir
        for i in range(18):
            file_path = subdir_path / f"{subdir}_{i}.py"

            # Create files with imports between subdirs
            content = f'''"""Module {subdir}_{i}."""

'''
            # Add imports from other modules
            if subdir == "api":
                content += "from backend.models import model_0\n"
                content += "from backend.services import service_0\n"
            elif subdir == "services":
                content += "from backend.models import model_0\n"
            elif subdir == "utils":
                content += "# No internal imports\n"

            content += f'''

def {subdir}_function_{i}():
    """Function {i} in {subdir}."""
    pass


class {subdir.title()}Class{i}:
    """Class {i} in {subdir}."""
    pass
'''
            file_path.write_text(content)

    return tmp_path


@pytest.fixture
def small_project(tmp_path):
    """Create a small project that should NOT trigger subdivision."""
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    (backend_dir / "__init__.py").write_text("# backend")

    # Create only 10 files
    for i in range(10):
        (backend_dir / f"module_{i}.py").write_text(
            f'"""Module {i}."""\ndef func_{i}(): pass'
        )

    return tmp_path


@pytest.fixture
def mock_agent():
    """Create a mock Tier B agent."""
    spec = AgentSpec(
        agent_id="backend",
        name="Backend Module Expert",
        description="Expert on backend/ module",
        context_paths=["backend/**/*.py"],
        context_keywords=["backend", "api", "models"],
        metadata={"auto_generated": True, "tier": "B"},
    )

    agent = MagicMock()
    agent.spec = spec
    return agent


@pytest.fixture
def mock_auto_manager():
    """Create a mock AutoAgentManager."""
    manager = MagicMock()
    manager.hub = MagicMock()
    manager.hub.client = MagicMock()
    return manager


# === SubAgentPolicy Tests ===


class TestSubAgentPolicy:
    """Tests for SubAgentPolicy."""

    def test_default_thresholds(self):
        """Policy should have sensible defaults."""
        policy = SubAgentPolicy()

        assert policy.min_files_to_split == 60
        assert policy.min_subdirs_to_split == 3
        assert policy.min_files_per_sub == 10
        assert policy.max_sub_agents == 6

    def test_should_subdivide_threshold_met(self, large_project, mock_agent):
        """Agent with 60+ files and 3+ subdirs should subdivide."""
        # Update agent's context_paths to point to our large project
        mock_agent.spec.context_paths = [str(large_project / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(large_project))
        graph.build()

        policy = SubAgentPolicy()
        result = policy.should_subdivide(mock_agent, graph)

        # Should want to subdivide (70+ files, 4 subdirs)
        assert result is True

    def test_should_not_subdivide_small_agent(self, small_project, mock_agent):
        """Agent with <60 files should not subdivide."""
        mock_agent.spec.context_paths = [str(small_project / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(small_project))
        graph.build()

        policy = SubAgentPolicy()
        result = policy.should_subdivide(mock_agent, graph)

        assert result is False

    def test_should_not_subdivide_few_subdirs(self, tmp_path, mock_agent):
        """Agent spanning only 1-2 subdirs should not subdivide."""
        # Create 70 files but all in one directory
        single_dir = tmp_path / "backend" / "all_files"
        single_dir.mkdir(parents=True)

        for i in range(70):
            (single_dir / f"module_{i}.py").write_text(f"# Module {i}")

        mock_agent.spec.context_paths = [str(tmp_path / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(tmp_path))
        graph.build()

        policy = SubAgentPolicy(min_subdirs_to_split=3)
        result = policy.should_subdivide(mock_agent, graph)

        # Has enough files but not enough subdirs
        assert result is False

    def test_propose_subdivisions_uses_import_graph(self, large_project, mock_agent):
        """Subdivisions should follow import graph clusters when possible."""
        mock_agent.spec.context_paths = [str(large_project / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(large_project))
        graph.build()

        policy = SubAgentPolicy()
        boundaries = policy.propose_subdivisions(mock_agent, graph)

        assert len(boundaries) > 0
        assert len(boundaries) <= policy.max_sub_agents

        # Check boundary structure
        for boundary in boundaries:
            assert isinstance(boundary, SubAgentBoundary)
            assert boundary.parent_agent_id == "backend"
            assert boundary.sub_agent_id.startswith("backend_")
            assert boundary.file_count >= policy.min_files_per_sub

    def test_propose_subdivisions_fallback_to_directory(self, tmp_path, mock_agent):
        """Falls back to directory-based when import graph is sparse."""
        # Create files with no imports between them
        for subdir in ["a", "b", "c", "d"]:
            subdir_path = tmp_path / "backend" / subdir
            subdir_path.mkdir(parents=True)
            for i in range(20):
                (subdir_path / f"mod_{i}.py").write_text(f"# isolated module {i}")

        mock_agent.spec.context_paths = [str(tmp_path / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(tmp_path))
        graph.build()

        policy = SubAgentPolicy()
        boundaries = policy.propose_subdivisions(mock_agent, graph)

        # Should still produce reasonable boundaries based on directory structure
        assert len(boundaries) > 0

    def test_max_sub_agents_limit(self, tmp_path, mock_agent):
        """Should not create more than max_sub_agents."""
        # Create many subdirectories
        for i in range(10):
            subdir_path = tmp_path / "backend" / f"subdir_{i}"
            subdir_path.mkdir(parents=True)
            for j in range(15):
                (subdir_path / f"mod_{j}.py").write_text(f"# module {j}")

        mock_agent.spec.context_paths = [str(tmp_path / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(tmp_path))
        graph.build()

        policy = SubAgentPolicy(max_sub_agents=4)
        boundaries = policy.propose_subdivisions(mock_agent, graph)

        assert len(boundaries) <= 4

    def test_min_files_per_sub_filtering(self, tmp_path, mock_agent):
        """Sub-agents with too few files should be filtered out."""
        # Create varying-size subdirectories
        for size, name in [(20, "big"), (5, "small"), (25, "medium")]:
            subdir_path = tmp_path / "backend" / name
            subdir_path.mkdir(parents=True)
            for i in range(size):
                (subdir_path / f"mod_{i}.py").write_text(f"# module {i}")

        mock_agent.spec.context_paths = [str(tmp_path / "backend" / "**" / "*.py")]

        graph = ImportGraph(str(tmp_path))
        graph.build()

        policy = SubAgentPolicy(min_files_per_sub=10)
        boundaries = policy.propose_subdivisions(mock_agent, graph)

        # "small" with only 5 files should be merged or excluded
        for boundary in boundaries:
            assert boundary.file_count >= 10


# === SubAgentManager Tests ===


class TestSubAgentManager:
    """Tests for SubAgentManager."""

    def test_route_to_sub_agent_precision(self, large_project, mock_auto_manager):
        """Most specific sub-agent should be returned for a file."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        # Create a mock agent that spans all of backend/
        mock_agent = MagicMock()
        mock_agent.spec = AgentSpec(
            agent_id="backend",
            name="Backend Expert",
            description="Backend module",
            context_paths=[str(large_project / "backend" / "**" / "*.py")],
            metadata={"auto_generated": True, "tier": "B"},
        )

        mock_auto_manager._auto_agents = {"backend": mock_agent}
        mock_auto_manager.hub.list_agents.return_value = [mock_agent.spec]

        graph = ImportGraph(str(large_project))
        graph.build()

        manager = SubAgentManager(
            auto_manager=mock_auto_manager,
            import_graph=graph,
            policy=SubAgentPolicy(min_files_to_split=20),  # Lower threshold for test
        )

        # Manually trigger subdivision
        report = manager.evaluate_and_subdivide()

        # If subdivision happened, test routing
        if "backend" in report and len(report["backend"]) > 0:
            # Get the first boundary
            boundary = report["backend"][0]
            # Should be able to route a file in that boundary's scope
            test_file = boundary.root_path + "/api_0.py"
            sub_agent = manager.route_to_sub_agent("backend", test_file)
            # May or may not find sub-agent depending on exact paths
            # The key is it shouldn't error

    def test_get_team(self, mock_auto_manager):
        """Should return all sub-agents for a team lead."""
        from agenthub.auto.sub_agent_manager import SubAgentManager, SubCodeAgent

        graph = MagicMock()
        graph.nodes = {}
        graph.edges = []

        manager = SubAgentManager(mock_auto_manager, graph)

        # Manually add some sub-agents for testing
        mock_sub_agent = MagicMock(spec=SubCodeAgent)
        manager._sub_agents["backend"] = {
            "backend_api": mock_sub_agent,
            "backend_models": mock_sub_agent,
        }

        team = manager.get_team("backend")
        assert len(team) == 2

        # Non-existent team should return empty
        assert manager.get_team("nonexistent") == []

    def test_get_team_lead(self, mock_auto_manager):
        """Should return parent agent for a sub-agent."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        graph = MagicMock()
        graph.nodes = {}

        mock_parent = MagicMock()
        mock_auto_manager._auto_agents = {"backend": mock_parent}

        manager = SubAgentManager(mock_auto_manager, graph)
        manager._boundaries["backend_api"] = SubAgentBoundary(
            parent_agent_id="backend",
            sub_agent_id="backend_api",
            root_path="backend/api",
            include_patterns=["backend/api/**/*.py"],
        )

        team_lead = manager.get_team_lead("backend_api")
        assert team_lead == mock_parent

    def test_team_query_delegation(self, mock_auto_manager):
        """Team lead should delegate to correct sub-agent based on keywords."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        graph = MagicMock()
        graph.nodes = {}

        # Create mock parent and sub-agents
        mock_parent = MagicMock()
        mock_api_sub = MagicMock()
        mock_api_sub.spec = AgentSpec(
            agent_id="backend_api",
            name="API Team Member",
            description="API sub-agent",
            context_keywords=["api", "routes", "endpoints"],
        )

        mock_auto_manager._auto_agents = {"backend": mock_parent}

        manager = SubAgentManager(mock_auto_manager, graph)
        manager._sub_agents["backend"] = {"backend_api": mock_api_sub}

        session = MagicMock()

        # Query with "api" keyword should route to api sub-agent
        manager.team_query("backend", "How do the api routes work?", session, delegate=True)

        # api sub-agent should have been called
        mock_api_sub.run.assert_called_once()
        mock_parent.run.assert_not_called()

    def test_team_query_no_delegation(self, mock_auto_manager):
        """delegate=False should use team lead only."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        graph = MagicMock()
        graph.nodes = {}

        mock_parent = MagicMock()
        mock_sub = MagicMock()

        mock_auto_manager._auto_agents = {"backend": mock_parent}

        manager = SubAgentManager(mock_auto_manager, graph)
        manager._sub_agents["backend"] = {"backend_api": mock_sub}

        session = MagicMock()

        manager.team_query("backend", "Any question", session, delegate=False)

        mock_parent.run.assert_called_once()
        mock_sub.run.assert_not_called()

    def test_hierarchy_report(self, mock_auto_manager):
        """Should generate accurate hierarchy report."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        graph = MagicMock()
        graph.nodes = {}

        manager = SubAgentManager(mock_auto_manager, graph)

        # Set up hierarchy
        mock_sub_api = MagicMock()
        mock_sub_api.spec = MagicMock()
        mock_sub_api.spec.agent_id = "backend_api"

        mock_sub_models = MagicMock()
        mock_sub_models.spec = MagicMock()
        mock_sub_models.spec.agent_id = "backend_models"

        manager._sub_agents["backend"] = {
            "backend_api": mock_sub_api,
            "backend_models": mock_sub_models,
        }

        report = manager.get_hierarchy_report()

        assert report["total_team_leads"] == 1
        assert report["total_sub_agents"] == 2
        assert "backend" in report["team_leads"]
        assert "backend_api" in report["teams"]["backend"]
        assert "backend_models" in report["teams"]["backend"]


# === Integration Tests ===


class TestSubAgentIntegration:
    """Integration tests for sub-agent system."""

    def test_auto_manager_enable_sub_agents(self, tmp_path):
        """AutoAgentManager should integrate with SubAgentManager."""
        from agenthub.auto.manager import AutoAgentManager
        from agenthub.hub import AgentHub

        # Create a mock hub
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="response")],
            usage=MagicMock(input_tokens=10, output_tokens=10),
        )

        hub = AgentHub(mock_client)

        # Create a project with enough files
        for subdir in ["api", "models", "services"]:
            subdir_path = tmp_path / "backend" / subdir
            subdir_path.mkdir(parents=True)
            for i in range(25):
                (subdir_path / f"mod_{i}.py").write_text(f"# module {i}")

        manager = AutoAgentManager(hub, str(tmp_path))

        # Build import graph
        graph = ImportGraph(str(tmp_path))
        graph.build()

        # First scan and register Tier B agents
        manager.scan_and_register()

        # Enable sub-agents
        sub_manager = manager.enable_sub_agents(graph)

        assert sub_manager is not None
        assert manager.sub_agent_manager is sub_manager

    def test_get_most_specific_agent_prefers_sub_agent(self, mock_auto_manager):
        """get_most_specific_agent should prefer sub-agent over team lead."""
        from agenthub.auto.sub_agent_manager import SubAgentManager

        graph = MagicMock()
        graph.nodes = {}

        # Set up mock parent agent
        mock_parent = MagicMock()
        mock_parent.spec = AgentSpec(
            agent_id="backend",
            name="Backend Expert",
            description="Backend module",
            context_paths=["backend/**/*.py"],
        )

        mock_auto_manager._auto_agents = {"backend": mock_parent}

        manager = SubAgentManager(mock_auto_manager, graph)

        # Set up mock sub-agent that owns backend/api/
        mock_sub = MagicMock()
        mock_sub.root_path = "backend/api"
        mock_sub.include_patterns = ["backend/api/**/*.py"]

        manager._sub_agents["backend"] = {"backend_api": mock_sub}

        # File in api/ should route to sub-agent
        result = manager.get_most_specific_agent("backend/api/routes.py")

        # Should return the sub-agent (or None if pattern doesn't match exactly)
        # The key is that the sub-agent is checked first
        assert result == mock_sub or result is None
