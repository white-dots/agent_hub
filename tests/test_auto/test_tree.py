"""Tests for tree visualization with sub-agent support."""

from unittest.mock import MagicMock

import pytest

from agenthub.auto.tree import (
    TreeNode,
    build_agent_tree,
    render_tree,
    print_agent_tree,
    get_routing_explanation,
    _get_icons,
    _get_type_icon,
)


# === Helper Functions ===


def create_mock_agent_spec(
    agent_id: str,
    description: str = "",
    tier: str = None,
    auto_generated: bool = False,
    module_type: str = "default",
    is_team_lead: bool = False,
    parent_agent_id: str = None,
    context_paths: list = None,
    context_keywords: list = None,
):
    """Create a mock AgentSpec for testing."""
    spec = MagicMock()
    spec.agent_id = agent_id
    spec.name = agent_id
    spec.description = description
    spec.context_paths = context_paths or []
    spec.context_keywords = context_keywords or []
    spec.metadata = {
        "auto_generated": auto_generated,
        "module_type": module_type,
        "is_team_lead": is_team_lead,
    }
    if tier:
        spec.metadata["tier"] = tier
    if parent_agent_id:
        spec.metadata["parent_agent_id"] = parent_agent_id
    return spec


# === TreeNode Tests ===


class TestTreeNode:
    """Tests for TreeNode dataclass."""

    def test_create_tree_node(self):
        """Should create TreeNode with default values."""
        node = TreeNode(name="test")
        assert node.name == "test"
        assert node.description == ""
        assert node.children == []
        assert node.icon == ""
        assert node.is_file is False

    def test_tree_node_with_children(self):
        """Should create TreeNode with children."""
        child = TreeNode(name="child")
        parent = TreeNode(name="parent", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].name == "child"

    def test_tree_node_post_init_none_children(self):
        """Should initialize None children to empty list."""
        node = TreeNode(name="test", children=None)
        assert node.children == []


# === Icon Tests ===


class TestIcons:
    """Tests for icon functions."""

    def test_get_ascii_icons(self):
        """Should return ASCII icons by default."""
        icons = _get_icons(use_ascii=True)
        assert icons["project"] == "[P]"
        assert icons["team_lead"] == "[TL]"
        assert icons["sub_agent"] == "->"

    def test_get_unicode_icons(self):
        """Should return Unicode icons when requested."""
        icons = _get_icons(use_ascii=False)
        assert "📦" in icons["project"]
        assert "👥" in icons["team_lead"]

    def test_get_type_icon_known_type(self):
        """Should return correct icon for known types."""
        assert _get_type_icon("api", use_ascii=True) == "(api)"
        assert _get_type_icon("service", use_ascii=True) == "(svc)"
        assert _get_type_icon("model", use_ascii=True) == "(mod)"

    def test_get_type_icon_unknown_type(self):
        """Should return default icon for unknown types."""
        assert _get_type_icon("unknown", use_ascii=True) == "(code)"


# === Build Agent Tree Tests ===


class TestBuildAgentTree:
    """Tests for build_agent_tree function."""

    @pytest.fixture
    def mock_hub(self):
        """Create mock AgentHub."""
        hub = MagicMock()
        hub.list_agents.return_value = []
        return hub

    def test_empty_hub(self, mock_hub):
        """Should build tree with no agents."""
        tree = build_agent_tree(mock_hub, "test-project")
        assert tree.name == "test-project"
        assert "0 agents total" in tree.description
        assert len(tree.children) == 0

    def test_tier_a_agents(self, mock_hub):
        """Should include Tier A agents."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "pricing_agent",
                "Pricing optimization",
                tier="A",
                context_keywords=["price", "discount"],
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")
        assert len(tree.children) == 1
        tier_a_node = tree.children[0]
        assert "Tier A" in tier_a_node.name
        assert len(tier_a_node.children) == 1
        assert tier_a_node.children[0].name == "pricing_agent"

    def test_tier_b_agents(self, mock_hub):
        """Should include Tier B agents."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "api_agent",
                "API endpoints",
                tier="B",
                auto_generated=True,
                module_type="api",
                context_paths=["src/api/routes.py"],
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")
        assert len(tree.children) == 1
        tier_b_node = tree.children[0]
        assert "Tier B" in tier_b_node.name
        assert len(tier_b_node.children) == 1

    def test_tier_c_meta_agents(self, mock_hub):
        """Should include Tier C meta agents."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "qc_agent",
                "Quality control agent",
                tier="C",
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")
        assert len(tree.children) == 1
        tier_c_node = tree.children[0]
        assert "Tier C" in tier_c_node.name

    def test_team_lead_with_sub_agents(self, mock_hub):
        """Should nest sub-agents under team leads."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "backend_agent",
                "Backend code",
                tier="B",
                auto_generated=True,
                is_team_lead=True,
            ),
            create_mock_agent_spec(
                "backend_api_agent",
                "API endpoints",
                tier="B",
                auto_generated=True,
                module_type="api",
                parent_agent_id="backend_agent",
                context_paths=["src/api/routes.py"],
            ),
            create_mock_agent_spec(
                "backend_models_agent",
                "Data models",
                tier="B",
                auto_generated=True,
                module_type="model",
                parent_agent_id="backend_agent",
                context_paths=["src/models/user.py"],
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")

        # Find Tier B node
        tier_b_node = next(n for n in tree.children if "Tier B" in n.name)

        # Should have 1 team lead
        assert len(tier_b_node.children) == 1
        team_lead = tier_b_node.children[0]
        assert team_lead.name == "backend_agent"
        assert "Team Lead" in team_lead.description

        # Team lead should have 2 sub-agents
        assert len(team_lead.children) == 2

    def test_regular_agents_separate_from_sub_agents(self, mock_hub):
        """Regular agents should not be nested under team leads."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "backend_agent",
                "Backend code",
                tier="B",
                auto_generated=True,
                is_team_lead=True,
            ),
            create_mock_agent_spec(
                "frontend_agent",
                "Frontend code",
                tier="B",
                auto_generated=True,
                module_type="default",
                # No parent_agent_id, not a team lead
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")
        tier_b_node = next(n for n in tree.children if "Tier B" in n.name)

        # Should have team lead and regular agent at same level
        assert len(tier_b_node.children) == 2

    def test_file_paths_shown_for_sub_agents(self, mock_hub):
        """Sub-agents should show their file paths."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "api_agent",
                "API endpoints",
                tier="B",
                auto_generated=True,
                module_type="api",
                context_paths=["src/api/routes.py", "src/api/auth.py", "src/api/users.py"],
            ),
        ]

        tree = build_agent_tree(mock_hub, "test-project")
        tier_b_node = tree.children[0]
        agent_node = tier_b_node.children[0]

        # Should have file path children
        file_children = [c for c in agent_node.children if c.is_file]
        assert len(file_children) == 3


# === Render Tree Tests ===


class TestRenderTree:
    """Tests for render_tree function."""

    def test_render_simple_node(self):
        """Should render a simple node."""
        node = TreeNode(name="test", icon="*", description="desc")
        output = render_tree(node)
        assert "*" in output
        assert "test" in output
        assert "desc" in output

    def test_render_with_children(self):
        """Should render children with proper indentation."""
        child = TreeNode(name="child", icon="-")
        parent = TreeNode(name="parent", icon="*", children=[child])
        output = render_tree(parent)
        assert "parent" in output
        assert "child" in output
        assert "+--" in output


# === Print Agent Tree Tests ===


class TestPrintAgentTree:
    """Tests for print_agent_tree function."""

    @pytest.fixture
    def mock_hub(self):
        """Create mock AgentHub."""
        hub = MagicMock()
        hub.list_agents.return_value = []
        return hub

    def test_print_empty_tree(self, mock_hub):
        """Should print tree header for empty hub."""
        output = print_agent_tree(mock_hub, "test-project")
        assert "[P]" in output
        assert "test-project" in output
        assert "0 agents total" in output

    def test_print_with_agents(self, mock_hub):
        """Should print full tree with agents."""
        mock_hub.list_agents.return_value = [
            create_mock_agent_spec(
                "pricing_agent",
                "Pricing optimization",
                tier="A",
            ),
            create_mock_agent_spec(
                "api_agent",
                "API endpoints",
                tier="B",
                auto_generated=True,
            ),
        ]

        output = print_agent_tree(mock_hub, "test-project")
        assert "[A]" in output
        assert "[B]" in output
        assert "pricing_agent" in output
        assert "api_agent" in output


# === Get Routing Explanation Tests ===


class TestGetRoutingExplanation:
    """Tests for get_routing_explanation function."""

    @pytest.fixture
    def mock_hub(self):
        """Create mock AgentHub with tier filtering."""
        hub = MagicMock()

        def list_agents(tier=None):
            all_agents = [
                create_mock_agent_spec(
                    "pricing_agent",
                    tier="A",
                    context_keywords=["price", "discount"],
                ),
                create_mock_agent_spec(
                    "api_agent",
                    tier="B",
                    auto_generated=True,
                    module_type="api",
                ),
            ]
            if tier == "A":
                return [a for a in all_agents if a.metadata.get("tier") == "A"]
            elif tier == "B":
                return [a for a in all_agents if a.metadata.get("tier") == "B"]
            elif tier == "C":
                return []
            return all_agents

        hub.list_agents = list_agents
        return hub

    def test_routing_explanation_format(self, mock_hub):
        """Should return formatted routing explanation."""
        output = get_routing_explanation(mock_hub)
        assert "Query Routing Logic" in output
        assert "Tier A" in output
        assert "Tier B" in output

    def test_routing_shows_keywords(self, mock_hub):
        """Should show keywords for Tier A agents."""
        output = get_routing_explanation(mock_hub)
        assert "price" in output or "pricing_agent" in output
