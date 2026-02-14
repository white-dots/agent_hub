from __future__ import annotations
"""Logical tree visualization for agent coverage.

This module provides tree visualization to show how agents
map to the codebase structure, including sub-agent hierarchies.

Example output (ASCII mode - Windows compatible):
    [P] smartstore/ --- 6 agents total
    +-- [A] Tier A: Business Agents --- 5 domain experts
    |   +-- * pricing_agent --- Pricing strategies, discount optimization
    |   |   +-- # Keywords: price, pricing, discount, margin
    |   +-- * naver_api_agent --- Naver Commerce API integration
    |   +-- * analytics_agent --- Sales analytics, traffic analysis
    |
    +-- [B] Tier B: Code Agents --- 4 auto-generated
        +-- [TL] backend_agent --- Backend code (Team Lead)
        |   +-- (api) backend_api_agent --- API endpoints
        |   +-- (svc) backend_services_agent --- Services layer
        |   +-- (mod) backend_models_agent --- Data models
        +-- (svc) frontend_agent --- Frontend components
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agenthub.hub import AgentHub
    from agenthub.models import AgentSpec
    from agenthub.auto.sub_agent_manager import SubAgentManager


@dataclass
class TreeNode:
    """A node in the agent tree."""

    name: str
    description: str = ""
    children: list["TreeNode"] = None
    icon: str = ""
    is_file: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []


def _get_icons(use_ascii: bool = True) -> dict[str, str]:
    """Get icon set based on mode.

    Args:
        use_ascii: Use ASCII characters (Windows compatible). Default True.

    Returns:
        Dict mapping icon names to characters.
    """
    if use_ascii:
        return {
            "project": "[P]",
            "tier_a": "[A]",
            "tier_b": "[B]",
            "tier_c": "[C]",
            "agent": "*",
            "team_lead": "[TL]",
            "sub_agent": "->",
            "tag": "#",
            "file": "-",
            "api": "(api)",
            "service": "(svc)",
            "model": "(mod)",
            "repository": "(repo)",
            "util": "(util)",
            "config": "(cfg)",
            "test": "(test)",
            "default": "(code)",
            "meta": "(meta)",
        }
    else:
        return {
            "project": "📦",
            "tier_a": "🤖",
            "tier_b": "🔧",
            "tier_c": "🔮",
            "agent": "◆",
            "team_lead": "👥",
            "sub_agent": "└→",
            "tag": "🏷️",
            "file": "📄",
            "api": "🌐",
            "service": "⚙️",
            "model": "📊",
            "repository": "🗃️",
            "util": "🔨",
            "config": "⚙️",
            "test": "🧪",
            "default": "📁",
            "meta": "🔮",
        }


def _get_type_icon(module_type: str, use_ascii: bool = True) -> str:
    """Get icon for module type."""
    icons = _get_icons(use_ascii)
    return icons.get(module_type, icons["default"])


def build_agent_tree(
    hub: "AgentHub",
    project_name: Optional[str] = None,
    use_ascii: bool = True,
) -> TreeNode:
    """Build a tree representation of agents and their coverage.

    Args:
        hub: AgentHub instance with registered agents.
        project_name: Optional project name for root node.
        use_ascii: Use ASCII characters instead of emojis (Windows compatible).

    Returns:
        TreeNode representing the agent hierarchy.
    """
    # Get agents by tier - check explicit tier metadata first
    all_specs = hub.list_agents()
    tier_a = []
    tier_b = []
    tier_c = []

    for spec in all_specs:
        explicit_tier = spec.metadata.get("tier")
        if explicit_tier == "C":
            tier_c.append(spec)
        elif explicit_tier == "B" or spec.metadata.get("auto_generated"):
            tier_b.append(spec)
        elif explicit_tier == "A" or not spec.metadata.get("auto_generated"):
            tier_a.append(spec)

    # Icons
    icons = _get_icons(use_ascii)

    # Root node
    root = TreeNode(
        name=project_name or "Project",
        icon=icons["project"],
        description=f"{len(all_specs)} agents total",
    )

    # Tier A branch
    if tier_a:
        tier_a_node = TreeNode(
            name="Tier A: Business Agents",
            icon=icons["tier_a"],
            description=f"{len(tier_a)} domain experts",
        )
        for spec in tier_a:
            agent_node = TreeNode(
                name=spec.agent_id,
                icon=icons["agent"],
                description=spec.description[:60] if spec.description else "",
            )
            # Add keywords as context
            if spec.context_keywords:
                keywords = ", ".join(spec.context_keywords[:5])
                agent_node.children.append(
                    TreeNode(name=f"Keywords: {keywords}", icon=icons["tag"])
                )
            tier_a_node.children.append(agent_node)
        root.children.append(tier_a_node)

    # Tier B branch
    if tier_b:
        tier_b_node = TreeNode(
            name="Tier B: Code Agents",
            icon=icons["tier_b"],
            description=f"{len(tier_b)} auto-generated",
        )

        # Separate team leads from regular agents and sub-agents
        team_leads: list["AgentSpec"] = []
        sub_agents: dict[str, list["AgentSpec"]] = {}  # parent_id -> [sub_agents]
        regular_agents: list["AgentSpec"] = []

        for spec in tier_b:
            parent_id = spec.metadata.get("parent_agent_id")
            is_team_lead = spec.metadata.get("is_team_lead", False)

            if parent_id:
                # This is a sub-agent
                sub_agents.setdefault(parent_id, []).append(spec)
            elif is_team_lead:
                # This is a team lead
                team_leads.append(spec)
            else:
                # Regular agent
                regular_agents.append(spec)

        # Build nodes for team leads with their sub-agents
        for lead_spec in team_leads:
            lead_node = TreeNode(
                name=lead_spec.agent_id,
                icon=icons["team_lead"],
                description=f"{lead_spec.description[:40]}... (Team Lead)" if lead_spec.description else "(Team Lead)",
            )

            # Add sub-agents under this team lead
            lead_sub_agents = sub_agents.get(lead_spec.agent_id, [])
            for sub_spec in lead_sub_agents:
                module_type = sub_spec.metadata.get("module_type", "other")
                sub_node = TreeNode(
                    name=sub_spec.agent_id,
                    icon=_get_type_icon(module_type, use_ascii),
                    description=sub_spec.description[:50] if sub_spec.description else "",
                )
                # Add file paths for sub-agent
                for path in sub_spec.context_paths[:3]:
                    sub_node.children.append(
                        TreeNode(name=path, icon=icons["file"], is_file=True)
                    )
                if len(sub_spec.context_paths) > 3:
                    sub_node.children.append(
                        TreeNode(
                            name=f"+{len(sub_spec.context_paths) - 3} more files",
                            icon="...",
                        )
                    )
                lead_node.children.append(sub_node)

            tier_b_node.children.append(lead_node)

        # Build nodes for regular agents (not team leads, not sub-agents)
        for spec in regular_agents:
            module_type = spec.metadata.get("module_type", "other")
            agent_node = TreeNode(
                name=spec.agent_id,
                icon=_get_type_icon(module_type, use_ascii),
                description=spec.description[:50] if spec.description else "",
            )
            # Add file paths
            for path in spec.context_paths[:5]:
                agent_node.children.append(
                    TreeNode(name=path, icon=icons["file"], is_file=True)
                )
            if len(spec.context_paths) > 5:
                agent_node.children.append(
                    TreeNode(
                        name=f"+{len(spec.context_paths) - 5} more files",
                        icon="...",
                    )
                )
            tier_b_node.children.append(agent_node)

        root.children.append(tier_b_node)

    # Tier C branch (meta-agents like QC)
    if tier_c:
        tier_c_node = TreeNode(
            name="Tier C: Meta Agents",
            icon=icons["tier_c"],
            description=f"{len(tier_c)} meta-agents",
        )
        for spec in tier_c:
            role = spec.metadata.get("role", "meta_agent")
            agent_node = TreeNode(
                name=spec.agent_id,
                icon=icons.get("meta", icons["agent"]),
                description=spec.description[:60] if spec.description else "",
            )
            # Add role info
            agent_node.children.append(
                TreeNode(name=f"Role: {role}", icon=icons["tag"])
            )
            tier_c_node.children.append(agent_node)
        root.children.append(tier_c_node)

    return root


def render_tree(node: TreeNode, prefix: str = "", is_last: bool = True) -> str:
    """Render a tree node as a string.

    Args:
        node: TreeNode to render.
        prefix: Current line prefix for indentation.
        is_last: Whether this is the last child.

    Returns:
        String representation of the tree.
    """
    lines = []

    # Current node - use ASCII tree characters
    connector = "+-- " if not is_last else "+-- "
    desc = f" --- {node.description}" if node.description else ""
    lines.append(f"{prefix}{connector}{node.icon} {node.name}{desc}")

    # Children
    child_prefix = prefix + ("|   " if not is_last else "    ")
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        lines.append(render_tree(child, child_prefix, is_last_child))

    return "\n".join(lines)


def print_agent_tree(
    hub: "AgentHub",
    project_name: Optional[str] = None,
    use_ascii: bool = True,
) -> str:
    """Print a visual tree of all agents and their coverage.

    This shows users how queries would be routed:
    - Tier A agents handle business/domain questions
    - Tier B agents handle code-specific questions

    Args:
        hub: AgentHub instance.
        project_name: Optional name for the project root.
        use_ascii: Use ASCII characters (Windows compatible). Default True.

    Returns:
        String representation of the tree.

    Example:
        >>> hub, _ = discover_all_agents("./my-project")
        >>> print(print_agent_tree(hub, "my-project"))
        [P] my-project/ --- 6 agents total
        +-- [A] Tier A: Business Agents --- 5 domain experts
        |   +-- * pricing_agent --- Pricing optimization
        |   +-- * analytics_agent --- Sales analytics
        +-- [B] Tier B: Code Agents --- 1 auto-generated
            +-- (api) api_agent --- API endpoints
            +-- (mod) model_agent --- Data models
    """
    tree = build_agent_tree(hub, project_name, use_ascii)

    # Render without the root connector
    lines = [f"{tree.icon} {tree.name}/ --- {tree.description}"]

    for i, child in enumerate(tree.children):
        is_last = i == len(tree.children) - 1
        lines.append(render_tree(child, "", is_last))

    return "\n".join(lines)


def get_routing_explanation(hub: "AgentHub") -> str:
    """Get a human-readable explanation of how routing works.

    Args:
        hub: AgentHub instance.

    Returns:
        Explanation string.
    """
    tier_a = hub.list_agents(tier="A")
    tier_b = hub.list_agents(tier="B")
    tier_c = hub.list_agents(tier="C")

    lines = [
        "Query Routing Logic:",
        "=" * 40,
        "",
        "1. First, check Tier A (Business) agents:",
    ]

    for spec in tier_a:
        keywords = ", ".join(spec.context_keywords[:5]) if spec.context_keywords else ""
        lines.append(f"   * {spec.name}: {keywords}")

    lines.append("")
    lines.append("2. Then, check Tier B (Code) agents:")

    for spec in tier_b:
        module_type = spec.metadata.get("module_type", "code")
        lines.append(f"   * {spec.name} ({module_type})")

    if tier_c:
        lines.append("")
        lines.append("3. Tier C (Meta) agents:")
        for spec in tier_c:
            role = spec.metadata.get("role", "meta_agent")
            lines.append(f"   * {spec.name} ({role})")

    lines.append("")
    lines.append(f"{4 if tier_c else 3}. Fallback: First registered agent")

    return "\n".join(lines)
