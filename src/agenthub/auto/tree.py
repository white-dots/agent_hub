"""Logical tree visualization for agent coverage.

This module provides tree visualization to show how agents
map to the codebase structure.

Example output (ASCII mode - Windows compatible):
    [P] smartstore/ --- 6 agents total
    +-- [A] Tier A: Business Agents --- 5 domain experts
    |   +-- * pricing_agent --- Pricing strategies, discount optimization
    |   |   +-- # Keywords: price, pricing, discount, margin
    |   +-- * naver_api_agent --- Naver Commerce API integration
    |   +-- * analytics_agent --- Sales analytics, traffic analysis
    |
    +-- [B] Tier B: Code Agents --- 1 auto-generated
        +-- (svc) service_agent --- Business logic layer
            +-- - backend/app/services/best_price.py
            +-- - backend/app/services/ingestion.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agenthub.hub import AgentHub
    from agenthub.models import AgentSpec


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
            "agent": "*",
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
        }
    else:
        return {
            "project": "📦",
            "tier_a": "🤖",
            "tier_b": "🔧",
            "agent": "◆",
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
    # Get agents by tier
    all_specs = hub.list_agents()
    tier_a = [s for s in all_specs if not s.metadata.get("auto_generated")]
    tier_b = [s for s in all_specs if s.metadata.get("auto_generated")]

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

        # Group by module type
        by_type: dict[str, list["AgentSpec"]] = {}
        for spec in tier_b:
            module_type = spec.metadata.get("module_type", "other")
            by_type.setdefault(module_type, []).append(spec)

        for module_type, specs in sorted(by_type.items()):
            for spec in specs:
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

    lines = [
        "Query Routing Logic:",
        "=" * 40,
        "",
        "1. First, check Tier A (Business) agents:",
    ]

    for spec in tier_a:
        keywords = ", ".join(spec.context_keywords[:5])
        lines.append(f"   * {spec.name}: {keywords}")

    lines.append("")
    lines.append("2. Then, check Tier B (Code) agents:")

    for spec in tier_b:
        module_type = spec.metadata.get("module_type", "code")
        lines.append(f"   * {spec.name} ({module_type})")

    lines.append("")
    lines.append("3. Fallback: First registered agent")

    return "\n".join(lines)
