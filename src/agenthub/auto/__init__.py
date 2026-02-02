"""Auto-agent generation for existing codebases.

This module provides automatic agent generation based on codebase
structure analysis. Point it at any existing project and get
intelligent code agents instantly.

Three approaches available:

1. **Unified (Recommended)**: Auto-discover Tier A business agents + generate Tier B code agents
   >>> from agenthub.auto import discover_all_agents
   >>> hub, summary = discover_all_agents("./my-project")
   >>> print(summary)
   >>> response = hub.run("How does pricing work?")

2. **Smart (Tier B only)**: Semantic analysis using AST parsing
   >>> from agenthub.auto import smart_enable_auto_agents
   >>> hub, summary = smart_enable_auto_agents("./my-project")
   >>> print(summary)
   >>> response = hub.run("How does authentication work?")

3. **Basic**: Folder-based analysis
   >>> from agenthub.auto import enable_auto_agents
   >>> agents = enable_auto_agents(hub, "./my-project")
"""

from typing import TYPE_CHECKING, Optional

from agenthub.auto.analyzer import AgentBoundary, CodebaseAnalyzer, FolderStats
from agenthub.auto.config import AutoAgentConfig, Presets
from agenthub.auto.discovery import CodebaseDiscovery, ModuleInfo, ProjectProfile
from agenthub.auto.factory import AutoAgentFactory, AutoCodeAgent
from agenthub.auto.manager import AutoAgentManager, print_coverage_map
from agenthub.auto.smart_factory import SmartAgentFactory, SmartCodeAgent
from agenthub.auto.import_graph import ImportGraph, ImportEdge, ModuleNode
from agenthub.auto.domain_analysis import Domain, DomainAnalysis
from agenthub.auto.dynamic_rnr import DynamicRnR, DynamicRnRGenerator
from agenthub.auto.tier_a_discovery import (
    TierADiscovery,
    DiscoveredAgent,
    DiscoveredFactory,
    TierADiscoveryResult,
    load_tier_a_agents,
    get_discovery_summary,
)
from agenthub.auto.tree import (
    print_agent_tree,
    build_agent_tree,
    get_routing_explanation,
    TreeNode,
)

if TYPE_CHECKING:
    from agenthub.hub import AgentHub


def discover_all_agents(
    project_root: str,
    config: Optional[AutoAgentConfig] = None,
    include_tier_a: bool = True,
    include_tier_b: bool = True,
) -> tuple["AgentHub", str]:
    """One-liner to create an AgentHub with BOTH Tier A and Tier B agents.

    This is the RECOMMENDED entry point. It:
    1. Creates an AgentHub
    2. Discovers Tier A agents from *_agents.py files (supports Claude & ChatGPT)
    3. Generates Tier B agents using AST-based code analysis
    4. Returns the hub ready to use

    Args:
        project_root: Path to existing codebase.
        config: Optional configuration for Tier B agents.
        include_tier_a: Whether to discover Tier A agents (default True).
        include_tier_b: Whether to generate Tier B agents (default True).

    Returns:
        Tuple of (configured AgentHub, summary string).

    Example:
        >>> from agenthub.auto import discover_all_agents
        >>>
        >>> # One line to get started with ALL agents
        >>> hub, summary = discover_all_agents("./my-smartstore-project")
        >>> print(summary)
        >>>
        >>> # Business questions go to Tier A agents
        >>> response = hub.run("What's the optimal discount rate?")
        >>> print(response.content)
        >>>
        >>> # Code questions go to Tier B agents
        >>> response = hub.run("How does the API authentication work?")
        >>> print(response.content)
    """
    import anthropic
    from agenthub.hub import AgentHub

    # Create hub and client
    client = anthropic.Anthropic()
    hub = AgentHub(client=client)

    summary_parts = []

    # Step 1: Discover and register Tier A agents
    tier_a_count = 0
    if include_tier_a:
        try:
            tier_a_agents = load_tier_a_agents(project_root)
            for agent in tier_a_agents:
                try:
                    hub.register(agent)
                    tier_a_count += 1
                except ValueError:
                    pass  # Already registered

            if tier_a_count > 0:
                summary_parts.append(f"Tier A (Business): {tier_a_count} agents discovered")
                for agent in tier_a_agents:
                    provider = agent.spec.metadata.get("llm_provider", "claude")
                    summary_parts.append(f"  - {agent.spec.name} [{provider}]")
        except Exception as e:
            summary_parts.append(f"Tier A discovery error: {e}")

    # Step 2: Generate Tier B agents
    tier_b_count = 0
    if include_tier_b:
        # Collect Tier A agent specs for context in dynamic domain detection
        # Note: hub.list_agents() returns AgentSpec objects, not BaseAgent
        tier_a_specs = [a for a in hub.list_agents() if a.metadata.get("tier") == "A"]

        factory = SmartAgentFactory(
            client,
            project_root,
            config,
            use_dynamic_domains=True,  # Use dynamic domain detection by default
            tier_a_agents=tier_a_specs,
        )

        # Show dynamic analysis summary
        tier_b_summary = factory.get_dynamic_summary()
        summary_parts.append("")
        summary_parts.append("Tier B (Code) Analysis - Dynamic Domains:")
        summary_parts.append(tier_b_summary)

        agents = factory.create_agents()
        for agent in agents:
            try:
                hub.register(agent)
                tier_b_count += 1
            except ValueError:
                pass  # Already registered

        summary_parts.append(f"\nTier B agents created: {tier_b_count}")

    # Summary header
    total = tier_a_count + tier_b_count
    summary_parts.insert(0, f"AgentHub initialized: {total} total agents")
    summary_parts.insert(1, "=" * 40)

    # Add agent tree visualization
    from pathlib import Path
    project_name = Path(project_root).name
    summary_parts.append("")
    summary_parts.append("Agent Coverage Tree:")
    summary_parts.append("-" * 40)
    summary_parts.append(print_agent_tree(hub, project_name))

    return hub, "\n".join(summary_parts)


def smart_enable_auto_agents(
    project_root: str,
    config: Optional[AutoAgentConfig] = None,
) -> tuple["AgentHub", str]:
    """One-liner to create an intelligent AgentHub for any codebase.

    This is the RECOMMENDED way to use auto-agents. It:
    1. Creates an AgentHub
    2. Analyzes the codebase using AST parsing
    3. Detects framework, modules, models, endpoints
    4. Creates specialized agents by module type (API, service, model, etc.)
    5. Returns the hub ready to use

    Args:
        project_root: Path to existing codebase.
        config: Optional configuration.

    Returns:
        Tuple of (configured AgentHub, project summary string).

    Example:
        >>> from agenthub.auto import smart_enable_auto_agents
        >>>
        >>> # One line to get started
        >>> hub, summary = smart_enable_auto_agents("./my-fastapi-project")
        >>> print(summary)
        >>>
        >>> # Start querying immediately
        >>> response = hub.run("How does user authentication work?")
        >>> print(response.content)
        >>>
        >>> response = hub.run("What database models exist?")
        >>> print(response.content)
    """
    import anthropic
    from agenthub.hub import AgentHub

    # Create hub and client
    client = anthropic.Anthropic()
    hub = AgentHub(client=client)

    # Create smart factory and analyze
    factory = SmartAgentFactory(client, project_root, config)
    summary = factory.get_project_summary()

    # Create and register agents
    agents = factory.create_agents()
    for agent in agents:
        hub.register(agent)

    return hub, summary


def enable_auto_agents(
    hub: "AgentHub",
    project_root: str,
    config: Optional[AutoAgentConfig] = None,
) -> list[str]:
    """Enable auto-agents for an existing codebase (basic folder-based).

    For smarter semantic analysis, use `smart_enable_auto_agents` instead.

    Args:
        hub: AgentHub instance to register agents with.
        project_root: Path to existing codebase.
        config: Optional configuration (uses defaults if None).

    Returns:
        List of auto-generated agent IDs.
    """
    return hub.enable_auto_agents(project_root, config)


def analyze_codebase(project_root: str) -> ProjectProfile:
    """Analyze a codebase and return detailed profile.

    Useful for understanding a project before creating agents.

    Args:
        project_root: Path to codebase.

    Returns:
        ProjectProfile with framework, modules, models, endpoints.

    Example:
        >>> profile = analyze_codebase("./my-project")
        >>> print(f"Framework: {profile.framework}")
        >>> print(f"Modules: {len(profile.modules)}")
        >>> for m in profile.modules:
        ...     print(f"  {m.path}: {m.description}")
    """
    discovery = CodebaseDiscovery(project_root)
    return discovery.analyze()


__all__ = [
    # Recommended entry point (Tier A + Tier B)
    "discover_all_agents",
    # Tier B only entry points
    "smart_enable_auto_agents",
    "analyze_codebase",
    "enable_auto_agents",
    # Configuration
    "AutoAgentConfig",
    "Presets",
    # Tree visualization
    "print_agent_tree",
    "build_agent_tree",
    "get_routing_explanation",
    "TreeNode",
    # Tier A discovery
    "TierADiscovery",
    "DiscoveredAgent",
    "DiscoveredFactory",
    "TierADiscoveryResult",
    "load_tier_a_agents",
    "get_discovery_summary",
    # Tier B discovery (semantic analysis)
    "CodebaseDiscovery",
    "ProjectProfile",
    "ModuleInfo",
    # Dynamic domain detection (new)
    "ImportGraph",
    "ImportEdge",
    "ModuleNode",
    "Domain",
    "DomainAnalysis",
    "DynamicRnR",
    "DynamicRnRGenerator",
    # Smart factory
    "SmartAgentFactory",
    "SmartCodeAgent",
    # Manager
    "AutoAgentManager",
    "print_coverage_map",
    # Basic analyzer (folder-based)
    "CodebaseAnalyzer",
    "FolderStats",
    "AgentBoundary",
    # Basic factory
    "AutoAgentFactory",
    "AutoCodeAgent",
]
