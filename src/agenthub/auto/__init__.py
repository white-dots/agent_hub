from __future__ import annotations
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

from pathlib import Path
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
from agenthub.auto.cross_context import (
    CrossAgentContextManager,
    CrossContextConfig,
    InjectedContext,
    format_injected_context,
)
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
from agenthub.auto.routing_index import (
    RoutingIndex,
    RoutingIndexBuilder,
    IndexedKeywordRouter,
    AgentRoutingMetadata,
)

if TYPE_CHECKING:
    from agenthub.hub import AgentHub

from agenthub.models import AgentSpec


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

        # Store import graph on hub for team execution support
        # The import graph is built during create_agents() with dynamic domains
        if factory._import_graph is not None:
            hub._import_graph = factory._import_graph

        # Store project root on hub for dashboard access
        hub._project_root = str(Path(project_root).resolve())

        summary_parts.append(f"\nTier B agents created: {tier_b_count}")

        # Step 3: Create sub-agents for large Tier B agents
        sub_agent_count = 0
        if factory._import_graph is not None:
            from agenthub.auto.sub_agent_manager import SubAgentManager
            from agenthub.auto.sub_agent_policy import SubAgentPolicy
            import json

            # Load policy settings from config file (set via dashboard)
            policy_config = {
                "min_files_to_split": 40,
                "min_subdirs_to_split": 2,
                "min_files_per_sub": 8,
                "max_sub_agents": 6,
            }
            try:
                config_file = Path.home() / ".agenthub" / "sub_agent_policy.json"
                if config_file.exists():
                    saved = json.loads(config_file.read_text())
                    policy_config.update(saved)
            except Exception:
                pass  # Use defaults if loading fails

            policy = SubAgentPolicy(
                min_files_to_split=policy_config["min_files_to_split"],
                min_subdirs_to_split=policy_config["min_subdirs_to_split"],
                min_files_per_sub=policy_config["min_files_per_sub"],
                max_sub_agents=policy_config["max_sub_agents"],
            )

            for agent in agents:
                if policy.should_subdivide(agent, factory._import_graph):
                    boundaries = policy.propose_subdivisions(agent, factory._import_graph)
                    for boundary in boundaries:
                        try:
                            from agenthub.auto.sub_agent_manager import SubCodeAgent

                            # Extract name from sub_agent_id (e.g., "backend_api" -> "Backend Api")
                            sub_name = boundary.sub_agent_id.replace("_", " ").title()

                            # Extract keywords from root_path and key_modules
                            keywords = []
                            if boundary.root_path:
                                keywords.extend(boundary.root_path.split("/"))
                            keywords.extend([Path(m).stem for m in boundary.key_modules[:3]])

                            sub_spec = AgentSpec(
                                agent_id=boundary.sub_agent_id,
                                name=f"{sub_name} Expert",
                                description=boundary.role_description or f"Expert on {boundary.root_path}",
                                context_keywords=keywords,
                                context_paths=boundary.include_patterns or [f"{boundary.root_path}/**/*.py"],
                                max_context_size=config.max_agent_context_kb * 1024 if config else 80 * 1024,
                                metadata={
                                    "tier": "B",
                                    "auto_generated": True,
                                    "parent_agent": agent.spec.agent_id,
                                    "is_sub_agent": True,
                                },
                            )
                            sub_agent = SubCodeAgent(
                                sub_spec,
                                client,
                                boundary.root_path,
                                boundary.include_patterns,
                                agent.spec.agent_id,
                            )
                            hub.register(sub_agent)
                            sub_agent_count += 1
                        except Exception:
                            pass  # Skip failed sub-agents

            if sub_agent_count > 0:
                summary_parts.append(f"Sub-agents created: {sub_agent_count}")

    # Summary header
    total = tier_a_count + tier_b_count
    summary_parts.insert(0, f"AgentHub initialized: {total} total agents")
    summary_parts.insert(1, "=" * 40)

    # Add agent tree visualization
    project_name = Path(project_root).name
    summary_parts.append("")
    summary_parts.append("Agent Coverage Tree:")
    summary_parts.append("-" * 40)
    summary_parts.append(print_agent_tree(hub, project_name))

    # Step 4: Build and store routing index for fast query-time routing
    # This is done at setup time so queries don't pay the indexing cost
    try:
        all_agent_specs = hub.list_agents()
        if all_agent_specs:
            index_builder = RoutingIndexBuilder(project_root)

            # Try to load cached index first
            cached_index = index_builder.load()
            if cached_index and cached_index.agent_count == len(all_agent_specs):
                # Cache is valid - use it
                hub._routing_index = cached_index
                summary_parts.append("")
                summary_parts.append(f"Routing index loaded from cache ({len(all_agent_specs)} agents)")
            else:
                # Build fresh index
                routing_index = index_builder.build(all_agent_specs)
                hub._routing_index = routing_index

                # Save to cache for next time
                cache_path = index_builder.save(routing_index)
                summary_parts.append("")
                summary_parts.append(f"Routing index built ({len(all_agent_specs)} agents, {len(routing_index.keyword_to_agents)} keywords)")
    except Exception as e:
        summary_parts.append(f"Routing index build warning: {e}")

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
    # Cross-agent context sharing
    "CrossAgentContextManager",
    "CrossContextConfig",
    "InjectedContext",
    "format_injected_context",
    # Routing index (pre-computed at setup time)
    "RoutingIndex",
    "RoutingIndexBuilder",
    "IndexedKeywordRouter",
    "AgentRoutingMetadata",
]
