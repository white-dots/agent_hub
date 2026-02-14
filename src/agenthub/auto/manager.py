from __future__ import annotations
"""Manager for auto-generated agent lifecycle."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agenthub.auto.analyzer import CodebaseAnalyzer
from agenthub.auto.config import AutoAgentConfig
from agenthub.auto.factory import AutoAgentFactory

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent
    from agenthub.auto.import_graph import ImportGraph
    from agenthub.auto.sub_agent_manager import SubAgentManager
    from agenthub.auto.sub_agent_policy import SubAgentPolicy
    from agenthub.hub import AgentHub


class AutoAgentManager:
    """Manages lifecycle of auto-generated agents.

    Handles scanning the codebase, creating agents, and keeping
    them in sync as code changes.

    Example:
        >>> manager = AutoAgentManager(hub, "./project")
        >>> agent_ids = manager.scan_and_register()
        >>> print(f"Created {len(agent_ids)} agents")
        >>>
        >>> # After code changes
        >>> added, removed = manager.refresh()
    """

    def __init__(
        self,
        hub: "AgentHub",
        project_root: str,
        config: Optional[AutoAgentConfig] = None,
    ):
        """Initialize AutoAgentManager.

        Args:
            hub: AgentHub instance to register agents with.
            project_root: Path to the project to analyze.
            config: Auto-generation configuration.
        """
        self.hub = hub
        self.project_root = Path(project_root).resolve()
        self.config = config or AutoAgentConfig()

        self.analyzer = CodebaseAnalyzer(str(self.project_root), self.config)
        self.factory = AutoAgentFactory(hub.client, self.config)

        # Track auto-generated agents
        self._auto_agents: dict[str, "BaseAgent"] = {}
        self._last_scan: Optional[datetime] = None

        # Sub-agent support
        self._sub_agent_manager: Optional["SubAgentManager"] = None

    def scan_and_register(self) -> list[str]:
        """Scan codebase and register auto-generated agents.

        Returns:
            List of newly registered agent IDs.
        """
        boundaries = self.analyzer.propose_boundaries()
        new_agents: list[str] = []

        for boundary in boundaries:
            if boundary.agent_id not in self._auto_agents:
                try:
                    agent = self.factory.create_agent(boundary)
                    self.hub.register(agent)
                    self._auto_agents[boundary.agent_id] = agent
                    new_agents.append(boundary.agent_id)
                except Exception as e:
                    print(f"Warning: Failed to create agent {boundary.agent_id}: {e}")

        self._last_scan = datetime.now()
        return new_agents

    def refresh(self) -> tuple[list[str], list[str]]:
        """Re-scan and update agents.

        Returns:
            Tuple of (added_agent_ids, removed_agent_ids).
        """
        current_boundaries = {
            b.agent_id: b for b in self.analyzer.propose_boundaries()
        }

        added: list[str] = []
        removed: list[str] = []

        # Find new agents
        for agent_id, boundary in current_boundaries.items():
            if agent_id not in self._auto_agents:
                try:
                    agent = self.factory.create_agent(boundary)
                    self.hub.register(agent)
                    self._auto_agents[agent_id] = agent
                    added.append(agent_id)
                except Exception as e:
                    print(f"Warning: Failed to create agent {agent_id}: {e}")

        # Find removed agents
        for agent_id in list(self._auto_agents.keys()):
            if agent_id not in current_boundaries:
                try:
                    self.hub.unregister(agent_id)
                    del self._auto_agents[agent_id]
                    removed.append(agent_id)
                except Exception:
                    pass

        # Refresh context for existing agents
        for agent_id, agent in self._auto_agents.items():
            if agent_id in current_boundaries:
                agent.get_context(force_refresh=True)

        self._last_scan = datetime.now()
        return added, removed

    def list_auto_agents(self) -> list[str]:
        """List all auto-generated agent IDs.

        Returns:
            List of agent IDs.
        """
        return list(self._auto_agents.keys())

    def get_coverage_report(self) -> dict:
        """Report on codebase coverage by auto-agents.

        Returns:
            Dict with coverage statistics.
        """
        stats = self.analyzer.analyze()

        total_kb = sum(s.total_size_kb for s in stats)
        total_files = sum(s.file_count for s in stats)

        # Calculate covered stats
        covered_paths = {
            Path(a.spec.metadata.get("root_path", ""))
            for a in self._auto_agents.values()
        }

        covered_kb = 0.0
        covered_files = 0
        covered_folders = 0

        for s in stats:
            if s.path in covered_paths or any(s.path.is_relative_to(p) for p in covered_paths if p.exists()):
                covered_kb += s.total_size_kb
                covered_files += s.file_count
                covered_folders += 1

        return {
            "total_folders": len(stats),
            "covered_folders": covered_folders,
            "total_files": total_files,
            "covered_files": covered_files,
            "total_kb": total_kb,
            "covered_kb": covered_kb,
            "coverage_percent": (covered_kb / total_kb * 100) if total_kb > 0 else 0,
            "auto_agents": len(self._auto_agents),
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }

    def unregister_all(self) -> int:
        """Unregister all auto-generated agents.

        Returns:
            Number of agents unregistered.
        """
        # Unregister sub-agents first
        sub_count = 0
        if self._sub_agent_manager:
            sub_count = self._sub_agent_manager.unregister_all_sub_agents()

        count = 0
        for agent_id in list(self._auto_agents.keys()):
            try:
                self.hub.unregister(agent_id)
                del self._auto_agents[agent_id]
                count += 1
            except Exception:
                pass
        return count + sub_count

    # === Sub-Agent Support ===

    def enable_sub_agents(
        self,
        import_graph: "ImportGraph",
        policy: Optional["SubAgentPolicy"] = None,
    ) -> "SubAgentManager":
        """Enable sub-agent subdivision for large Tier B agents.

        This analyzes all Tier B agents and subdivides those that are
        too large (based on the policy thresholds) into focused sub-agents.

        Args:
            import_graph: ImportGraph for clustering analysis.
            policy: SubAgentPolicy for subdivision rules. Uses defaults if None.

        Returns:
            SubAgentManager instance for managing sub-agents.

        Example:
            >>> from agenthub.auto.import_graph import ImportGraph
            >>> graph = ImportGraph(project_root)
            >>> graph.build()
            >>> sub_manager = auto_manager.enable_sub_agents(graph)
            >>> print(sub_manager.get_hierarchy_report())
        """
        from agenthub.auto.sub_agent_manager import SubAgentManager
        from agenthub.auto.sub_agent_policy import SubAgentPolicy

        self._sub_agent_manager = SubAgentManager(
            auto_manager=self,
            import_graph=import_graph,
            policy=policy or SubAgentPolicy(),
        )
        self._sub_agent_manager.evaluate_and_subdivide()
        return self._sub_agent_manager

    def get_most_specific_agent(self, file_path: str) -> Optional["BaseAgent"]:
        """Get the most specific agent owning a file.

        Checks sub-agents first (most specific), then Tier B agents.

        Args:
            file_path: Path to the file.

        Returns:
            Most specific agent owning this file, or None.
        """
        # Check sub-agents first
        if self._sub_agent_manager:
            sub_agent = self._sub_agent_manager.get_most_specific_agent(file_path)
            if sub_agent:
                return sub_agent

        # Fall back to Tier B agents
        return self._get_agent_for_file(file_path)

    def _get_agent_for_file(self, file_path: str) -> Optional["BaseAgent"]:
        """Get the Tier B agent that owns a file."""
        for agent_id, agent in self._auto_agents.items():
            if self._file_in_agent_scope(file_path, agent):
                return agent
        return None

    def _file_in_agent_scope(self, file_path: str, agent: "BaseAgent") -> bool:
        """Check if a file falls within an agent's context_paths."""
        import fnmatch

        file_path = file_path.replace("\\", "/")

        for context_path in agent.spec.context_paths:
            context_path = context_path.replace("\\", "/")

            # Glob pattern match
            if fnmatch.fnmatch(file_path, context_path):
                return True

            # Directory prefix match
            if context_path.endswith("/*") or context_path.endswith("/**"):
                dir_prefix = context_path.rstrip("/*")
                if file_path.startswith(dir_prefix):
                    return True

            # Simple prefix match
            if file_path.startswith(context_path.rstrip("/")):
                return True

        return False

    @property
    def sub_agent_manager(self) -> Optional["SubAgentManager"]:
        """Get the sub-agent manager if enabled."""
        return self._sub_agent_manager


def print_coverage_map(hub: "AgentHub", project_root: str) -> None:
    """Print visual map of agent coverage.

    Args:
        hub: AgentHub instance.
        project_root: Project root path.
    """
    if not hub._auto_manager:
        print("Auto-agents not enabled")
        return

    report = hub._auto_manager.get_coverage_report()

    print("\n Agent Coverage Report")
    print("=" * 50)
    print(f"Total: {report['total_folders']} folders, {report['total_kb']:.0f}KB")
    print(f"Covered: {report['covered_folders']} folders, {report['covered_kb']:.0f}KB")
    print(f"Coverage: {report['coverage_percent']:.1f}%")
    print()

    # List agents by tier
    tier_a = hub.list_agents(tier="A")
    tier_b = hub.list_agents(tier="B")

    if tier_a:
        print(f"[A] Tier A (Business): {len(tier_a)} agents")
        for agent in tier_a:
            print(f"    * {agent.agent_id}: {agent.description}")
        print()

    if tier_b:
        print(f"[B] Tier B (Auto-Code): {len(tier_b)} agents")
        for agent in tier_b:
            print(f"    * {agent.agent_id}: {agent.description}")
