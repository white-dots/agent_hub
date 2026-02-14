from __future__ import annotations
"""Manager for sub-Tier B agents.

This module handles the lifecycle of sub-agents — specialized team members
that are created when a Tier B agent's domain is too large for effective
handling by a single agent.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agenthub.agents.base import BaseAgent
from agenthub.auto.sub_agent_policy import SubAgentPolicy
from agenthub.context import ContextBuilder
from agenthub.models import AgentResponse, AgentSpec, Session, SubAgentBoundary

if TYPE_CHECKING:
    import anthropic

    from agenthub.auto.import_graph import ImportGraph
    from agenthub.auto.manager import AutoAgentManager


class SubCodeAgent(BaseAgent):
    """Auto-generated sub-agent for a focused sub-domain.

    Similar to AutoCodeAgent but with additional hierarchy awareness
    and scoped to a smaller portion of a parent agent's domain.
    """

    def __init__(
        self,
        spec: AgentSpec,
        client: "anthropic.Anthropic",
        root_path: str,
        include_patterns: list[str],
        parent_agent_id: str,
    ):
        """Initialize SubCodeAgent.

        Args:
            spec: Agent specification.
            client: Anthropic client for API calls.
            root_path: Root path this sub-agent covers.
            include_patterns: Glob patterns for files to include.
            parent_agent_id: ID of the parent team lead agent.
        """
        super().__init__(spec, client)
        self.root_path = root_path
        self.include_patterns = include_patterns
        self.parent_agent_id = parent_agent_id
        self.context_builder = ContextBuilder(root_path)

    def build_context(self) -> str:
        """Build context from the sub-agent's module files."""
        parts: list[str] = []

        # Note hierarchy
        parts.append(f"## Sub-Domain Context (Parent: {self.parent_agent_id})")

        # Directory structure
        parts.append("\n### Module Structure\n```")
        parts.append(self.context_builder.read_directory_structure(max_depth=2))
        parts.append("```")

        # Source files
        parts.append("\n### Source Code")
        parts.append(
            self.context_builder.read_files(
                patterns=self.include_patterns,
                max_size=self.spec.max_context_size,
            )
        )

        return "\n".join(parts)


class SubAgentManager:
    """Manages sub-Tier B agent lifecycle.

    Handles:
    - Evaluating which Tier B agents should be subdivided
    - Creating sub-agents from subdivision boundaries
    - Routing queries and files to the most specific sub-agent
    - Managing the team lead → sub-agent hierarchy

    Example:
        >>> manager = SubAgentManager(auto_manager, import_graph)
        >>> report = manager.evaluate_and_subdivide()
        >>> print(f"Created {sum(len(v) for v in report.values())} sub-agents")
        >>>
        >>> # Route a file to its owning sub-agent
        >>> agent = manager.route_to_sub_agent("backend", "backend/api/routes.py")
        >>> if agent:
        ...     response = agent.run("Explain this module", session)
    """

    def __init__(
        self,
        auto_manager: "AutoAgentManager",
        import_graph: "ImportGraph",
        policy: Optional[SubAgentPolicy] = None,
    ):
        """Initialize SubAgentManager.

        Args:
            auto_manager: AutoAgentManager that manages parent Tier B agents.
            import_graph: Import graph for clustering analysis.
            policy: SubAgentPolicy for subdivision rules.
        """
        self._auto_manager = auto_manager
        self._graph = import_graph
        self._policy = policy or SubAgentPolicy()

        # Track sub-agents by parent
        # parent_agent_id → {sub_agent_id: SubCodeAgent}
        self._sub_agents: dict[str, dict[str, SubCodeAgent]] = {}

        # Track sub-agent boundaries for routing
        # sub_agent_id → SubAgentBoundary
        self._boundaries: dict[str, SubAgentBoundary] = {}

        # Reverse mapping: parent → list of sub-agent IDs
        self._parent_to_children: dict[str, list[str]] = {}

    def evaluate_and_subdivide(self) -> dict[str, list[SubAgentBoundary]]:
        """Scan all Tier B agents and subdivide where needed.

        Flow:
        1. For each Tier B agent:
           a. Check policy.should_subdivide()
           b. If yes: propose_subdivisions()
           c. Create sub-agents from boundaries
           d. Update parent's children_ids and is_team_lead
           e. Register sub-agents with hub
        2. Return report of subdivisions

        Returns:
            Dict mapping parent_agent_id to list of SubAgentBoundary created.
        """
        report: dict[str, list[SubAgentBoundary]] = {}

        # Get all Tier B agents
        tier_b_agents = self._auto_manager.hub.list_agents(tier="B")

        for agent_spec in tier_b_agents:
            agent_id = agent_spec.agent_id

            # Skip if already a sub-agent (has parent)
            if agent_spec.parent_agent_id is not None:
                continue

            # Get the actual agent instance
            agent = self._auto_manager._auto_agents.get(agent_id)
            if agent is None:
                continue

            # Check if subdivision is warranted
            if not self._policy.should_subdivide(agent, self._graph):
                continue

            # Propose subdivisions
            boundaries = self._policy.propose_subdivisions(agent, self._graph)

            if not boundaries:
                continue

            # Create sub-agents
            created_boundaries: list[SubAgentBoundary] = []
            children_ids: list[str] = []

            for boundary in boundaries:
                try:
                    sub_agent = self._create_sub_agent(boundary)

                    # Track sub-agent
                    if agent_id not in self._sub_agents:
                        self._sub_agents[agent_id] = {}
                    self._sub_agents[agent_id][boundary.sub_agent_id] = sub_agent
                    self._boundaries[boundary.sub_agent_id] = boundary
                    children_ids.append(boundary.sub_agent_id)

                    # Register with hub (but not auto_manager tracking)
                    self._auto_manager.hub.register(sub_agent)

                    created_boundaries.append(boundary)
                except Exception as e:
                    print(f"Warning: Failed to create sub-agent {boundary.sub_agent_id}: {e}")

            # Update parent agent's hierarchy fields
            if children_ids:
                agent.spec.children_ids = children_ids
                agent.spec.is_team_lead = True
                self._parent_to_children[agent_id] = children_ids
                report[agent_id] = created_boundaries

        return report

    def _create_sub_agent(self, boundary: SubAgentBoundary) -> SubCodeAgent:
        """Create a sub-agent from a boundary definition.

        Args:
            boundary: SubAgentBoundary defining the sub-agent's domain.

        Returns:
            Configured SubCodeAgent instance.
        """
        spec = AgentSpec(
            agent_id=boundary.sub_agent_id,
            name=self._generate_name(boundary),
            description=boundary.role_description or self._generate_description(boundary),
            context_paths=boundary.include_patterns,
            context_keywords=self._extract_keywords(boundary),
            estimated_tokens=int(boundary.estimated_context_kb * 400),
            max_context_size=50000,  # 50KB default for sub-agents
            system_prompt=self._generate_system_prompt(boundary),
            metadata={
                "auto_generated": True,
                "tier": "B",
                "sub_agent": True,
                "root_path": boundary.root_path,
                "generated_at": datetime.now().isoformat(),
            },
            # Hierarchy fields
            parent_agent_id=boundary.parent_agent_id,
            children_ids=[],
            hierarchy_level=1,  # Sub-agents are level 1
            is_team_lead=False,
        )

        return SubCodeAgent(
            spec=spec,
            client=self._auto_manager.hub.client,
            root_path=boundary.root_path,
            include_patterns=boundary.include_patterns,
            parent_agent_id=boundary.parent_agent_id,
        )

    def _generate_name(self, boundary: SubAgentBoundary) -> str:
        """Generate human-readable name for sub-agent."""
        # Extract meaningful name from sub_agent_id
        # e.g., "backend_api" -> "Backend API Team Member"
        parts = boundary.sub_agent_id.split("_")
        if len(parts) > 1:
            # Skip parent prefix, use the rest
            name_part = " ".join(p.title() for p in parts[1:])
        else:
            name_part = parts[0].title()

        return f"{name_part} Team Member"

    def _generate_description(self, boundary: SubAgentBoundary) -> str:
        """Generate description for sub-agent."""
        return (
            f"Specialized team member handling {boundary.root_path} "
            f"({boundary.file_count} files, {boundary.estimated_context_kb:.0f}KB)"
        )

    def _extract_keywords(self, boundary: SubAgentBoundary) -> list[str]:
        """Extract routing keywords for sub-agent."""
        keywords: set[str] = set()

        # Add ID parts as keywords
        for part in boundary.sub_agent_id.split("_"):
            if part and len(part) > 2:
                keywords.add(part.lower())

        # Add root path parts
        for part in Path(boundary.root_path).parts:
            if part and not part.startswith(".") and len(part) > 2:
                keywords.add(part.lower())

        # Add key module names
        for module in boundary.key_modules:
            stem = Path(module).stem
            if stem and stem not in ("__init__", "index"):
                keywords.add(stem.lower())

        return list(keywords)[:15]

    def _generate_system_prompt(self, boundary: SubAgentBoundary) -> str:
        """Generate system prompt for sub-agent."""
        interfaces = ", ".join(boundary.interfaces_with) if boundary.interfaces_with else "none"

        return f"""You are a specialized team member focusing on the {boundary.root_path} sub-domain.

You are part of a team led by {boundary.parent_agent_id}.

Your focused area:
- Root path: {boundary.root_path}
- {boundary.file_count} files in your domain
- Key modules: {', '.join(boundary.key_modules) or 'see context below'}
- Interfaces with: {interfaces}

When answering:
- Reference specific files and functions in YOUR sub-domain
- Be precise — you own a focused subset of the codebase
- If asked about code outside your sub-domain, acknowledge the boundary
  and suggest asking your team lead ({boundary.parent_agent_id}) instead

Your scope is LIMITED to {boundary.root_path}. Stay focused on your specialty."""

    # === Query and Routing Methods ===

    def get_team(self, parent_agent_id: str) -> list[SubCodeAgent]:
        """Get all sub-agents for a team lead.

        Args:
            parent_agent_id: ID of the parent/team lead agent.

        Returns:
            List of SubCodeAgent instances on this team.
        """
        team = self._sub_agents.get(parent_agent_id, {})
        return list(team.values())

    def get_team_lead(self, sub_agent_id: str) -> Optional[BaseAgent]:
        """Get the team lead for a sub-agent.

        Args:
            sub_agent_id: ID of the sub-agent.

        Returns:
            Parent BaseAgent or None if not found.
        """
        boundary = self._boundaries.get(sub_agent_id)
        if boundary is None:
            return None

        return self._auto_manager._auto_agents.get(boundary.parent_agent_id)

    def route_to_sub_agent(
        self, parent_agent_id: str, file_path: str
    ) -> Optional[SubCodeAgent]:
        """Route a file to the most specific owning sub-agent.

        Args:
            parent_agent_id: ID of the parent agent to check.
            file_path: File path to route.

        Returns:
            Most specific SubCodeAgent owning this file, or None.
        """
        team = self._sub_agents.get(parent_agent_id, {})

        for sub_agent in team.values():
            if self._file_in_sub_agent_scope(file_path, sub_agent):
                return sub_agent

        return None

    def _file_in_sub_agent_scope(
        self, file_path: str, sub_agent: SubCodeAgent
    ) -> bool:
        """Check if a file falls within a sub-agent's scope."""
        import fnmatch

        file_path = file_path.replace("\\", "/")

        # Check against include patterns
        for pattern in sub_agent.include_patterns:
            pattern = pattern.replace("\\", "/")
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # Handle directory prefix
            if pattern.endswith("/**/*"):
                dir_prefix = pattern[:-5]
                if file_path.startswith(dir_prefix):
                    return True
            elif pattern.endswith("**"):
                dir_prefix = pattern[:-2].rstrip("/")
                if file_path.startswith(dir_prefix):
                    return True

        # Check against root_path
        if file_path.startswith(sub_agent.root_path.rstrip("/") + "/"):
            return True

        return False

    def team_query(
        self,
        parent_agent_id: str,
        query: str,
        session: Session,
        delegate: bool = True,
    ) -> AgentResponse:
        """Query the team, optionally delegating to relevant sub-agents.

        Args:
            parent_agent_id: ID of the team lead.
            query: User query.
            session: Current session.
            delegate: If True, route to relevant sub-agent(s).
                      If False, only team lead answers.

        Returns:
            AgentResponse from the appropriate agent(s).
        """
        # Get team lead
        team_lead = self._auto_manager._auto_agents.get(parent_agent_id)
        if team_lead is None:
            raise ValueError(f"Team lead {parent_agent_id} not found")

        if not delegate:
            # Team lead answers directly
            return team_lead.run(query, session)

        # Try to identify relevant sub-agent(s) from query
        team = self.get_team(parent_agent_id)

        if not team:
            # No sub-agents, team lead handles it
            return team_lead.run(query, session)

        # Simple keyword routing to find most relevant sub-agent
        best_match: Optional[SubCodeAgent] = None
        best_score = 0

        query_lower = query.lower()
        for sub_agent in team:
            score = 0
            for keyword in sub_agent.spec.context_keywords:
                if keyword in query_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_match = sub_agent

        if best_match is not None and best_score > 0:
            # Delegate to sub-agent
            return best_match.run(query, session)

        # No clear match, team lead handles it
        return team_lead.run(query, session)

    def get_most_specific_agent(self, file_path: str) -> Optional[BaseAgent]:
        """Get the most specific agent owning a file.

        Checks sub-agents first, then falls back to team leads.

        Args:
            file_path: Path to the file.

        Returns:
            Most specific agent owning this file.
        """
        # Check all sub-agents first
        for parent_id, team in self._sub_agents.items():
            for sub_agent in team.values():
                if self._file_in_sub_agent_scope(file_path, sub_agent):
                    return sub_agent

        # No sub-agent match, caller should check Tier B agents
        return None

    # === Cleanup Methods ===

    def unregister_all_sub_agents(self) -> int:
        """Unregister all sub-agents.

        Returns:
            Number of sub-agents unregistered.
        """
        count = 0

        for parent_id, team in list(self._sub_agents.items()):
            for sub_agent_id in list(team.keys()):
                try:
                    self._auto_manager.hub.unregister(sub_agent_id)
                    count += 1
                except Exception:
                    pass

            # Clear parent's hierarchy
            parent = self._auto_manager._auto_agents.get(parent_id)
            if parent:
                parent.spec.children_ids = []
                parent.spec.is_team_lead = False

        self._sub_agents.clear()
        self._boundaries.clear()
        self._parent_to_children.clear()

        return count

    def get_hierarchy_report(self) -> dict:
        """Get a report of the current agent hierarchy.

        Returns:
            Dict with hierarchy statistics.
        """
        total_sub_agents = sum(len(team) for team in self._sub_agents.values())
        team_leads = list(self._sub_agents.keys())

        return {
            "team_leads": team_leads,
            "total_team_leads": len(team_leads),
            "total_sub_agents": total_sub_agents,
            "teams": {
                parent_id: [sa.spec.agent_id for sa in team.values()]
                for parent_id, team in self._sub_agents.items()
            },
        }
