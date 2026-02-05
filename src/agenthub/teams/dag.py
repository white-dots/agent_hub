"""DAG data structures for team execution.

This module provides the core data structures for representing and executing
multi-agent queries in topological order based on agent dependencies.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthub.auto.import_graph import ImportGraph
    from agenthub.hub import AgentHub


@dataclass
class DAGNode:
    """Represents an agent node in the execution DAG."""

    agent_id: str
    dependencies: list[str] = field(default_factory=list)  # Agent IDs this node depends on
    sub_question: str = ""  # Filled by decomposer
    result: str = ""  # Filled after execution
    status: str = "pending"  # "pending" | "running" | "done" | "failed"
    error: str = ""  # Error message if failed
    tokens_used: int = 0
    execution_time_ms: int = 0


@dataclass
class ExecutionDAG:
    """Represents the execution order for a multi-agent query.

    The DAG captures which agents need to run and in what order,
    based on their dependencies derived from the import graph.
    """

    nodes: dict[str, DAGNode] = field(default_factory=dict)  # agent_id -> DAGNode
    _layers: list[list[str]] | None = field(default=None, repr=False)

    def add_node(self, agent_id: str, dependencies: list[str] | None = None) -> DAGNode:
        """Add a node to the DAG.

        Args:
            agent_id: The agent's ID.
            dependencies: List of agent IDs this agent depends on.

        Returns:
            The created DAGNode.
        """
        node = DAGNode(agent_id=agent_id, dependencies=dependencies or [])
        self.nodes[agent_id] = node
        self._layers = None  # Invalidate cached layers
        return node

    def add_edge(self, from_agent: str, to_agent: str) -> None:
        """Add a dependency edge (from_agent depends on to_agent).

        Args:
            from_agent: Agent that has the dependency.
            to_agent: Agent that is depended upon.
        """
        if from_agent in self.nodes and to_agent not in self.nodes[from_agent].dependencies:
            self.nodes[from_agent].dependencies.append(to_agent)
            self._layers = None  # Invalidate cached layers

    def get_ready_nodes(self) -> list[str]:
        """Get nodes whose dependencies are all 'done'.

        Returns:
            List of agent IDs ready to execute.
        """
        ready = []
        for agent_id, node in self.nodes.items():
            if node.status != "pending":
                continue
            # Check if all dependencies are done
            all_deps_done = all(
                self.nodes.get(dep, DAGNode(agent_id=dep)).status == "done"
                for dep in node.dependencies
            )
            if all_deps_done:
                ready.append(agent_id)
        return ready

    def topological_layers(self) -> list[list[str]]:
        """Return agents grouped by execution layer.

        Layer 0: agents with no dependencies (run first, in parallel)
        Layer 1: agents that depend only on layer 0 (run next, in parallel)
        ...

        Uses Kahn's algorithm for topological sorting.

        Returns:
            List of layers, each layer is a list of agent IDs.
        """
        if self._layers is not None:
            return self._layers

        if not self.nodes:
            self._layers = []
            return self._layers

        # Build in-degree map (only counting dependencies within our node set)
        in_degree: dict[str, int] = {}
        for agent_id in self.nodes:
            in_degree[agent_id] = 0

        for agent_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    in_degree[agent_id] = in_degree.get(agent_id, 0) + 1

        # Start with nodes that have no dependencies
        layers: list[list[str]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            # Find all nodes with in-degree 0
            layer = [
                agent_id
                for agent_id in remaining
                if in_degree.get(agent_id, 0) == 0
            ]

            if not layer:
                # Cycle detected - break it by picking the node with lowest in-degree
                min_degree = min(in_degree.get(a, 0) for a in remaining)
                layer = [
                    a for a in remaining if in_degree.get(a, 0) == min_degree
                ][:1]

            layers.append(layer)
            remaining -= set(layer)

            # Decrease in-degree for nodes that depend on this layer
            for agent_id in layer:
                for other_id, other_node in self.nodes.items():
                    if agent_id in other_node.dependencies and other_id in remaining:
                        in_degree[other_id] = in_degree.get(other_id, 1) - 1

        self._layers = layers
        return self._layers

    def mark_running(self, agent_id: str) -> None:
        """Mark a node as running."""
        if agent_id in self.nodes:
            self.nodes[agent_id].status = "running"

    def mark_done(
        self,
        agent_id: str,
        result: str,
        tokens: int = 0,
        time_ms: int = 0,
    ) -> None:
        """Mark a node as done with its result."""
        if agent_id in self.nodes:
            node = self.nodes[agent_id]
            node.status = "done"
            node.result = result
            node.tokens_used = tokens
            node.execution_time_ms = time_ms

    def mark_failed(self, agent_id: str, error: str) -> None:
        """Mark a node as failed with an error message."""
        if agent_id in self.nodes:
            node = self.nodes[agent_id]
            node.status = "failed"
            node.error = error

    def is_complete(self) -> bool:
        """Check if all nodes are done or failed."""
        return all(
            node.status in ("done", "failed")
            for node in self.nodes.values()
        )

    def get_results(self) -> dict[str, str]:
        """Get all results from completed nodes."""
        return {
            agent_id: node.result
            for agent_id, node in self.nodes.items()
            if node.status == "done"
        }

    def get_dependency_results(self, agent_id: str) -> dict[str, str]:
        """Get results from an agent's dependencies.

        Args:
            agent_id: The agent to get dependency results for.

        Returns:
            Dict mapping dependency agent_id to their results.
        """
        if agent_id not in self.nodes:
            return {}

        node = self.nodes[agent_id]
        results = {}
        for dep_id in node.dependencies:
            if dep_id in self.nodes and self.nodes[dep_id].status == "done":
                results[dep_id] = self.nodes[dep_id].result
        return results

    @classmethod
    def from_import_graph(
        cls,
        matched_agents: list[str],
        import_graph: "ImportGraph",
        hub: "AgentHub",
    ) -> "ExecutionDAG":
        """Build an ExecutionDAG from the import graph and matched agents.

        Determines dependencies between agents by checking if modules
        owned by one agent import modules owned by another agent.

        Args:
            matched_agents: List of agent IDs that matched the query.
            import_graph: The project's ImportGraph.
            hub: AgentHub instance for accessing agent specs.

        Returns:
            ExecutionDAG with nodes and dependency edges.
        """
        dag = cls()

        # First, add all matched agents as nodes
        for agent_id in matched_agents:
            dag.add_node(agent_id)

        # Build a mapping of file paths to agent IDs
        path_to_agent: dict[str, str] = {}
        for agent_id in matched_agents:
            agent = hub.get_agent(agent_id)
            if agent and agent.spec.context_paths:
                for path in agent.spec.context_paths:
                    # Normalize the path
                    normalized = path.lstrip("./").rstrip("/")
                    path_to_agent[normalized] = agent_id

        # Determine edges based on import relationships
        for agent_a in matched_agents:
            agent_a_obj = hub.get_agent(agent_a)
            if not agent_a_obj:
                continue

            # Get modules owned by agent_a
            modules_a = set()
            for path in agent_a_obj.spec.context_paths:
                normalized = path.lstrip("./").rstrip("/")
                modules_a.add(normalized)
                # Also add files under this path if it's a directory
                for node_path in import_graph.nodes:
                    if node_path.startswith(normalized):
                        modules_a.add(node_path)

            # Check what agent_a's modules import
            for mod_a in modules_a:
                if mod_a not in import_graph.nodes:
                    continue

                node = import_graph.nodes[mod_a]
                for imported_path in node.imports:
                    # Check if this import belongs to another matched agent
                    for agent_b in matched_agents:
                        if agent_b == agent_a:
                            continue

                        agent_b_obj = hub.get_agent(agent_b)
                        if not agent_b_obj:
                            continue

                        # Check if the imported path belongs to agent_b
                        for path_b in agent_b_obj.spec.context_paths:
                            normalized_b = path_b.lstrip("./").rstrip("/")
                            # Check if imported_path starts with agent_b's path
                            # or if there's a resolved edge to agent_b's modules
                            if imported_path.startswith(normalized_b) or imported_path == normalized_b:
                                # agent_a imports from agent_b, so agent_a depends on agent_b
                                dag.add_edge(agent_a, agent_b)
                                break

                # Also check resolved edges
                for edge in import_graph.edges:
                    if edge.source != mod_a:
                        continue
                    target_path = edge.target
                    # Find which agent owns the target
                    for agent_b in matched_agents:
                        if agent_b == agent_a:
                            continue
                        agent_b_obj = hub.get_agent(agent_b)
                        if not agent_b_obj:
                            continue
                        for path_b in agent_b_obj.spec.context_paths:
                            normalized_b = path_b.lstrip("./").rstrip("/")
                            if target_path.startswith(normalized_b) or target_path == normalized_b:
                                dag.add_edge(agent_a, agent_b)
                                break

        return dag

    def to_dict(self) -> dict:
        """Convert DAG to a dictionary for serialization."""
        return {
            "nodes": {
                agent_id: {
                    "agent_id": node.agent_id,
                    "dependencies": node.dependencies,
                    "sub_question": node.sub_question,
                    "status": node.status,
                    "tokens_used": node.tokens_used,
                    "execution_time_ms": node.execution_time_ms,
                }
                for agent_id, node in self.nodes.items()
            },
            "execution_layers": self.topological_layers(),
        }
