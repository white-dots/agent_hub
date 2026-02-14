from __future__ import annotations
"""Policy for determining when and how to subdivide Tier B agents.

This module defines the rules for splitting large Tier B agents into
focused sub-agents (team members) when their domain becomes too large
for a single agent to handle effectively.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent
    from agenthub.auto.import_graph import ImportGraph

from agenthub.models import SubAgentBoundary


@dataclass
class SubAgentPolicy:
    """Determines when a Tier B agent should be subdivided into sub-agents.

    The policy uses two types of thresholds:
    1. Triggering thresholds: When to consider subdivision
    2. Stop conditions: When to stop subdividing further

    Example:
        >>> policy = SubAgentPolicy(min_files_to_split=60)
        >>> if policy.should_subdivide(agent, graph):
        ...     boundaries = policy.propose_subdivisions(agent, graph)
        ...     for boundary in boundaries:
        ...         print(f"Create sub-agent: {boundary.sub_agent_id}")
    """

    # === Thresholds for triggering subdivision ===
    # Only subdivide when BOTH conditions are met
    min_files_to_split: int = 60
    """Agent must cover at least this many files to be considered for splitting."""

    min_subdirs_to_split: int = 3
    """Agent's context_paths must span at least this many distinct subdirectories."""

    # === Stop conditions ===
    min_files_per_sub: int = 10
    """Each sub-agent must have at least this many files (prevents over-splitting)."""

    max_sub_agents: int = 6
    """Maximum number of sub-agents to create from a single Tier B agent."""

    # === Context efficiency ===
    context_utilization_threshold: float = 0.8
    """Only split if agent's context exceeds this fraction of max_context_size."""

    def should_subdivide(
        self, agent: "BaseAgent", graph: "ImportGraph"
    ) -> bool:
        """Check if a Tier B agent warrants subdivision into sub-agents.

        Conditions (ALL must be true):
        1. Agent covers more than min_files_to_split files
        2. Agent's context_paths span 3+ distinct subdirectories
        3. Import graph shows distinct sub-clusters within domain
        4. Agent's estimated context exceeds utilization threshold

        Args:
            agent: The Tier B agent to evaluate.
            graph: The import graph for the codebase.

        Returns:
            True if the agent should be subdivided.
        """
        spec = agent.spec

        # Condition 1: Check file count
        files_in_scope = self._count_files_in_scope(spec.context_paths, graph)
        if files_in_scope < self.min_files_to_split:
            return False

        # Condition 2: Check subdirectory count
        subdirs = self._count_distinct_subdirs(spec.context_paths, graph)
        if subdirs < self.min_subdirs_to_split:
            return False

        # Condition 3: Check if import graph shows distinct sub-clusters
        sub_clusters = self._find_sub_clusters(spec.context_paths, graph)
        if len(sub_clusters) < 2:
            return False

        # Condition 4: Check context utilization
        estimated_context = self._estimate_context_size(spec.context_paths, graph)
        if estimated_context < spec.max_context_size * self.context_utilization_threshold:
            return False

        return True

    def propose_subdivisions(
        self, agent: "BaseAgent", graph: "ImportGraph"
    ) -> list[SubAgentBoundary]:
        """Propose how to split a Tier B agent into sub-agents.

        Strategy:
        1. Get all files in agent's context_paths
        2. Build sub-graph restricted to these files
        3. Cluster sub-graph (reuse ImportGraph clustering)
        4. Each sub-cluster → SubAgentBoundary
        5. Identify inter-sub-agent interfaces
        6. Generate role descriptions

        Falls back to directory-based splitting if import graph is too sparse.

        Args:
            agent: The Tier B agent to subdivide.
            graph: The import graph for the codebase.

        Returns:
            List of SubAgentBoundary proposals for sub-agents.
        """
        spec = agent.spec
        parent_id = spec.agent_id

        # Get all files in agent's scope
        files_in_scope = self._get_files_in_scope(spec.context_paths, graph)

        # Try import-graph-based clustering first
        sub_clusters = self._find_sub_clusters(spec.context_paths, graph)

        # Check how many clusters would survive the size filter
        viable_clusters = [c for c in sub_clusters if len(c) >= self.min_files_per_sub]

        # Fall back to directory-based if import graph doesn't give enough viable clusters
        if len(viable_clusters) < 2:
            sub_clusters = self._split_by_directory(files_in_scope)
        else:
            sub_clusters = viable_clusters

        # Limit to max_sub_agents
        if len(sub_clusters) > self.max_sub_agents:
            sub_clusters = self._merge_smallest_clusters(
                sub_clusters, self.max_sub_agents
            )

        # Filter out clusters that are too small (needed for directory-based fallback)
        sub_clusters = [
            c for c in sub_clusters if len(c) >= self.min_files_per_sub
        ]

        # Build SubAgentBoundary for each cluster
        boundaries: list[SubAgentBoundary] = []
        cluster_to_id: dict[int, str] = {}  # For tracking interfaces

        for i, cluster in enumerate(sub_clusters):
            # Determine root path for this cluster
            root_path = self._find_common_root(cluster)

            # Generate sub-agent ID
            sub_agent_id = self._generate_sub_agent_id(parent_id, root_path, i)
            cluster_to_id[i] = sub_agent_id

            # Generate include patterns
            include_patterns = self._generate_include_patterns(cluster, root_path)

            # Find key modules (central files in this cluster)
            key_modules = self._find_key_modules(cluster, graph)

            # Calculate estimated context size
            estimated_kb = self._calculate_cluster_size_kb(cluster, graph)

            # Generate role description
            role_description = self._generate_role_description(
                cluster, root_path, graph
            )

            boundary = SubAgentBoundary(
                parent_agent_id=parent_id,
                sub_agent_id=sub_agent_id,
                root_path=root_path,
                include_patterns=include_patterns,
                estimated_context_kb=estimated_kb,
                file_count=len(cluster),
                role_description=role_description,
                key_modules=key_modules,
                interfaces_with=[],  # Filled in below
            )
            boundaries.append(boundary)

        # Identify inter-sub-agent interfaces
        self._identify_interfaces(boundaries, sub_clusters, graph, cluster_to_id)

        return boundaries

    # === Helper methods ===

    def _count_files_in_scope(
        self, context_paths: list[str], graph: "ImportGraph"
    ) -> int:
        """Count files in the agent's context_paths that exist in the graph."""
        count = 0
        for module_path in graph.nodes:
            if self._path_matches_context(module_path, context_paths):
                count += 1
        return count

    def _get_files_in_scope(
        self, context_paths: list[str], graph: "ImportGraph"
    ) -> set[str]:
        """Get all files in the agent's context_paths."""
        files: set[str] = set()
        for module_path in graph.nodes:
            if self._path_matches_context(module_path, context_paths):
                files.add(module_path)
        return files

    def _path_matches_context(
        self, module_path: str, context_paths: list[str]
    ) -> bool:
        """Check if a module path falls under any of the context paths."""
        for ctx_path in context_paths:
            # Handle glob patterns
            if "**" in ctx_path or "*" in ctx_path:
                import fnmatch
                if fnmatch.fnmatch(module_path, ctx_path):
                    return True
            else:
                # Direct path prefix match
                if module_path.startswith(ctx_path.rstrip("/")):
                    return True
        return False

    def _count_distinct_subdirs(
        self, context_paths: list[str], graph: "ImportGraph"
    ) -> int:
        """Count distinct top-level subdirectories in context.

        Finds the common root of all files and counts distinct
        subdirectories at the first level below that root.
        """
        files = self._get_files_in_scope(context_paths, graph)
        if not files:
            return 0

        # Find common root of all files
        paths = [Path(f).parts for f in files]
        common_parts = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common_parts.append(parts[0])
            else:
                break

        # Count distinct subdirectories relative to common root
        subdirs: set[str] = set()
        common_len = len(common_parts)

        for file_path in files:
            parts = Path(file_path).parts
            # Get the directory at the first level below common root
            if len(parts) > common_len + 1:
                # There's at least one dir between common root and file
                subdirs.add(parts[common_len])
            elif len(parts) == common_len + 1:
                # File is directly in common root, count as "root"
                subdirs.add("__root__")

        return len(subdirs)

    def _find_sub_clusters(
        self, context_paths: list[str], graph: "ImportGraph"
    ) -> list[set[str]]:
        """Find clusters within the agent's domain using import graph."""
        files_in_scope = self._get_files_in_scope(context_paths, graph)

        if not files_in_scope:
            return []

        # Build sub-graph adjacency for files in scope
        adjacency: dict[str, set[str]] = defaultdict(set)

        for edge in graph.edges:
            if edge.source in files_in_scope and edge.target in files_in_scope:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

        # Find connected components
        visited: set[str] = set()
        clusters: list[set[str]] = []

        for node in files_in_scope:
            if node in visited:
                continue

            # BFS to find connected component
            cluster: set[str] = set()
            queue = [node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if cluster:
                clusters.append(cluster)

        # Add isolated files as single-file clusters (will be filtered later)
        for file in files_in_scope:
            if file not in visited:
                clusters.append({file})

        return clusters

    def _split_by_directory(self, files: set[str]) -> list[set[str]]:
        """Split files by their parent directory relative to common root.

        Finds the common root first, then groups by the first level below it.
        This ensures proper splitting for deep directory structures like
        backend/app/api vs backend/app/models vs backend/app/services.
        """
        if not files:
            return []

        # Find common root of all files
        paths = [Path(f).parts for f in files]
        common_parts = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common_parts.append(parts[0])
            else:
                break

        common_len = len(common_parts)
        dir_groups: dict[str, set[str]] = defaultdict(set)

        for file_path in files:
            parts = Path(file_path).parts
            if len(parts) > common_len + 1:
                # Group by directory immediately below common root
                dir_key = "/".join(common_parts + [parts[common_len]])
            elif len(parts) > common_len:
                # File is directly in common root
                dir_key = "/".join(common_parts) if common_parts else "root"
            else:
                dir_key = "root"

            dir_groups[dir_key].add(file_path)

        return list(dir_groups.values())

    def _merge_smallest_clusters(
        self, clusters: list[set[str]], max_clusters: int
    ) -> list[set[str]]:
        """Merge smallest clusters until we have at most max_clusters."""
        # Sort by size (ascending)
        sorted_clusters = sorted(clusters, key=len)

        while len(sorted_clusters) > max_clusters:
            # Merge the two smallest
            smallest = sorted_clusters.pop(0)
            second_smallest = sorted_clusters.pop(0)
            merged = smallest | second_smallest
            sorted_clusters.append(merged)
            sorted_clusters.sort(key=len)

        return sorted_clusters

    def _find_common_root(self, cluster: set[str]) -> str:
        """Find the common root path for a cluster of files."""
        if not cluster:
            return ""

        paths = [Path(f).parts for f in cluster]

        # Find common prefix
        common = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common.append(parts[0])
            else:
                break

        if common:
            return "/".join(common)
        return ""

    def _generate_sub_agent_id(
        self, parent_id: str, root_path: str, index: int
    ) -> str:
        """Generate a unique ID for a sub-agent."""
        if root_path:
            # Use the last meaningful directory name
            parts = Path(root_path).parts
            suffix = parts[-1] if parts else f"sub_{index}"
            return f"{parent_id}_{suffix}"
        return f"{parent_id}_sub_{index}"

    def _generate_include_patterns(
        self, cluster: set[str], root_path: str
    ) -> list[str]:
        """Generate glob patterns that match the files in this cluster."""
        if not root_path:
            return [f for f in cluster]

        # Group by extension
        extensions: set[str] = set()
        for file_path in cluster:
            ext = Path(file_path).suffix
            if ext:
                extensions.add(ext)

        patterns = []
        for ext in extensions:
            patterns.append(f"{root_path}/**/*{ext}")

        return patterns if patterns else [f"{root_path}/**/*"]

    def _find_key_modules(
        self, cluster: set[str], graph: "ImportGraph"
    ) -> list[str]:
        """Find the most central modules in a cluster."""
        # Calculate in-degree within cluster
        in_degree: dict[str, int] = defaultdict(int)

        for file_path in cluster:
            if file_path in graph.nodes:
                node = graph.nodes[file_path]
                # Count imports from within this cluster
                for imported_by in node.imported_by:
                    if imported_by in cluster:
                        in_degree[file_path] += 1

        # Sort by in-degree and return top 3
        sorted_modules = sorted(
            in_degree.items(), key=lambda x: x[1], reverse=True
        )
        return [path for path, _ in sorted_modules[:3]]

    def _calculate_cluster_size_kb(
        self, cluster: set[str], graph: "ImportGraph"
    ) -> float:
        """Calculate total size of files in cluster in KB."""
        total_bytes = 0
        for file_path in cluster:
            if file_path in graph.nodes:
                total_bytes += graph.nodes[file_path].size_bytes
        return total_bytes / 1024

    def _generate_role_description(
        self, cluster: set[str], root_path: str, graph: "ImportGraph"
    ) -> str:
        """Generate a description of what this sub-agent handles."""
        # Collect function and class names from cluster
        functions: list[str] = []
        classes: list[str] = []

        for file_path in cluster:
            if file_path in graph.nodes:
                node = graph.nodes[file_path]
                functions.extend(node.functions[:5])  # Limit to avoid verbosity
                classes.extend(node.classes[:5])

        # Build description
        parts = []
        if root_path:
            parts.append(f"Handles {root_path}/ subdomain")

        if classes:
            parts.append(f"Key classes: {', '.join(classes[:5])}")
        if functions:
            parts.append(f"Key functions: {', '.join(functions[:5])}")

        if not parts:
            parts.append(f"Manages {len(cluster)} files")

        return ". ".join(parts)

    def _identify_interfaces(
        self,
        boundaries: list[SubAgentBoundary],
        clusters: list[set[str]],
        graph: "ImportGraph",
        cluster_to_id: dict[int, str],
    ) -> None:
        """Identify which sub-agents interface with which others."""
        # Build reverse mapping: file -> cluster index
        file_to_cluster: dict[str, int] = {}
        for i, cluster in enumerate(clusters):
            for file_path in cluster:
                file_to_cluster[file_path] = i

        # For each cluster, find which other clusters it imports from
        for i, cluster in enumerate(clusters):
            interfaces: set[str] = set()

            for file_path in cluster:
                if file_path in graph.nodes:
                    node = graph.nodes[file_path]
                    # Check what this file imports
                    for imported in node.imports:
                        if imported in file_to_cluster:
                            other_cluster = file_to_cluster[imported]
                            if other_cluster != i:
                                other_id = cluster_to_id.get(other_cluster)
                                if other_id:
                                    interfaces.add(other_id)

            if i < len(boundaries):
                boundaries[i].interfaces_with = list(interfaces)

    def _estimate_context_size(
        self, context_paths: list[str], graph: "ImportGraph"
    ) -> int:
        """Estimate total context size in bytes for an agent's scope."""
        files = self._get_files_in_scope(context_paths, graph)
        total = 0
        for file_path in files:
            if file_path in graph.nodes:
                total += graph.nodes[file_path].size_bytes
        return total
