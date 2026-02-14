from __future__ import annotations
"""Domain analysis for dynamic Tier B agent generation.

This module identifies natural domains in a codebase by combining:
1. Import graph clusters
2. Folder structure hints
3. Tier A agent context (business keywords)
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agenthub.auto.import_graph import ImportGraph

if TYPE_CHECKING:
    from agenthub.models import AgentSpec


@dataclass
class Domain:
    """Represents a discovered domain in the codebase."""

    name: str  # Human-readable domain name
    agent_id: str  # Generated agent ID (e.g., "auth_agent")
    modules: list[str]  # File paths in this domain
    central_module: str  # Most connected/important file
    keywords: list[str]  # Keywords extracted from code
    related_tier_a: list[str] = field(default_factory=list)  # Related Tier A agent IDs
    folder_prefix: str = ""  # Common folder prefix if any
    description: str = ""  # Auto-generated description
    total_size_bytes: int = 0  # Total size of all modules in bytes
    primary_language: str = "python"  # Primary language ("python", "typescript", "javascript")


class DomainAnalysis:
    """Identifies natural domains from multiple signals.

    This class combines import graph analysis, folder structure, and
    Tier A agent context to discover domains without relying on
    hardcoded patterns like "api", "service", "model".

    Example:
        >>> graph = ImportGraph("./project")
        >>> graph.build()
        >>> analysis = DomainAnalysis(graph, tier_a_agents=[])
        >>> domains = analysis.analyze()
        >>> for domain in domains:
        ...     print(f"{domain.name}: {len(domain.modules)} modules")
    """

    def __init__(
        self,
        import_graph: ImportGraph,
        tier_a_agents: list["AgentSpec"] | None = None,
        min_cluster_size: int = 2,
        max_domains: int = 10,
    ):
        """Initialize domain analysis.

        Args:
            import_graph: Built ImportGraph instance.
            tier_a_agents: List of Tier A agent specs for context.
            min_cluster_size: Minimum modules to form a domain.
            max_domains: Maximum number of domains to create.
        """
        self.graph = import_graph
        self.tier_a_agents = tier_a_agents or []
        self.min_cluster_size = min_cluster_size
        self.max_domains = max_domains

        # Extract Tier A keywords for matching
        self.tier_a_keywords: dict[str, set[str]] = {}
        for agent in self.tier_a_agents:
            keywords = set(kw.lower() for kw in (agent.context_keywords or []))
            # Also add words from agent name and description
            keywords.update(
                word.lower()
                for word in re.findall(r"\w+", agent.name + " " + agent.description)
                if len(word) > 2
            )
            self.tier_a_keywords[agent.agent_id] = keywords

    def analyze(self) -> list[Domain]:
        """Analyze the codebase and identify domains.

        Returns:
            List of Domain objects representing discovered domains.
        """
        # Get import-based clusters
        raw_clusters = self.graph.get_clusters(min_connections=1)

        # Refine clusters with folder structure
        refined_clusters = self._refine_with_folders(raw_clusters)

        # Merge small clusters
        merged_clusters = self._merge_small_clusters(refined_clusters)

        # Create domains from clusters
        domains = []
        for cluster in merged_clusters[: self.max_domains]:
            if len(cluster) >= self.min_cluster_size or len(merged_clusters) <= 3:
                domain = self._create_domain(cluster)
                if domain:
                    domains.append(domain)

        # Map to Tier A agents
        self._map_to_tier_a(domains)

        return domains

    def _refine_with_folders(self, clusters: list[set[str]]) -> list[set[str]]:
        """Refine clusters by considering folder structure.

        Modules in the same folder that aren't strongly connected
        to other clusters might belong together.
        """
        # Group all modules by their parent folder
        folder_groups: dict[str, set[str]] = defaultdict(set)
        for cluster in clusters:
            for module in cluster:
                folder = str(Path(module).parent)
                folder_groups[folder].add(module)

        # Check if folder groups are split across clusters
        refined: list[set[str]] = []

        for cluster in clusters:
            # Find the dominant folder(s) in this cluster
            folder_counts: Counter[str] = Counter()
            for module in cluster:
                folder = str(Path(module).parent)
                folder_counts[folder] += 1

            # If cluster is dominated by one folder, check for stragglers
            if folder_counts:
                dominant_folder, dominant_count = folder_counts.most_common(1)[0]
                ratio = dominant_count / len(cluster)

                # If >60% from one folder, consider adding related modules
                if ratio > 0.6:
                    folder_modules = folder_groups[dominant_folder]
                    # Add modules from same folder that aren't strongly connected elsewhere
                    enhanced_cluster = set(cluster)
                    for module in folder_modules:
                        # Check if this module is strongly connected to another cluster
                        other_cluster_size = 0
                        for other_cluster in clusters:
                            if module in other_cluster and other_cluster != cluster:
                                other_cluster_size = len(other_cluster)
                                break

                        # If not strongly connected elsewhere, add to this cluster
                        if other_cluster_size < len(cluster):
                            enhanced_cluster.add(module)

                    refined.append(enhanced_cluster)
                else:
                    refined.append(cluster)
            else:
                refined.append(cluster)

        return refined

    def _merge_small_clusters(self, clusters: list[set[str]]) -> list[set[str]]:
        """Merge very small clusters with related larger ones."""
        if not clusters:
            return []

        # Separate small and large clusters
        large_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]
        small_clusters = [c for c in clusters if len(c) < self.min_cluster_size]

        if not large_clusters:
            # All clusters are small, just return as-is
            return clusters

        # Try to merge each small cluster with a related large one
        for small in small_clusters:
            best_match = None
            best_score = 0

            for large in large_clusters:
                # Calculate connection score
                score = self._cluster_similarity(small, large)
                if score > best_score:
                    best_score = score
                    best_match = large

            if best_match and best_score > 0:
                best_match.update(small)
            else:
                # Can't merge, keep as separate
                large_clusters.append(small)

        return large_clusters

    def _cluster_similarity(self, cluster_a: set[str], cluster_b: set[str]) -> float:
        """Calculate similarity between two clusters based on imports and folders."""
        score = 0.0

        # Check import connections
        for module_a in cluster_a:
            neighbors = self.graph.get_module_neighbors(module_a)
            for module_b in cluster_b:
                if module_b in neighbors["imports"]:
                    score += 1.0
                if module_b in neighbors["imported_by"]:
                    score += 1.0

        # Check folder similarity
        folders_a = {str(Path(m).parent) for m in cluster_a}
        folders_b = {str(Path(m).parent) for m in cluster_b}
        common_folders = folders_a & folders_b
        if common_folders:
            score += len(common_folders) * 0.5

        return score

    def _create_domain(self, modules: set[str]) -> Domain | None:
        """Create a Domain object from a cluster of modules."""
        if not modules:
            return None

        module_list = sorted(modules)

        # Find central module
        central = self._find_central_module(module_list)

        # Extract keywords from all modules
        keywords = self._extract_keywords(module_list)

        # Generate domain name
        name = self._generate_domain_name(module_list, keywords)

        # Generate agent ID
        agent_id = self._generate_agent_id(name)

        # Find common folder prefix
        folder_prefix = self._find_common_prefix(module_list)

        # Generate description
        description = self._generate_description(module_list, keywords, central)

        # Calculate total size and primary language
        total_size = 0
        language_counts: Counter[str] = Counter()
        for module_path in module_list:
            if module_path in self.graph.nodes:
                node = self.graph.nodes[module_path]
                total_size += node.size_bytes
                language_counts[node.language] += 1

        primary_language = "python"
        if language_counts:
            primary_language = language_counts.most_common(1)[0][0]

        return Domain(
            name=name,
            agent_id=agent_id,
            modules=module_list,
            central_module=central,
            keywords=keywords,
            folder_prefix=folder_prefix,
            description=description,
            total_size_bytes=total_size,
            primary_language=primary_language,
        )

    def _find_central_module(self, modules: list[str]) -> str:
        """Find the most central/important module in a cluster."""
        if len(modules) == 1:
            return modules[0]

        # Score each module by connections within the cluster
        scores: dict[str, float] = {}
        module_set = set(modules)

        for module in modules:
            if module not in self.graph.nodes:
                scores[module] = 0
                continue

            node = self.graph.nodes[module]

            # Count in-cluster connections
            imports_in_cluster = sum(1 for m in node.imports if m in module_set)
            imported_by_in_cluster = sum(
                1 for m in node.imported_by if m in module_set
            )

            # Favor modules that are imported by others (foundational)
            scores[module] = imported_by_in_cluster * 2 + imports_in_cluster

            # Bonus for __init__.py or main.py
            if module.endswith("__init__.py") or module.endswith("main.py"):
                scores[module] += 1

        # Return highest scoring module
        return max(modules, key=lambda m: scores.get(m, 0))

    def _extract_keywords(self, modules: list[str]) -> list[str]:
        """Extract meaningful keywords from module names and content."""
        all_words: Counter[str] = Counter()

        for module_path in modules:
            # Extract from file/folder names
            path = Path(module_path)
            name_words = re.findall(r"[a-z]+", path.stem.lower())
            for word in name_words:
                if len(word) > 2 and word not in {"__init__", "test", "tests"}:
                    all_words[word] += 2  # Higher weight for names

            # Extract from folder path
            for part in path.parts[:-1]:
                part_words = re.findall(r"[a-z]+", part.lower())
                for word in part_words:
                    if len(word) > 2:
                        all_words[word] += 1

            # Extract from function/class names
            if module_path in self.graph.nodes:
                node = self.graph.nodes[module_path]

                for func in node.functions:
                    func_words = re.findall(r"[a-z]+", func.lower())
                    for word in func_words:
                        if len(word) > 2:
                            all_words[word] += 1

                for cls in node.classes:
                    cls_words = re.findall(r"[a-z]+", cls.lower())
                    for word in cls_words:
                        if len(word) > 2:
                            all_words[word] += 1

        # Return top keywords
        return [word for word, _ in all_words.most_common(15)]

    def _generate_domain_name(self, modules: list[str], keywords: list[str]) -> str:
        """Generate a human-readable domain name."""
        # Try to use common folder name
        folders = [str(Path(m).parent) for m in modules]
        folder_counts = Counter(folders)

        if folder_counts:
            dominant_folder, count = folder_counts.most_common(1)[0]
            # If most modules share a folder, use it
            if count >= len(modules) * 0.5 and dominant_folder != ".":
                folder_name = Path(dominant_folder).name
                if folder_name and folder_name not in {"src", "lib", "app"}:
                    # Convert to title case
                    return folder_name.replace("_", " ").replace("-", " ").title()

        # Fall back to keywords
        if keywords:
            # Use first 1-2 meaningful keywords
            name_parts = []
            for kw in keywords[:2]:
                if kw not in {"init", "main", "base", "utils", "helpers"}:
                    name_parts.append(kw)

            if name_parts:
                return " ".join(name_parts).title()

        # Last resort: use central module name
        if modules:
            stem = Path(modules[0]).stem
            if stem != "__init__":
                return stem.replace("_", " ").title()

        return "Module Group"

    def _generate_agent_id(self, name: str) -> str:
        """Generate an agent ID from a domain name."""
        # Convert to snake_case
        agent_id = name.lower().replace(" ", "_")
        agent_id = re.sub(r"[^a-z0-9_]", "", agent_id)
        return f"{agent_id}_agent"

    def _find_common_prefix(self, modules: list[str]) -> str:
        """Find the common folder prefix for a set of modules."""
        if not modules:
            return ""

        paths = [Path(m).parts for m in modules]

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

    def _generate_description(
        self, modules: list[str], keywords: list[str], central: str
    ) -> str:
        """Generate a description for the domain."""
        parts = []

        # Describe what's in the domain
        if len(modules) == 1:
            stem = Path(modules[0]).stem
            parts.append(f"Handles {stem.replace('_', ' ')} functionality")
        else:
            parts.append(f"Expert on {len(modules)} related modules")

        # Add keyword context
        if keywords:
            top_keywords = keywords[:3]
            parts.append(f"covering {', '.join(top_keywords)}")

        return " ".join(parts)

    def _map_to_tier_a(self, domains: list[Domain]) -> None:
        """Map domains to related Tier A agents based on keywords."""
        for domain in domains:
            domain_keywords = set(kw.lower() for kw in domain.keywords)

            # Find Tier A agents with overlapping keywords
            related = []
            for agent_id, agent_keywords in self.tier_a_keywords.items():
                overlap = domain_keywords & agent_keywords
                if overlap:
                    related.append((agent_id, len(overlap)))

            # Sort by overlap and take top matches
            related.sort(key=lambda x: x[1], reverse=True)
            domain.related_tier_a = [agent_id for agent_id, _ in related[:3]]

    def get_domain_for_module(self, module_path: str) -> Domain | None:
        """Find which domain a specific module belongs to.

        Args:
            module_path: Relative path to the module.

        Returns:
            Domain containing this module, or None.
        """
        domains = self.analyze()
        for domain in domains:
            if module_path in domain.modules:
                return domain
        return None

    def get_summary(self) -> str:
        """Get a text summary of the domain analysis."""
        domains = self.analyze()

        lines = [
            f"Discovered {len(domains)} domains:",
            "",
        ]

        for domain in domains:
            lines.append(f"## {domain.name} [{domain.agent_id}]")
            lines.append(f"   Modules: {len(domain.modules)}")
            lines.append(f"   Central: {domain.central_module}")
            lines.append(f"   Keywords: {', '.join(domain.keywords[:5])}")
            if domain.related_tier_a:
                lines.append(f"   Related Tier A: {', '.join(domain.related_tier_a)}")
            lines.append("")

        return "\n".join(lines)
