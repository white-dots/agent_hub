"""Import graph analysis for dynamic domain detection.

This module builds and analyzes import dependency graphs to identify
natural module clusters without relying on hardcoded patterns.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImportEdge:
    """Represents an import relationship between two modules."""

    source: str  # File that imports (relative path)
    target: str  # File being imported (relative path)
    import_type: str  # "direct" | "from" | "relative"
    imported_names: list[str] = field(default_factory=list)  # Specific names imported


@dataclass
class ModuleNode:
    """Represents a module in the import graph."""

    path: str  # Relative path from project root
    imports: list[str] = field(default_factory=list)  # Modules this imports
    imported_by: list[str] = field(default_factory=list)  # Modules that import this
    functions: list[str] = field(default_factory=list)  # Function names defined
    classes: list[str] = field(default_factory=list)  # Class names defined


class ImportGraph:
    """Builds and analyzes import dependency graphs.

    This class parses Python files to extract import statements,
    resolves them to actual file paths, and provides methods to
    analyze the dependency structure.

    Example:
        >>> graph = ImportGraph("./my-project")
        >>> graph.build()
        >>> clusters = graph.get_clusters()
        >>> for cluster in clusters:
        ...     print(f"Cluster: {cluster}")
    """

    def __init__(self, root_path: str, ignore_patterns: list[str] | None = None):
        """Initialize the import graph.

        Args:
            root_path: Path to the project root directory.
            ignore_patterns: Patterns to ignore (e.g., ["test_*", "__pycache__"]).
        """
        self.root_path = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "*.egg-info",
            "build",
            "dist",
        ]

        self.nodes: dict[str, ModuleNode] = {}
        self.edges: list[ImportEdge] = []
        self._built = False

    def build(self) -> None:
        """Parse all Python files and build the import graph."""
        # Find all Python files
        python_files = self._find_python_files()

        # Parse each file
        for file_path in python_files:
            rel_path = str(file_path.relative_to(self.root_path))
            self._parse_file(file_path, rel_path)

        # Resolve imports to actual file paths
        self._resolve_imports()

        self._built = True

    def _find_python_files(self) -> list[Path]:
        """Find all Python files in the project."""
        files = []
        for path in self.root_path.rglob("*.py"):
            # Check ignore patterns
            rel_path = path.relative_to(self.root_path)
            if any(
                part in self.ignore_patterns or part.startswith(".")
                for part in rel_path.parts
            ):
                continue
            files.append(path)
        return files

    def _parse_file(self, file_path: Path, rel_path: str) -> None:
        """Parse a single Python file for imports and definitions."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError):
            return

        node = ModuleNode(path=rel_path)

        for ast_node in ast.walk(tree):
            # Extract imports
            if isinstance(ast_node, ast.Import):
                for alias in ast_node.names:
                    self.edges.append(
                        ImportEdge(
                            source=rel_path,
                            target=alias.name,
                            import_type="direct",
                            imported_names=[alias.asname or alias.name],
                        )
                    )
                    node.imports.append(alias.name)

            elif isinstance(ast_node, ast.ImportFrom):
                module = ast_node.module or ""
                level = ast_node.level  # Number of dots for relative import

                import_type = "relative" if level > 0 else "from"
                names = [alias.name for alias in ast_node.names]

                self.edges.append(
                    ImportEdge(
                        source=rel_path,
                        target=module,
                        import_type=import_type,
                        imported_names=names,
                    )
                )
                if module:
                    node.imports.append(module)

            # Extract function definitions
            elif isinstance(ast_node, ast.FunctionDef):
                if not ast_node.name.startswith("_"):
                    node.functions.append(ast_node.name)

            elif isinstance(ast_node, ast.AsyncFunctionDef):
                if not ast_node.name.startswith("_"):
                    node.functions.append(ast_node.name)

            # Extract class definitions
            elif isinstance(ast_node, ast.ClassDef):
                if not ast_node.name.startswith("_"):
                    node.classes.append(ast_node.name)

        self.nodes[rel_path] = node

    def _resolve_imports(self) -> None:
        """Resolve import targets to actual file paths within the project."""
        # Build a mapping of module names to file paths
        # We need multiple mappings to handle different import patterns
        module_map: dict[str, str] = {}

        # Determine the project package name from root path
        project_name = self.root_path.name

        for rel_path in self.nodes:
            # Convert path to module name
            # e.g., "auto/config.py" -> "auto.config"
            module_name = rel_path.replace("/", ".").replace("\\", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]

            # Map the full relative path
            module_map[module_name] = rel_path

            # Map with project name prefix (e.g., "agenthub.auto.config")
            module_map[f"{project_name}.{module_name}"] = rel_path

            # Map partial paths from the end
            parts = module_name.split(".")
            for i in range(len(parts)):
                partial = ".".join(parts[i:])
                if partial not in module_map:
                    module_map[partial] = rel_path

            # Map just the file stem (e.g., "config")
            stem = Path(rel_path).stem
            if stem != "__init__" and stem not in module_map:
                module_map[stem] = rel_path

        # Resolve each edge
        resolved_edges: list[ImportEdge] = []
        for edge in self.edges:
            target = edge.target

            # Skip stdlib and third-party imports
            if target in {
                "os", "sys", "re", "json", "typing", "pathlib", "datetime",
                "collections", "functools", "itertools", "dataclasses",
                "abc", "ast", "enum", "copy", "time", "logging", "argparse",
                "anthropic", "openai", "pydantic", "fastapi", "uvicorn",
                "watchdog", "dotenv",
            }:
                continue

            # Try to find the target in our module map
            resolved_path = module_map.get(target)

            # Try stripping common package prefixes
            if not resolved_path:
                for prefix in [f"{project_name}.", "src.", "lib.", "app."]:
                    if target.startswith(prefix):
                        stripped = target[len(prefix):]
                        resolved_path = module_map.get(stripped)
                        if resolved_path:
                            break

            # Try the last part of the import path
            if not resolved_path and "." in target:
                last_part = target.split(".")[-1]
                resolved_path = module_map.get(last_part)

            # Try partial matching from the end
            if not resolved_path:
                for mod_name, path in module_map.items():
                    # Check if our target is a suffix of a known module
                    if mod_name.endswith("." + target) or mod_name == target:
                        resolved_path = path
                        break
                    # Check if a known module is a suffix of our target
                    if target.endswith("." + mod_name):
                        resolved_path = path
                        break

            if resolved_path and resolved_path != edge.source:
                edge.target = resolved_path
                resolved_edges.append(edge)

                # Update imported_by list
                if resolved_path in self.nodes:
                    self.nodes[resolved_path].imported_by.append(edge.source)

        self.edges = resolved_edges

    def get_clusters(
        self, min_connections: int = 1, max_cluster_size: int = 15
    ) -> list[set[str]]:
        """Find natural clusters of modules based on import relationships.

        Uses a connected components algorithm, then splits large clusters
        based on folder structure to create more meaningful groupings.

        Args:
            min_connections: Minimum connections to be considered part of a cluster.
            max_cluster_size: Maximum modules per cluster before splitting by folder.

        Returns:
            List of sets, where each set contains module paths in a cluster.
        """
        if not self._built:
            self.build()

        # Build adjacency list (bidirectional)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

        # Filter nodes with enough connections
        connected_nodes = {
            node
            for node, neighbors in adjacency.items()
            if len(neighbors) >= min_connections
        }

        # Find connected components using BFS
        visited: set[str] = set()
        raw_clusters: list[set[str]] = []

        for start_node in connected_nodes:
            if start_node in visited:
                continue

            # BFS to find all connected nodes
            cluster: set[str] = set()
            queue = [start_node]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                cluster.add(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited and neighbor in connected_nodes:
                        queue.append(neighbor)

            if cluster:
                raw_clusters.append(cluster)

        # Add isolated nodes as single-module clusters
        for node_path in self.nodes:
            if node_path not in visited:
                raw_clusters.append({node_path})

        # Split large clusters by folder structure
        final_clusters: list[set[str]] = []
        for cluster in raw_clusters:
            if len(cluster) <= max_cluster_size:
                final_clusters.append(cluster)
            else:
                # Split by top-level folder
                subclusters = self._split_cluster_by_folder(cluster, adjacency)
                final_clusters.extend(subclusters)

        # Sort clusters by size (largest first)
        final_clusters.sort(key=lambda c: len(c), reverse=True)

        return final_clusters

    def _split_cluster_by_folder(
        self, cluster: set[str], adjacency: dict[str, set[str]]
    ) -> list[set[str]]:
        """Split a large cluster into smaller ones based on folder structure.

        Modules in the same top-level folder are grouped together,
        but only if they have internal connections.
        """
        # Group by top-level folder
        folder_groups: dict[str, set[str]] = defaultdict(set)
        for module in cluster:
            parts = Path(module).parts
            if len(parts) > 1:
                folder = parts[0]  # Top-level folder
            else:
                folder = "root"  # Files in root
            folder_groups[folder].add(module)

        # If we only have one folder, try splitting by subfolder
        if len(folder_groups) == 1:
            folder_groups = defaultdict(set)
            for module in cluster:
                parts = Path(module).parts
                if len(parts) > 2:
                    folder = "/".join(parts[:2])  # Two levels
                elif len(parts) > 1:
                    folder = parts[0]
                else:
                    folder = "root"
                folder_groups[folder].add(module)

        # If still only one group, return as-is
        if len(folder_groups) <= 1:
            return [cluster]

        # Return each folder group as a separate cluster
        subclusters = []
        for folder, modules in folder_groups.items():
            if modules:
                subclusters.append(modules)

        return subclusters

    def get_central_modules(self, top_n: int = 5) -> list[str]:
        """Get the most central modules based on import connections.

        Returns modules that are imported by many other modules (high in-degree).

        Args:
            top_n: Number of central modules to return.

        Returns:
            List of module paths sorted by centrality.
        """
        if not self._built:
            self.build()

        # Calculate in-degree for each module
        in_degree: dict[str, int] = defaultdict(int)
        for node_path, node in self.nodes.items():
            in_degree[node_path] = len(node.imported_by)

        # Sort by in-degree
        sorted_modules = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)

        return [path for path, _ in sorted_modules[:top_n]]

    def get_module_role(self, module_path: str) -> str:
        """Determine the role of a module in the graph.

        Args:
            module_path: Relative path to the module.

        Returns:
            One of: "hub" (many imports/exports), "leaf" (mostly imported),
            "bridge" (connects clusters), "isolated" (few connections).
        """
        if not self._built:
            self.build()

        if module_path not in self.nodes:
            return "unknown"

        node = self.nodes[module_path]
        imports_count = len(node.imports)
        imported_by_count = len(node.imported_by)
        total_connections = imports_count + imported_by_count

        if total_connections == 0:
            return "isolated"

        # High in both directions = hub
        if imports_count >= 3 and imported_by_count >= 3:
            return "hub"

        # Mostly imported, doesn't import much = leaf (foundational module)
        if imported_by_count >= 3 and imports_count <= 1:
            return "leaf"

        # Imports many but not imported = consumer
        if imports_count >= 3 and imported_by_count <= 1:
            return "consumer"

        # Moderate in both = bridge
        if imports_count >= 1 and imported_by_count >= 1:
            return "bridge"

        return "isolated"

    def get_module_neighbors(self, module_path: str) -> dict[str, list[str]]:
        """Get modules that this module imports and modules that import it.

        Args:
            module_path: Relative path to the module.

        Returns:
            Dict with "imports" and "imported_by" keys.
        """
        if not self._built:
            self.build()

        if module_path not in self.nodes:
            return {"imports": [], "imported_by": []}

        node = self.nodes[module_path]
        return {
            "imports": list(node.imports),
            "imported_by": list(node.imported_by),
        }

    def get_cluster_for_module(self, module_path: str) -> set[str]:
        """Get the cluster that contains a specific module.

        Args:
            module_path: Relative path to the module.

        Returns:
            Set of module paths in the same cluster.
        """
        clusters = self.get_clusters()
        for cluster in clusters:
            if module_path in cluster:
                return cluster
        return {module_path}

    def get_stats(self) -> dict:
        """Get statistics about the import graph.

        Returns:
            Dict with graph statistics.
        """
        if not self._built:
            self.build()

        clusters = self.get_clusters()

        return {
            "total_modules": len(self.nodes),
            "total_edges": len(self.edges),
            "num_clusters": len(clusters),
            "largest_cluster_size": len(clusters[0]) if clusters else 0,
            "isolated_modules": sum(1 for c in clusters if len(c) == 1),
            "hub_modules": sum(
                1 for p in self.nodes if self.get_module_role(p) == "hub"
            ),
            "leaf_modules": sum(
                1 for p in self.nodes if self.get_module_role(p) == "leaf"
            ),
        }
