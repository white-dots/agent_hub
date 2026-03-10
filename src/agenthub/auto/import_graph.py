from __future__ import annotations
"""Import graph analysis for dynamic domain detection.

This module builds and analyzes import dependency graphs to identify
natural module clusters without relying on hardcoded patterns.

Supports:
- Python (.py) files with AST parsing
- TypeScript/JavaScript (.ts, .tsx, .js, .jsx) files with regex parsing
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agenthub.auto.ignore import IgnorePatterns


@dataclass
class ImportEdge:
    """Represents an import relationship between two modules."""

    source: str  # File that imports (relative path)
    target: str  # File being imported (relative path)
    import_type: str  # "direct" | "from" | "relative" | "require"
    imported_names: list[str] = field(default_factory=list)  # Specific names imported


@dataclass
class ModuleNode:
    """Represents a module in the import graph."""

    path: str  # Relative path from project root
    language: str = "python"  # "python" | "typescript" | "javascript"
    imports: list[str] = field(default_factory=list)  # Modules this imports
    imported_by: list[str] = field(default_factory=list)  # Modules that import this
    functions: list[str] = field(default_factory=list)  # Function names defined
    classes: list[str] = field(default_factory=list)  # Class names defined
    size_bytes: int = 0  # File size in bytes


class ImportGraph:
    """Builds and analyzes import dependency graphs.

    This class parses Python and TypeScript/JavaScript files to extract
    import statements, resolves them to actual file paths, and provides
    methods to analyze the dependency structure.

    Supports:
    - Python (.py) using AST parsing
    - TypeScript (.ts, .tsx) using regex parsing
    - JavaScript (.js, .jsx) using regex parsing

    Example:
        >>> graph = ImportGraph("./my-project")
        >>> graph.build()
        >>> clusters = graph.get_clusters()
        >>> for cluster in clusters:
        ...     print(f"Cluster: {cluster}")
    """

    # Supported file extensions by language
    EXTENSIONS = {
        "python": [".py"],
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx"],
    }

    # Regex patterns for TypeScript/JavaScript imports
    TS_IMPORT_PATTERNS = [
        # import x from 'y'
        re.compile(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"),
        # import { x, y } from 'z'
        re.compile(r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"),
        # import * as x from 'y'
        re.compile(r"import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"),
        # import 'x' (side-effect import)
        re.compile(r"import\s+['\"]([^'\"]+)['\"]"),
        # const x = require('y')
        re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        # require('x')
        re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        # export { x } from 'y'
        re.compile(r"export\s+\{[^}]+\}\s+from\s+['\"]([^'\"]+)['\"]"),
        # export * from 'y'
        re.compile(r"export\s+\*\s+from\s+['\"]([^'\"]+)['\"]"),
    ]

    # Regex for TS/JS function and class definitions
    TS_FUNCTION_PATTERN = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|"
        r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>|"
        r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function"
    )
    TS_CLASS_PATTERN = re.compile(r"(?:export\s+)?class\s+(\w+)")

    def __init__(
        self,
        root_path: str,
        ignore_patterns: list[str] | None = None,
        extensions: list[str] | None = None,
    ):
        """Initialize the import graph.

        Args:
            root_path: Path to the project root directory.
            ignore_patterns: Additional patterns to ignore.
            extensions: File extensions to include. Defaults to all supported.
        """
        self.root_path = Path(root_path).resolve()

        # Use IgnorePatterns for smart filtering
        self._ignore = IgnorePatterns(root_path)

        # Additional ignore patterns (legacy support)
        self.ignore_patterns = ignore_patterns or []

        # Determine which extensions to use
        if extensions:
            self.extensions = extensions
        else:
            # Default: all supported extensions
            self.extensions = []
            for ext_list in self.EXTENSIONS.values():
                self.extensions.extend(ext_list)

        self.nodes: dict[str, ModuleNode] = {}
        self.edges: list[ImportEdge] = []
        self._built = False

    def build(self) -> None:
        """Parse all source files and build the import graph."""
        # Find all source files
        source_files = self._find_source_files()

        # Parse each file
        for file_path in source_files:
            rel_path = str(file_path.relative_to(self.root_path))
            self._parse_file(file_path, rel_path)

        # Resolve imports to actual file paths
        self._resolve_imports()

        self._built = True

    def _find_source_files(self) -> list[Path]:
        """Find all source files in the project."""
        files = []

        for ext in self.extensions:
            for path in self.root_path.rglob(f"*{ext}"):
                # Get relative path
                rel_path = path.relative_to(self.root_path)
                rel_str = str(rel_path)

                # Check .agenthubignore patterns
                if self._ignore.is_ignored(rel_str):
                    continue

                # Check legacy ignore patterns
                if any(
                    part in self.ignore_patterns or part.startswith(".")
                    for part in rel_path.parts
                ):
                    continue

                files.append(path)

        return files

    def _get_language(self, file_path: Path) -> str:
        """Determine the language of a file based on extension."""
        suffix = file_path.suffix.lower()
        for lang, exts in self.EXTENSIONS.items():
            if suffix in exts:
                return lang
        return "unknown"

    def _parse_file(self, file_path: Path, rel_path: str) -> None:
        """Parse a source file for imports and definitions."""
        language = self._get_language(file_path)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_size = file_path.stat().st_size
        except (OSError, UnicodeDecodeError):
            return

        if language == "python":
            self._parse_python_file(content, rel_path, file_size)
        elif language in ("typescript", "javascript"):
            self._parse_ts_js_file(content, rel_path, language, file_size)

    def _parse_python_file(
        self, content: str, rel_path: str, file_size: int
    ) -> None:
        """Parse a Python file using AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        node = ModuleNode(path=rel_path, language="python", size_bytes=file_size)

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

    def _parse_ts_js_file(
        self, content: str, rel_path: str, language: str, file_size: int
    ) -> None:
        """Parse a TypeScript/JavaScript file using regex."""
        node = ModuleNode(path=rel_path, language=language, size_bytes=file_size)

        # Extract imports using regex patterns
        for pattern in self.TS_IMPORT_PATTERNS:
            for match in pattern.finditer(content):
                groups = match.groups()
                # Last non-None group is typically the import path
                import_path = None
                imported_names = []

                for g in reversed(groups):
                    if g is not None:
                        if import_path is None:
                            import_path = g
                        else:
                            # This is imported names
                            if "," in g:
                                imported_names = [
                                    n.strip().split(" as ")[0].strip()
                                    for n in g.split(",")
                                ]
                            else:
                                imported_names = [g.strip()]

                if import_path:
                    # Determine import type
                    if import_path.startswith("."):
                        import_type = "relative"
                    elif "require" in pattern.pattern:
                        import_type = "require"
                    else:
                        import_type = "from"

                    self.edges.append(
                        ImportEdge(
                            source=rel_path,
                            target=import_path,
                            import_type=import_type,
                            imported_names=imported_names,
                        )
                    )
                    node.imports.append(import_path)

        # Extract functions
        for match in self.TS_FUNCTION_PATTERN.finditer(content):
            for name in match.groups():
                if name and not name.startswith("_"):
                    node.functions.append(name)

        # Extract classes
        for match in self.TS_CLASS_PATTERN.finditer(content):
            name = match.group(1)
            if name and not name.startswith("_"):
                node.classes.append(name)

        # Deduplicate
        node.functions = list(dict.fromkeys(node.functions))
        node.classes = list(dict.fromkeys(node.classes))
        node.imports = list(dict.fromkeys(node.imports))

        self.nodes[rel_path] = node

    def _resolve_imports(self) -> None:
        """Resolve import targets to actual file paths within the project."""
        # Build a mapping of module names to file paths
        # We need multiple mappings to handle different import patterns
        module_map: dict[str, str] = {}

        # Determine the project package name from root path
        project_name = self.root_path.name

        for rel_path in self.nodes:
            node = self.nodes[rel_path]

            if node.language == "python":
                # Python: Convert path to module name
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

            else:
                # TypeScript/JavaScript: Map by relative path patterns
                # e.g., "src/components/Button.tsx" -> various patterns

                # Map without extension
                no_ext = str(Path(rel_path).with_suffix(""))
                module_map[no_ext] = rel_path

                # Map with ./ prefix (common in TS/JS)
                module_map["./" + no_ext] = rel_path

                # Map relative from src/
                if no_ext.startswith("src/"):
                    module_map[no_ext[4:]] = rel_path
                    module_map["./" + no_ext[4:]] = rel_path

                # Map just the filename (without extension)
                stem = Path(rel_path).stem
                if stem != "index" and stem not in module_map:
                    module_map[stem] = rel_path

                # Map directory (for index.ts imports)
                if stem == "index":
                    dir_path = str(Path(rel_path).parent)
                    if dir_path not in module_map:
                        module_map[dir_path] = rel_path
                    module_map["./" + dir_path] = rel_path

        # TS/JS: Skip common third-party packages
        ts_js_stdlib = {
            "react", "react-dom", "next", "vue", "svelte", "angular",
            "express", "fastify", "koa", "hono",
            "axios", "fetch", "node-fetch",
            "lodash", "underscore", "ramda",
            "moment", "dayjs", "date-fns",
            "zod", "yup", "joi",
            "tailwindcss", "styled-components", "emotion",
            "@tanstack/react-query", "@reduxjs/toolkit", "redux",
            "zustand", "jotai", "recoil", "mobx",
            "prisma", "drizzle", "typeorm", "sequelize",
            "vitest", "jest", "mocha", "chai",
            "typescript", "ts-node",
            "path", "fs", "http", "https", "url", "crypto", "util",
            "child_process", "stream", "events", "os", "buffer",
        }

        # Python stdlib/third-party to skip
        python_stdlib = {
            "os", "sys", "re", "json", "typing", "pathlib", "datetime",
            "collections", "functools", "itertools", "dataclasses",
            "abc", "ast", "enum", "copy", "time", "logging", "argparse",
            "anthropic", "openai", "pydantic", "fastapi", "uvicorn",
            "watchdog", "dotenv", "asyncio", "concurrent", "threading",
            "multiprocessing", "socket", "http", "urllib", "email",
            "html", "xml", "sqlite3", "hashlib", "hmac", "secrets",
            "base64", "binascii", "struct", "codecs", "io", "tempfile",
            "shutil", "glob", "fnmatch", "linecache", "pickle", "shelve",
            "csv", "configparser", "tomllib", "subprocess", "signal",
            "contextvars", "inspect", "traceback", "warnings", "unittest",
            "pytest", "numpy", "pandas", "scipy", "matplotlib", "sklearn",
            "sqlalchemy", "alembic", "redis", "celery", "requests", "httpx",
            "aiohttp", "flask", "django", "starlette", "pydantic_settings",
        }

        # Resolve each edge
        resolved_edges: list[ImportEdge] = []
        for edge in self.edges:
            target = edge.target
            source_node = self.nodes.get(edge.source)
            source_lang = source_node.language if source_node else "python"

            # Skip stdlib and third-party imports
            skip_set = python_stdlib if source_lang == "python" else ts_js_stdlib

            # Check if target is a known library
            target_base = target.split("/")[0].split(".")[0].lstrip("@")
            if target_base in skip_set or target.startswith("@"):
                # Skip @scoped packages unless they resolve locally
                if target.startswith("@") and target not in module_map:
                    continue
                elif target_base in skip_set:
                    continue

            # Try to find the target in our module map
            resolved_path = module_map.get(target)

            # For relative TS/JS imports, resolve the path
            if not resolved_path and target.startswith("."):
                source_dir = str(Path(edge.source).parent)
                # Normalize the relative import
                if target.startswith("./"):
                    candidate = source_dir + "/" + target[2:] if source_dir != "." else target[2:]
                elif target.startswith("../"):
                    parts = source_dir.split("/")
                    target_parts = target.split("/")
                    up_count = 0
                    for p in target_parts:
                        if p == "..":
                            up_count += 1
                        else:
                            break
                    base = "/".join(parts[:-up_count]) if up_count <= len(parts) else ""
                    rest = "/".join(target_parts[up_count:])
                    candidate = base + "/" + rest if base else rest
                else:
                    candidate = target

                # Try with different extensions
                for ext in ["", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"]:
                    test_path = candidate + ext
                    if test_path in module_map:
                        resolved_path = module_map[test_path]
                        break
                    # Also check direct match in nodes
                    if test_path in self.nodes:
                        resolved_path = test_path
                        break

            # Try stripping common package prefixes (Python)
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

        # Calculate size stats
        total_size = sum(n.size_bytes for n in self.nodes.values())
        py_count = sum(1 for n in self.nodes.values() if n.language == "python")
        ts_count = sum(1 for n in self.nodes.values() if n.language == "typescript")
        js_count = sum(1 for n in self.nodes.values() if n.language == "javascript")

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
            "total_size_bytes": total_size,
            "total_size_kb": total_size / 1024,
            "python_modules": py_count,
            "typescript_modules": ts_count,
            "javascript_modules": js_count,
        }

    def get_cluster_size(self, cluster: set[str]) -> int:
        """Get total size in bytes of all modules in a cluster.

        Args:
            cluster: Set of module paths.

        Returns:
            Total size in bytes.
        """
        return sum(
            self.nodes[path].size_bytes
            for path in cluster
            if path in self.nodes
        )

    def get_modules_by_language(self, language: str) -> list[str]:
        """Get all module paths for a specific language.

        Args:
            language: "python", "typescript", or "javascript"

        Returns:
            List of module paths.
        """
        return [
            path for path, node in self.nodes.items()
            if node.language == language
        ]

    # ------------------------------------------------------------------
    # Impact analysis methods
    # ------------------------------------------------------------------

    def is_test_file(self, path: str) -> bool:
        """Check if a file path is a test file by naming/path convention.

        Detects:
        - Python: test_*.py, *_test.py, conftest.py, tests/ directory
        - TS/JS: *.test.ts, *.spec.ts, *.test.tsx, *.spec.tsx,
                 __tests__/ directory, *.test.js, *.spec.js
        """
        p = Path(path)
        name = p.name.lower()
        parts_lower = [part.lower() for part in p.parts]

        # Directory-based
        if "tests" in parts_lower or "test" in parts_lower or "__tests__" in parts_lower:
            return True

        # Python
        if name.startswith("test_") and name.endswith(".py"):
            return True
        if name.endswith("_test.py"):
            return True
        if name == "conftest.py":
            return True

        # TS/JS
        for ext in (".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx",
                     ".test.js", ".spec.js", ".test.jsx", ".spec.jsx"):
            if name.endswith(ext):
                return True

        return False

    def get_transitive_importers(self, path: str, max_depth: int = 10) -> list[str]:
        """Get all files that transitively depend on this file.

        Walks imported_by edges via BFS up to max_depth hops.

        Args:
            path: Relative path to the module.
            max_depth: Maximum traversal depth.

        Returns:
            List of module paths that transitively import this file.
            Does NOT include the input path itself.
        """
        if not self._built:
            self.build()

        if path not in self.nodes:
            return []

        visited: set[str] = {path}
        queue: list[tuple[str, int]] = [(path, 0)]
        result: list[str] = []

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            node = self.nodes.get(current)
            if not node:
                continue

            for importer in node.imported_by:
                if importer not in visited:
                    visited.add(importer)
                    result.append(importer)
                    queue.append((importer, depth + 1))

        return result

    def get_affected_tests(self, paths: list[str]) -> list[str]:
        """Get test files that could be affected by changes to the given files.

        For each input path, finds transitive importers and filters to test files.

        Args:
            paths: List of relative file paths that were changed.

        Returns:
            Sorted, deduplicated list of affected test file paths.
        """
        if not self._built:
            self.build()

        test_files: set[str] = set()

        for path in paths:
            # The changed file itself might be a test
            if path in self.nodes and self.is_test_file(path):
                test_files.add(path)

            # Check all transitive importers
            for importer in self.get_transitive_importers(path):
                if self.is_test_file(importer):
                    test_files.add(importer)

        return sorted(test_files)

    def get_exported_interface(self, path: str) -> dict:
        """Get the exported interface of a file.

        Re-parses the file to extract the current exported interface.
        For Python, respects __all__ if present.

        Args:
            path: Relative path to the module.

        Returns:
            Dict with keys: classes, functions, constants, language.
        """
        empty = {"classes": [], "functions": [], "constants": [], "language": "unknown"}

        node = self.nodes.get(path) if self._built else None
        language = node.language if node else self._guess_language(path)
        empty["language"] = language

        full_path = self.root_path / path
        if not full_path.exists():
            return empty

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return empty

        if language == "python":
            return self._get_python_interface(content)
        elif language in ("typescript", "javascript"):
            return self._get_ts_js_interface(content, language)
        return empty

    def _guess_language(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        for lang, exts in self.EXTENSIONS.items():
            if suffix in exts:
                return lang
        return "unknown"

    def _get_python_interface(self, content: str) -> dict:
        """Extract exported interface from Python source."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"classes": [], "functions": [], "constants": [], "language": "python"}

        # Check for __all__
        all_names: set[str] | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            all_names = set()
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_names.add(elt.value)

        classes = []
        functions = []
        constants = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if all_names is not None and node.name not in all_names:
                    continue
                if node.name.startswith("_") and all_names is None:
                    continue
                methods = []
                bases = [self._ast_name(b) for b in node.bases if self._ast_name(b)]
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not item.name.startswith("_") or item.name in ("__init__", "__call__"):
                            methods.append(item.name)
                classes.append({"name": node.name, "methods": methods, "bases": bases})

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if all_names is not None and node.name not in all_names:
                    continue
                if node.name.startswith("_") and all_names is None:
                    continue
                args = [a.arg for a in node.args.args if a.arg != "self"]
                functions.append({
                    "name": node.name,
                    "args": args,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                })

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = []
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.append(target.id)
                for name in names:
                    if all_names is not None and name not in all_names:
                        continue
                    if name.startswith("_") and all_names is None:
                        continue
                    if name == "__all__":
                        continue
                    # Only include UPPER_CASE constants
                    if name.isupper() or (all_names is not None and name in all_names):
                        ann = None
                        if isinstance(node, ast.AnnAssign) and node.annotation:
                            ann = ast.dump(node.annotation)
                        constants.append({"name": name, "type_annotation": ann})

        return {"classes": classes, "functions": functions, "constants": constants, "language": "python"}

    @staticmethod
    def _ast_name(node) -> str:
        """Extract name string from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    # Regex for exported TS/JS interface extraction
    _TS_EXPORT_FUNC = re.compile(
        r"export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
    )
    _TS_EXPORT_CONST_FUNC = re.compile(
        r"export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>"
    )
    _TS_EXPORT_CLASS = re.compile(
        r"export\s+(?:default\s+)?class\s+(\w+)"
    )
    _TS_EXPORT_CONST = re.compile(
        r"export\s+const\s+(\w+)\s*(?::\s*\w[^=]*)?\s*="
    )
    _TS_EXPORT_TYPE = re.compile(
        r"export\s+(?:type|interface)\s+(\w+)"
    )

    def _get_ts_js_interface(self, content: str, language: str) -> dict:
        """Extract exported interface from TypeScript/JavaScript source."""
        classes = []
        functions = []
        constants = []

        # Exported functions
        func_names: set[str] = set()
        for match in self._TS_EXPORT_FUNC.finditer(content):
            name = match.group(1)
            args_str = match.group(2).strip()
            args = [a.strip().split(":")[0].strip() for a in args_str.split(",") if a.strip()] if args_str else []
            functions.append({"name": name, "args": args, "is_async": "async" in match.group(0)})
            func_names.add(name)

        # Arrow function exports
        for match in self._TS_EXPORT_CONST_FUNC.finditer(content):
            name = match.group(1)
            if name not in func_names:
                functions.append({"name": name, "args": [], "is_async": "async" in match.group(0)})
                func_names.add(name)

        # Exported classes
        for match in self._TS_EXPORT_CLASS.finditer(content):
            classes.append({"name": match.group(1), "methods": [], "bases": []})

        # Exported constants (exclude those already captured as functions)
        for match in self._TS_EXPORT_CONST.finditer(content):
            name = match.group(1)
            if name not in func_names:
                constants.append({"name": name, "type_annotation": None})

        # Exported types/interfaces
        for match in self._TS_EXPORT_TYPE.finditer(content):
            constants.append({"name": match.group(1), "type_annotation": "type"})

        return {"classes": classes, "functions": functions, "constants": constants, "language": language}
