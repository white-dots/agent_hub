"""Intelligent codebase discovery for automatic agent generation.

This module provides semantic analysis of codebases to automatically:
1. Detect project type (FastAPI, Django, Flask, React, Next.js, etc.)
2. Extract meaningful module descriptions from code
3. Identify API endpoints, database models, services, components
4. Generate smart keywords from code content
5. Build rich context automatically

Supports:
- Python (.py) files with AST parsing
- TypeScript/JavaScript (.ts, .tsx, .js, .jsx) files with regex parsing
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agenthub.auto.ignore import IgnorePatterns


@dataclass
class ModuleInfo:
    """Extracted information about a code module."""

    path: Path
    name: str
    description: str
    module_type: str  # "api", "service", "model", "util", "component", etc.
    language: str = "python"  # "python", "typescript", "javascript"
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    docstring: str = ""
    size_kb: float = 0
    line_count: int = 0


@dataclass
class ProjectProfile:
    """Detected profile of the entire project."""

    framework: str  # "fastapi", "django", "flask", "react", "nextjs", "vue", "generic"
    project_type: str  # "web_api", "cli", "library", "data_pipeline", "frontend", "fullstack"
    language: str  # "python", "typescript", "javascript", "mixed"
    modules: list[ModuleInfo] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    database_models: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)  # React/Vue components
    total_size_kb: float = 0


class CodebaseDiscovery:
    """Intelligent codebase analyzer that extracts semantic information.

    Unlike simple folder-based analysis, this class:
    - Parses Python AST to extract classes, functions, docstrings
    - Parses TypeScript/JavaScript using regex for components, functions
    - Detects project framework (FastAPI, Django, React, Next.js, etc.)
    - Identifies module types (API routes, services, models, components)
    - Extracts meaningful keywords from code
    - Builds rich descriptions for agents

    Example:
        >>> discovery = CodebaseDiscovery("./my-project")
        >>> profile = discovery.analyze()
        >>> print(profile.framework)  # "fastapi" or "react"
        >>> for module in profile.modules:
        ...     print(f"{module.name}: {module.description}")
    """

    # Supported file extensions
    EXTENSIONS = {
        "python": [".py"],
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx"],
    }

    # Patterns to detect module types
    MODULE_TYPE_PATTERNS = {
        "api": ["api", "routes", "endpoints", "views", "routers"],
        "service": ["service", "services", "logic", "business"],
        "model": ["model", "models", "schema", "schemas", "entities"],
        "repository": ["repository", "repositories", "dao", "dal"],
        "util": ["util", "utils", "helper", "helpers", "common", "lib"],
        "config": ["config", "settings", "configuration"],
        "test": ["test", "tests", "spec", "specs", "__tests__"],
        "migration": ["migration", "migrations", "alembic"],
        # Frontend-specific types
        "component": ["component", "components"],
        "page": ["page", "pages", "views", "screens"],
        "hook": ["hook", "hooks"],
        "store": ["store", "stores", "state", "redux", "zustand"],
        "style": ["style", "styles", "css", "theme"],
    }

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        # Backend
        "fastapi": ["from fastapi", "import fastapi", "FastAPI()", "@app.get", "@app.post", "@router."],
        "django": ["from django", "import django", "INSTALLED_APPS", "urlpatterns"],
        "flask": ["from flask", "import flask", "Flask(__name__)", "@app.route"],
        "sqlalchemy": ["from sqlalchemy", "import sqlalchemy", "Base.metadata", "declarative_base"],
        # Frontend
        "react": ["from 'react'", "from \"react\"", "import React", "useState", "useEffect", "jsx", "tsx"],
        "nextjs": ["from 'next", "from \"next", "next/router", "next/link", "getServerSideProps", "getStaticProps"],
        "vue": ["from 'vue'", "from \"vue\"", "defineComponent", "ref(", "reactive(", ".vue"],
        "svelte": [".svelte", "from 'svelte'"],
        "angular": ["@angular", "from '@angular"],
    }

    # Regex for TS/JS parsing
    TS_FUNCTION_PATTERN = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|"
        r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>|"
        r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?function"
    )
    TS_CLASS_PATTERN = re.compile(r"(?:export\s+)?class\s+(\w+)")
    TS_COMPONENT_PATTERN = re.compile(
        r"(?:export\s+)?(?:default\s+)?(?:function|const)\s+([A-Z]\w+)|"
        r"(?:export\s+)?class\s+([A-Z]\w+)\s+extends\s+(?:React\.)?Component"
    )
    TS_IMPORT_PATTERN = re.compile(r"(?:import|from)\s+['\"]([^'\"]+)['\"]")

    def __init__(
        self,
        root_path: str,
        ignore_patterns: list[str] | None = None,
        extensions: list[str] | None = None,
    ):
        """Initialize CodebaseDiscovery.

        Args:
            root_path: Root directory of the project.
            ignore_patterns: Additional folder/file patterns to ignore.
            extensions: File extensions to include. Defaults to all supported.
        """
        self.root = Path(root_path).resolve()

        # Use IgnorePatterns for smart filtering
        self._ignore = IgnorePatterns(root_path)

        # Legacy ignore patterns support
        self.ignore_patterns = ignore_patterns or []

        # Determine which extensions to use
        if extensions:
            self.extensions = extensions
        else:
            self.extensions = []
            for ext_list in self.EXTENSIONS.values():
                self.extensions.extend(ext_list)

    def analyze(self) -> ProjectProfile:
        """Analyze the entire codebase and return a project profile.

        Returns:
            ProjectProfile with detected framework, modules, and metadata.
        """
        # Collect all source files
        source_files = list(self._find_source_files())

        # Detect framework from imports
        framework = self._detect_framework(source_files)

        # Analyze each module
        modules = []
        total_size = 0
        for source_file in source_files:
            module_info = self._analyze_module(source_file)
            if module_info:
                modules.append(module_info)
                total_size += module_info.size_kb

        # Extract global info
        entry_points = self._find_entry_points(modules)
        db_models = self._find_database_models(modules)
        api_endpoints = self._find_api_endpoints(modules)
        components = self._find_components(modules)

        # Determine project type and language
        project_type = self._determine_project_type(framework, modules)
        language = self._determine_language(modules)

        return ProjectProfile(
            framework=framework,
            project_type=project_type,
            language=language,
            modules=modules,
            entry_points=entry_points,
            database_models=db_models,
            api_endpoints=api_endpoints,
            components=components,
            total_size_kb=total_size,
        )

    def _find_source_files(self):
        """Find all source files, respecting ignore patterns."""
        for ext in self.extensions:
            for source_file in self.root.rglob(f"*{ext}"):
                rel_path = source_file.relative_to(self.root)
                rel_str = str(rel_path)

                # Check .agenthubignore patterns
                if self._ignore.is_ignored(rel_str):
                    continue

                # Check legacy ignore patterns
                skip = False
                for pattern in self.ignore_patterns:
                    if pattern in str(source_file):
                        skip = True
                        break
                if not skip:
                    yield source_file

    def _get_language(self, file_path: Path) -> str:
        """Determine the language of a file based on extension."""
        suffix = file_path.suffix.lower()
        for lang, exts in self.EXTENSIONS.items():
            if suffix in exts:
                return lang
        return "unknown"

    def _determine_language(self, modules: list[ModuleInfo]) -> str:
        """Determine the primary language of the project."""
        lang_counts: dict[str, int] = {}
        for m in modules:
            lang_counts[m.language] = lang_counts.get(m.language, 0) + 1

        if not lang_counts:
            return "unknown"

        # Check if mixed
        total = sum(lang_counts.values())
        for lang, count in lang_counts.items():
            if count / total > 0.8:
                return lang

        return "mixed"

    def _detect_framework(self, source_files: list[Path]) -> str:
        """Detect the framework used in the project."""
        framework_scores: dict[str, int] = {}

        for source_file in source_files[:100]:  # Sample first 100 files
            try:
                content = source_file.read_text(encoding="utf-8", errors="ignore")
                for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in content:
                            framework_scores[framework] = framework_scores.get(framework, 0) + 1
            except Exception:
                continue

        if framework_scores:
            return max(framework_scores, key=framework_scores.get)
        return "generic"

    def _analyze_module(self, source_file: Path) -> Optional[ModuleInfo]:
        """Analyze a single source module."""
        language = self._get_language(source_file)

        if language == "python":
            return self._analyze_python_module(source_file)
        elif language in ("typescript", "javascript"):
            return self._analyze_ts_js_module(source_file, language)
        return None

    def _analyze_python_module(self, py_file: Path) -> Optional[ModuleInfo]:
        """Analyze a single Python module using AST."""
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            return None

        # Extract basic info
        relative_path = py_file.relative_to(self.root)
        name = py_file.stem

        # Get module docstring
        docstring = ast.get_docstring(tree) or ""

        # Extract classes and functions
        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])

        # Determine module type
        module_type = self._classify_module_type(py_file, content, classes, functions)

        # Generate keywords
        keywords = self._extract_keywords(name, classes, functions, content)

        # Generate description
        description = self._generate_description(
            name, module_type, classes, functions, docstring
        )

        return ModuleInfo(
            path=relative_path,
            name=name,
            description=description,
            module_type=module_type,
            language="python",
            classes=classes,
            functions=functions,
            imports=list(set(imports)),
            keywords=keywords,
            docstring=docstring[:500] if docstring else "",
            size_kb=len(content) / 1024,
            line_count=len(content.splitlines()),
        )

    def _analyze_ts_js_module(self, source_file: Path, language: str) -> Optional[ModuleInfo]:
        """Analyze a TypeScript/JavaScript module using regex."""
        try:
            content = source_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        # Extract basic info
        relative_path = source_file.relative_to(self.root)
        name = source_file.stem

        # Extract first comment block as docstring
        docstring = ""
        doc_match = re.search(r"/\*\*\s*(.*?)\s*\*/", content, re.DOTALL)
        if doc_match:
            docstring = doc_match.group(1).strip()
            # Clean up * prefixes
            docstring = re.sub(r"^\s*\*\s?", "", docstring, flags=re.MULTILINE)

        # Extract classes
        classes = []
        for match in self.TS_CLASS_PATTERN.finditer(content):
            classes.append(match.group(1))

        # Extract functions
        functions = []
        for match in self.TS_FUNCTION_PATTERN.finditer(content):
            for name_group in match.groups():
                if name_group and not name_group.startswith("_"):
                    functions.append(name_group)

        # Extract components (PascalCase functions in TSX/JSX)
        components = []
        if source_file.suffix in (".tsx", ".jsx"):
            for match in self.TS_COMPONENT_PATTERN.finditer(content):
                for name_group in match.groups():
                    if name_group:
                        components.append(name_group)

        # Extract imports
        imports = []
        for match in self.TS_IMPORT_PATTERN.finditer(content):
            imp = match.group(1)
            # Get base package name
            if imp.startswith("."):
                imports.append(imp)
            else:
                imports.append(imp.split("/")[0].lstrip("@"))

        # Deduplicate
        functions = list(dict.fromkeys(functions))
        classes = list(dict.fromkeys(classes + components))

        # Determine module type
        module_type = self._classify_ts_js_module_type(source_file, content, classes, functions)

        # Generate keywords
        keywords = self._extract_keywords(name, classes, functions, content)

        # Generate description
        description = self._generate_ts_js_description(
            name, module_type, classes, functions, components, docstring
        )

        return ModuleInfo(
            path=relative_path,
            name=name,
            description=description,
            module_type=module_type,
            language=language,
            classes=classes,
            functions=functions,
            imports=list(set(imports)),
            keywords=keywords,
            docstring=docstring[:500] if docstring else "",
            size_kb=len(content) / 1024,
            line_count=len(content.splitlines()),
        )

    def _classify_ts_js_module_type(
        self,
        source_file: Path,
        content: str,
        classes: list[str],
        functions: list[str],
    ) -> str:
        """Classify TypeScript/JavaScript module type."""
        path_str = str(source_file).lower()
        file_name = source_file.name.lower()

        # Check path patterns
        for module_type, patterns in self.MODULE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if f"/{pattern}/" in path_str or path_str.endswith(f"/{pattern}"):
                    return module_type

        # Check file naming conventions
        if file_name.endswith(".test.ts") or file_name.endswith(".spec.ts"):
            return "test"
        if file_name.endswith(".test.tsx") or file_name.endswith(".spec.tsx"):
            return "test"

        # Check content patterns for React
        if "useState" in content or "useEffect" in content:
            if source_file.suffix in (".tsx", ".jsx"):
                return "component"

        # Check for hooks (use* functions)
        hook_pattern = re.compile(r"export\s+(?:function|const)\s+use[A-Z]\w+")
        if hook_pattern.search(content):
            return "hook"

        # Check for API routes (Next.js style)
        if "pages/api/" in path_str or "app/api/" in path_str:
            return "api"

        # Check for pages
        if "/pages/" in path_str or "/app/" in path_str:
            if source_file.suffix in (".tsx", ".jsx"):
                return "page"

        # Check for store/state management
        if "createSlice" in content or "createStore" in content or "create(" in content:
            return "store"

        # Default based on extension
        if source_file.suffix in (".tsx", ".jsx"):
            # Check if it exports a component (PascalCase)
            if any(c[0].isupper() for c in classes if c):
                return "component"

        return "module"

    def _generate_ts_js_description(
        self,
        name: str,
        module_type: str,
        classes: list[str],
        functions: list[str],
        components: list[str],
        docstring: str,
    ) -> str:
        """Generate description for TypeScript/JavaScript module."""
        # Use docstring if available
        if docstring:
            first_line = docstring.split("\n")[0].strip()
            if len(first_line) > 20:
                return first_line[:200]

        type_labels = {
            "api": "API routes",
            "component": "React component",
            "page": "Page component",
            "hook": "React hook",
            "store": "State management",
            "util": "Utilities",
            "service": "Service layer",
            "test": "Tests",
            "style": "Styles",
        }

        type_label = type_labels.get(module_type, "Module")

        if components:
            comp_str = ", ".join(components[:3])
            if len(components) > 3:
                comp_str += f" (+{len(components) - 3} more)"
            return f"{type_label}: {comp_str}"

        if classes:
            class_str = ", ".join(classes[:3])
            if len(classes) > 3:
                class_str += f" (+{len(classes) - 3} more)"
            return f"{type_label} with {class_str}"

        if functions:
            func_str = ", ".join(functions[:3])
            if len(functions) > 3:
                func_str += f" (+{len(functions) - 3} more)"
            return f"{type_label}: {func_str}"

        return f"{type_label} for {name}"

    def _classify_module_type(
        self,
        py_file: Path,
        content: str,
        classes: list[str],
        functions: list[str]
    ) -> str:
        """Classify module type based on path, content, and structure."""
        path_str = str(py_file).lower()

        # Check path patterns
        for module_type, patterns in self.MODULE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in path_str:
                    return module_type

        # Check content patterns
        if any(x in content for x in ["@app.", "@router.", "APIRouter", "FastAPI"]):
            return "api"
        if any(x in content for x in ["Base.metadata", "Column(", "relationship("]):
            return "model"
        if "Service" in "".join(classes):
            return "service"

        return "module"

    def _extract_keywords(
        self,
        name: str,
        classes: list[str],
        functions: list[str],
        content: str
    ) -> list[str]:
        """Extract meaningful keywords from module."""
        keywords = set()

        # Add module name parts
        for part in re.split(r"[_\-]", name):
            if len(part) > 2:
                keywords.add(part.lower())

        # Add class names (split camelCase)
        for cls in classes:
            keywords.add(cls.lower())
            # Split CamelCase
            parts = re.findall(r"[A-Z][a-z]+", cls)
            keywords.update(p.lower() for p in parts if len(p) > 2)

        # Add function names
        for func in functions[:20]:  # Limit
            parts = func.split("_")
            keywords.update(p.lower() for p in parts if len(p) > 2)

        # Extract domain terms from content
        domain_terms = re.findall(r"\b(product|order|user|customer|price|payment|inventory|cart|shipping|campaign|ad|stat|analytics)\w*", content.lower())
        keywords.update(domain_terms[:10])

        return list(keywords)[:30]

    def _generate_description(
        self,
        name: str,
        module_type: str,
        classes: list[str],
        functions: list[str],
        docstring: str,
    ) -> str:
        """Generate a human-readable description for the module."""
        # Use docstring if available
        if docstring:
            first_line = docstring.split("\n")[0].strip()
            if len(first_line) > 20:
                return first_line[:200]

        # Generate from structure
        type_labels = {
            "api": "API endpoints",
            "service": "Business logic",
            "model": "Data models",
            "repository": "Data access",
            "util": "Utilities",
            "config": "Configuration",
            "test": "Tests",
        }

        type_label = type_labels.get(module_type, "Module")

        if classes:
            class_str = ", ".join(classes[:3])
            if len(classes) > 3:
                class_str += f" (+{len(classes) - 3} more)"
            return f"{type_label} with {class_str}"

        if functions:
            func_str = ", ".join(functions[:3])
            if len(functions) > 3:
                func_str += f" (+{len(functions) - 3} more)"
            return f"{type_label}: {func_str}"

        return f"{type_label} for {name}"

    def _find_entry_points(self, modules: list[ModuleInfo]) -> list[str]:
        """Find likely entry points (main.py, app.py, etc.)."""
        entry_names = ["main", "app", "server", "cli", "run", "__main__"]
        return [
            str(m.path) for m in modules
            if m.name in entry_names
        ]

    def _find_database_models(self, modules: list[ModuleInfo]) -> list[str]:
        """Find database model classes."""
        models = []
        for m in modules:
            if m.module_type == "model":
                models.extend(m.classes)
        return models

    def _find_api_endpoints(self, modules: list[ModuleInfo]) -> list[str]:
        """Find API endpoint functions."""
        endpoints = []
        for m in modules:
            if m.module_type == "api":
                endpoints.extend(m.functions)
        return endpoints

    def _find_components(self, modules: list[ModuleInfo]) -> list[str]:
        """Find React/Vue components."""
        components = []
        for m in modules:
            if m.module_type in ("component", "page"):
                # Add PascalCase classes (likely components)
                for cls in m.classes:
                    if cls[0].isupper():
                        components.append(cls)
        return components

    def _determine_project_type(self, framework: str, modules: list[ModuleInfo]) -> str:
        """Determine overall project type."""
        has_api = any(m.module_type == "api" for m in modules)
        has_models = any(m.module_type == "model" for m in modules)
        has_components = any(m.module_type in ("component", "page") for m in modules)
        has_frontend = any(m.language in ("typescript", "javascript") for m in modules)
        has_backend = any(m.language == "python" for m in modules)

        # Check for fullstack
        if has_backend and has_frontend and (has_api or has_components):
            return "fullstack"

        # Backend frameworks
        if framework in ("fastapi", "flask", "django") and has_api:
            return "web_api"

        # Frontend frameworks
        if framework in ("react", "nextjs", "vue", "angular", "svelte"):
            return "frontend"

        if has_components and not has_api:
            return "frontend"

        if has_models and not has_api:
            return "data_pipeline"

        if any(m.name == "cli" for m in modules):
            return "cli"

        return "library"


def discover_and_summarize(project_root: str) -> str:
    """One-liner to discover and summarize a project.

    Args:
        project_root: Path to project.

    Returns:
        Human-readable summary of the project.
    """
    discovery = CodebaseDiscovery(project_root)
    profile = discovery.analyze()

    lines = [
        f"Project Type: {profile.project_type} ({profile.framework})",
        f"Modules: {len(profile.modules)}",
        f"Database Models: {len(profile.database_models)}",
        f"API Endpoints: {len(profile.api_endpoints)}",
        "",
        "Key Modules:",
    ]

    # Group by type
    by_type: dict[str, list[ModuleInfo]] = {}
    for m in profile.modules:
        by_type.setdefault(m.module_type, []).append(m)

    for module_type, mods in sorted(by_type.items()):
        lines.append(f"\n  [{module_type.upper()}]")
        for m in mods[:5]:  # Top 5 per type
            lines.append(f"    - {m.path}: {m.description}")
        if len(mods) > 5:
            lines.append(f"    ... and {len(mods) - 5} more")

    return "\n".join(lines)
