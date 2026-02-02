"""Intelligent codebase discovery for automatic agent generation.

This module provides semantic analysis of codebases to automatically:
1. Detect project type (FastAPI, Django, Flask, etc.)
2. Extract meaningful module descriptions from code
3. Identify API endpoints, database models, services
4. Generate smart keywords from code content
5. Build rich context automatically
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModuleInfo:
    """Extracted information about a code module."""

    path: Path
    name: str
    description: str
    module_type: str  # "api", "service", "model", "util", etc.
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

    framework: str  # "fastapi", "django", "flask", "generic"
    project_type: str  # "web_api", "cli", "library", "data_pipeline"
    language: str  # "python", "typescript", etc.
    modules: list[ModuleInfo] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    database_models: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)


class CodebaseDiscovery:
    """Intelligent codebase analyzer that extracts semantic information.

    Unlike simple folder-based analysis, this class:
    - Parses Python AST to extract classes, functions, docstrings
    - Detects project framework (FastAPI, Django, etc.)
    - Identifies module types (API routes, services, models)
    - Extracts meaningful keywords from code
    - Builds rich descriptions for agents

    Example:
        >>> discovery = CodebaseDiscovery("./my-project")
        >>> profile = discovery.analyze()
        >>> print(profile.framework)  # "fastapi"
        >>> for module in profile.modules:
        ...     print(f"{module.name}: {module.description}")
    """

    # Patterns to detect module types
    MODULE_TYPE_PATTERNS = {
        "api": ["api", "routes", "endpoints", "views", "routers"],
        "service": ["service", "services", "logic", "business"],
        "model": ["model", "models", "schema", "schemas", "entities"],
        "repository": ["repository", "repositories", "dao", "dal"],
        "util": ["util", "utils", "helper", "helpers", "common"],
        "config": ["config", "settings", "configuration"],
        "test": ["test", "tests", "spec", "specs"],
        "migration": ["migration", "migrations", "alembic"],
    }

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "fastapi": ["from fastapi", "import fastapi", "FastAPI()", "@app.get", "@app.post", "@router."],
        "django": ["from django", "import django", "INSTALLED_APPS", "urlpatterns"],
        "flask": ["from flask", "import flask", "Flask(__name__)", "@app.route"],
        "sqlalchemy": ["from sqlalchemy", "import sqlalchemy", "Base.metadata", "declarative_base"],
    }

    def __init__(self, root_path: str, ignore_patterns: list[str] | None = None):
        """Initialize CodebaseDiscovery.

        Args:
            root_path: Root directory of the project.
            ignore_patterns: Folder/file patterns to ignore.
        """
        self.root = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or [
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "dist", "build", ".pytest_cache", ".mypy_cache", "*.egg-info",
        ]

    def analyze(self) -> ProjectProfile:
        """Analyze the entire codebase and return a project profile.

        Returns:
            ProjectProfile with detected framework, modules, and metadata.
        """
        # Collect all Python files
        py_files = list(self._find_python_files())

        # Detect framework from imports
        framework = self._detect_framework(py_files)

        # Analyze each module
        modules = []
        for py_file in py_files:
            module_info = self._analyze_module(py_file)
            if module_info:
                modules.append(module_info)

        # Extract global info
        entry_points = self._find_entry_points(modules)
        db_models = self._find_database_models(modules)
        api_endpoints = self._find_api_endpoints(modules)

        # Determine project type
        project_type = self._determine_project_type(framework, modules)

        return ProjectProfile(
            framework=framework,
            project_type=project_type,
            language="python",
            modules=modules,
            entry_points=entry_points,
            database_models=db_models,
            api_endpoints=api_endpoints,
        )

    def _find_python_files(self):
        """Find all Python files, respecting ignore patterns."""
        for py_file in self.root.rglob("*.py"):
            # Check ignore patterns
            skip = False
            for pattern in self.ignore_patterns:
                if pattern in str(py_file):
                    skip = True
                    break
            if not skip:
                yield py_file

    def _detect_framework(self, py_files: list[Path]) -> str:
        """Detect the web framework used in the project."""
        framework_scores: dict[str, int] = {}

        for py_file in py_files[:50]:  # Sample first 50 files
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in content:
                            framework_scores[framework] = framework_scores.get(framework, 0) + 1
            except Exception:
                continue

        if framework_scores:
            return max(framework_scores, key=framework_scores.get)
        return "generic"

    def _analyze_module(self, py_file: Path) -> Optional[ModuleInfo]:
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
            classes=classes,
            functions=functions,
            imports=list(set(imports)),
            keywords=keywords,
            docstring=docstring[:500] if docstring else "",
            size_kb=len(content) / 1024,
            line_count=len(content.splitlines()),
        )

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

    def _determine_project_type(self, framework: str, modules: list[ModuleInfo]) -> str:
        """Determine overall project type."""
        has_api = any(m.module_type == "api" for m in modules)
        has_models = any(m.module_type == "model" for m in modules)

        if framework in ("fastapi", "flask", "django") and has_api:
            return "web_api"
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
