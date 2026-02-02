"""Tier A (Business/Domain) Agent Discovery.

This module automatically discovers and loads Tier A agents from project files.
It looks for files matching patterns like `*_agents.py` and extracts agent classes.

Discovery patterns:
1. Files named `*_agents.py` in project root
2. Files named `agents.py` in any subdirectory
3. Classes inheriting from BaseAgent
4. Factory functions like `create_*_hub()` or `register_agents()`
"""

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from agenthub.agents.base import BaseAgent
    from agenthub.hub import AgentHub
    from agenthub.llm.base import LLMClient


@dataclass
class DiscoveredAgent:
    """Information about a discovered Tier A agent."""

    class_name: str
    file_path: str
    module_name: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    llm_provider: str = "claude"  # Detected from imports


@dataclass
class DiscoveredFactory:
    """Information about a discovered hub factory function."""

    function_name: str
    file_path: str
    module_name: str


@dataclass
class TierADiscoveryResult:
    """Result of Tier A agent discovery."""

    agents: list[DiscoveredAgent]
    factories: list[DiscoveredFactory]
    file_paths: list[str]
    errors: list[str] = field(default_factory=list)


class TierADiscovery:
    """Discover Tier A (business/domain) agents in a project.

    This class scans project directories for agent definitions,
    identifying both Claude-based and ChatGPT-based agents.

    Example:
        >>> discovery = TierADiscovery("./my-project")
        >>> result = discovery.scan()
        >>> for agent in result.agents:
        ...     print(f"Found: {agent.class_name} ({agent.llm_provider})")
    """

    # File patterns to search for agent definitions
    AGENT_FILE_PATTERNS = [
        "*_agents.py",
        "*_agent.py",
        "agents.py",
        "business_agents.py",
        "domain_agents.py",
    ]

    def __init__(self, project_root: str):
        """Initialize the discovery.

        Args:
            project_root: Path to the project to scan.
        """
        self.project_root = Path(project_root).resolve()

    def scan(self) -> TierADiscoveryResult:
        """Scan project for Tier A agents.

        Returns:
            TierADiscoveryResult with discovered agents and factories.
        """
        agents: list[DiscoveredAgent] = []
        factories: list[DiscoveredFactory] = []
        file_paths: list[str] = []
        errors: list[str] = []

        # Find all matching files
        for pattern in self.AGENT_FILE_PATTERNS:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip(file_path):
                    continue

                file_paths.append(str(file_path.relative_to(self.project_root)))

                try:
                    file_agents, file_factories = self._analyze_file(file_path)
                    agents.extend(file_agents)
                    factories.extend(file_factories)
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")

        # Also check subdirectories (1 level deep)
        for subdir in self.project_root.iterdir():
            if not subdir.is_dir() or self._should_skip(subdir):
                continue

            for pattern in self.AGENT_FILE_PATTERNS:
                for file_path in subdir.glob(pattern):
                    if self._should_skip(file_path):
                        continue

                    file_paths.append(str(file_path.relative_to(self.project_root)))

                    try:
                        file_agents, file_factories = self._analyze_file(file_path)
                        agents.extend(file_agents)
                        factories.extend(file_factories)
                    except Exception as e:
                        errors.append(f"{file_path}: {str(e)}")

        return TierADiscoveryResult(
            agents=agents,
            factories=factories,
            file_paths=file_paths,
            errors=errors,
        )

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        skip_names = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            "dist",
            "build",
        }
        return any(part in skip_names for part in path.parts)

    def _analyze_file(
        self, file_path: Path
    ) -> tuple[list[DiscoveredAgent], list[DiscoveredFactory]]:
        """Analyze a Python file for agent definitions.

        Args:
            file_path: Path to Python file.

        Returns:
            Tuple of (agents, factories) found in file.
        """
        agents: list[DiscoveredAgent] = []
        factories: list[DiscoveredFactory] = []

        content = file_path.read_text(encoding="utf-8", errors="ignore")

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return agents, factories

        # Detect LLM provider from imports
        llm_provider = self._detect_llm_provider(tree)

        module_name = file_path.stem
        rel_path = str(file_path.relative_to(self.project_root))

        for node in ast.walk(tree):
            # Find agent classes
            if isinstance(node, ast.ClassDef):
                if self._is_agent_class(node):
                    agent = DiscoveredAgent(
                        class_name=node.name,
                        file_path=rel_path,
                        module_name=module_name,
                        keywords=self._extract_keywords(node),
                        description=self._extract_docstring(node),
                        llm_provider=llm_provider,
                    )
                    agents.append(agent)

            # Find factory functions
            elif isinstance(node, ast.FunctionDef):
                if self._is_factory_function(node):
                    factory = DiscoveredFactory(
                        function_name=node.name,
                        file_path=rel_path,
                        module_name=module_name,
                    )
                    factories.append(factory)

        return agents, factories

    def _detect_llm_provider(self, tree: ast.AST) -> str:
        """Detect which LLM provider is used based on imports.

        Args:
            tree: AST of the Python file.

        Returns:
            Provider name ('claude' or 'openai').
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "anthropic" in alias.name:
                        return "claude"
                    if "openai" in alias.name:
                        return "openai"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if "anthropic" in node.module:
                        return "claude"
                    if "openai" in node.module:
                        return "openai"

        return "claude"  # Default

    def _is_agent_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an agent definition.

        Looks for:
        - Inherits from BaseAgent
        - Has 'Agent' in name
        - Has build_context method
        """
        # Check inheritance
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if "Agent" in base_name:
                return True

        # Check name pattern
        if node.name.endswith("Agent"):
            # Also check for build_context method
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "build_context":
                        return True

        return False

    def _is_factory_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a hub factory.

        Looks for patterns like:
        - create_*_hub()
        - register_*_agents()
        - setup_agents()
        """
        name = node.name.lower()

        patterns = [
            "create_" in name and "_hub" in name,
            "register_" in name and "agent" in name,
            name == "setup_agents",
            name == "create_hub",
            name == "get_agents",
        ]

        return any(patterns)

    def _extract_keywords(self, node: ast.ClassDef) -> list[str]:
        """Extract keywords from agent class definition."""
        keywords: list[str] = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                # Look for context_keywords in AgentSpec
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.keyword) and stmt.arg == "context_keywords":
                        if isinstance(stmt.value, ast.List):
                            for elt in stmt.value.elts:
                                if isinstance(elt, ast.Constant):
                                    keywords.append(str(elt.value))

        return keywords[:20]  # Limit

    def _extract_docstring(self, node: ast.ClassDef) -> str:
        """Extract docstring from class."""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                doc = str(node.body[0].value.value)
                # First line only
                return doc.split("\n")[0].strip()
        return ""


def load_tier_a_agents(
    project_root: str,
    llm_client: Optional["LLMClient"] = None,
    hub: Optional["AgentHub"] = None,
) -> list["BaseAgent"]:
    """Discover and load all Tier A agents from a project.

    This function:
    1. Scans for agent files using TierADiscovery
    2. If a factory function is found, calls it to get agents
    3. Otherwise, instantiates discovered agent classes directly

    Args:
        project_root: Path to project.
        llm_client: LLM client to use. If None, creates one based on discovered provider.
        hub: Optional AgentHub instance. If provided and factory returns a hub,
             agents are extracted from it.

    Returns:
        List of instantiated Tier A agents.
    """
    discovery = TierADiscovery(project_root)
    result = discovery.scan()

    agents: list["BaseAgent"] = []

    # If we have factory functions, try to use them
    for factory in result.factories:
        try:
            module_agents = _load_from_factory(
                factory,
                project_root,
                llm_client,
            )
            agents.extend(module_agents)
        except Exception as e:
            print(f"Warning: Failed to load from factory {factory.function_name}: {e}")

    # If no factories worked, try instantiating agent classes directly
    if not agents:
        for agent_info in result.agents:
            try:
                agent = _instantiate_agent(agent_info, project_root, llm_client)
                if agent:
                    agents.append(agent)
            except Exception as e:
                print(f"Warning: Failed to instantiate {agent_info.class_name}: {e}")

    return agents


def _load_from_factory(
    factory: DiscoveredFactory,
    project_root: str,
    llm_client: Optional["LLMClient"] = None,
) -> list["BaseAgent"]:
    """Load agents by calling a factory function."""
    # Load the module
    file_path = Path(project_root) / factory.file_path
    spec = importlib.util.spec_from_file_location(factory.module_name, file_path)
    if not spec or not spec.loader:
        return []

    module = importlib.util.module_from_spec(spec)

    # Add project root to path temporarily
    old_path = sys.path.copy()
    sys.path.insert(0, str(project_root))

    try:
        spec.loader.exec_module(module)

        func = getattr(module, factory.function_name, None)
        if not func:
            return []

        # Try calling with different signatures
        result = None

        # Try: create_hub(project_root, enable_auto_agents=False)
        try:
            result = func(project_root, enable_auto_agents=False)
        except TypeError:
            pass

        # Try: create_hub(project_root)
        if result is None:
            try:
                result = func(project_root)
            except TypeError:
                pass

        # Try: create_hub()
        if result is None:
            try:
                result = func()
            except TypeError:
                pass

        if result is None:
            return []

        # Extract agents from result
        if hasattr(result, "list_agents"):
            # It's an AgentHub - get its Tier A agents
            specs = result.list_agents(tier="A")
            agents = []
            for spec in specs:
                agent = result.get_agent(spec.agent_id)
                if agent:
                    agents.append(agent)
            return agents

        elif isinstance(result, list):
            # It's a list of agents
            return result

        elif hasattr(result, "spec"):
            # Single agent
            return [result]

        return []

    finally:
        sys.path = old_path


def _instantiate_agent(
    agent_info: DiscoveredAgent,
    project_root: str,
    llm_client: Optional["LLMClient"] = None,
) -> Optional["BaseAgent"]:
    """Instantiate an agent class directly."""
    file_path = Path(project_root) / agent_info.file_path
    spec = importlib.util.spec_from_file_location(agent_info.module_name, file_path)
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)

    # Add project root to path temporarily
    old_path = sys.path.copy()
    sys.path.insert(0, str(project_root))

    try:
        spec.loader.exec_module(module)

        agent_class = getattr(module, agent_info.class_name, None)
        if not agent_class:
            return None

        # Create appropriate client
        if llm_client:
            # Use LLM client's underlying client if compatible
            if agent_info.llm_provider == "claude" and hasattr(llm_client, "anthropic_client"):
                return agent_class(llm_client.anthropic_client, project_root)
            elif agent_info.llm_provider == "openai" and hasattr(llm_client, "openai_client"):
                return agent_class(llm_client.openai_client, project_root)

        # Try to create with just project_root
        try:
            return agent_class(project_root=project_root)
        except TypeError:
            pass

        # Create default client based on provider
        if agent_info.llm_provider == "claude":
            import anthropic
            client = anthropic.Anthropic()
            return agent_class(client, project_root)
        elif agent_info.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            return agent_class(client, project_root)

        return None

    finally:
        sys.path = old_path


def get_discovery_summary(project_root: str) -> str:
    """Get a summary of discovered Tier A agents.

    Args:
        project_root: Path to project.

    Returns:
        Human-readable summary string.
    """
    discovery = TierADiscovery(project_root)
    result = discovery.scan()

    lines = [
        f"Tier A Agent Discovery Summary",
        "=" * 40,
        f"Project: {project_root}",
        f"Agent files found: {len(result.file_paths)}",
        "",
    ]

    if result.file_paths:
        lines.append("Files:")
        for path in result.file_paths:
            lines.append(f"  - {path}")
        lines.append("")

    if result.agents:
        lines.append(f"Agent classes found: {len(result.agents)}")
        for agent in result.agents:
            provider_tag = f"[{agent.llm_provider}]"
            lines.append(f"  - {agent.class_name} {provider_tag}")
            if agent.description:
                lines.append(f"    {agent.description}")
        lines.append("")

    if result.factories:
        lines.append(f"Factory functions found: {len(result.factories)}")
        for factory in result.factories:
            lines.append(f"  - {factory.function_name}() in {factory.file_path}")
        lines.append("")

    if result.errors:
        lines.append(f"Errors: {len(result.errors)}")
        for error in result.errors:
            lines.append(f"  - {error}")

    return "\n".join(lines)
