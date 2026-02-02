"""Factory for creating auto-generated agents."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agenthub.agents.base import BaseAgent
from agenthub.auto.analyzer import AgentBoundary
from agenthub.auto.config import AutoAgentConfig
from agenthub.context import ContextBuilder
from agenthub.models import AgentSpec

if TYPE_CHECKING:
    import anthropic


class AutoCodeAgent(BaseAgent):
    """Auto-generated agent for a specific code module.

    This agent is created automatically by the AutoAgentFactory
    and knows about a specific portion of the codebase.
    """

    def __init__(
        self,
        spec: AgentSpec,
        client: "anthropic.Anthropic",
        root_path: str,
        include_patterns: list[str],
    ):
        """Initialize AutoCodeAgent.

        Args:
            spec: Agent specification.
            client: Anthropic client for API calls.
            root_path: Root path this agent covers.
            include_patterns: Glob patterns for files to include.
        """
        super().__init__(spec, client)
        self.root_path = root_path
        self.include_patterns = include_patterns
        self.context_builder = ContextBuilder(root_path)

    def build_context(self) -> str:
        """Build context from the agent's module files."""
        parts: list[str] = []

        # Directory structure
        parts.append("## Module Structure\n```")
        parts.append(self.context_builder.read_directory_structure(max_depth=2))
        parts.append("```")

        # Source files
        parts.append("\n## Source Code")
        parts.append(
            self.context_builder.read_files(
                patterns=self.include_patterns,
                max_size=self.spec.max_context_size,
            )
        )

        return "\n".join(parts)


class AutoAgentFactory:
    """Creates agents from boundaries.

    Takes proposed agent boundaries from the CodebaseAnalyzer
    and creates actual agent instances.

    Example:
        >>> factory = AutoAgentFactory(client, config)
        >>> agent = factory.create_agent(boundary)
        >>> hub.register(agent)
    """

    def __init__(
        self,
        client: "anthropic.Anthropic",
        config: AutoAgentConfig | None = None,
    ):
        """Initialize AutoAgentFactory.

        Args:
            client: Anthropic client for agent API calls.
            config: Auto-agent configuration.
        """
        self.client = client
        self.config = config or AutoAgentConfig()

    def create_agent(self, boundary: AgentBoundary) -> BaseAgent:
        """Create an agent from a boundary definition.

        Args:
            boundary: Agent boundary from analyzer.

        Returns:
            Configured agent instance.
        """
        spec = AgentSpec(
            agent_id=boundary.agent_id,
            name=self._generate_name(boundary),
            description=self._generate_description(boundary),
            context_paths=[
                str(boundary.root_path / p) for p in boundary.include_patterns
            ],
            context_keywords=self._extract_keywords(boundary),
            estimated_tokens=int(boundary.estimated_context_kb * 400),  # ~400 tokens/KB
            max_context_size=int(self.config.max_agent_context_kb * 1024),
            system_prompt=self._generate_system_prompt(boundary),
            metadata={
                "auto_generated": True,
                "tier": "B",
                "root_path": str(boundary.root_path),
                "generated_at": datetime.now().isoformat(),
            },
        )

        return AutoCodeAgent(
            spec=spec,
            client=self.client,
            root_path=str(boundary.root_path),
            include_patterns=boundary.include_patterns,
        )

    def _generate_name(self, boundary: AgentBoundary) -> str:
        """Generate human-readable name: src/api -> 'API Module Expert'.

        Args:
            boundary: Agent boundary.

        Returns:
            Human-readable agent name.
        """
        folder_name = boundary.root_path.name
        # Clean up common names
        name_map = {
            "src": "Source",
            "api": "API",
            "db": "Database",
            "utils": "Utilities",
            "lib": "Library",
            "core": "Core",
            "common": "Common",
            "models": "Models",
            "views": "Views",
            "controllers": "Controllers",
            "services": "Services",
            "handlers": "Handlers",
            "tests": "Tests",
        }
        display_name = name_map.get(folder_name.lower(), folder_name.title())
        return f"{display_name} Module Expert"

    def _generate_description(self, boundary: AgentBoundary) -> str:
        """Generate description from folder contents.

        Args:
            boundary: Agent boundary.

        Returns:
            Agent description.
        """
        return (
            f"Expert on {boundary.root_path.name}/ module "
            f"({boundary.file_count} files, {boundary.estimated_context_kb:.0f}KB)"
        )

    def _extract_keywords(self, boundary: AgentBoundary) -> list[str]:
        """Extract keywords from filenames and folder names.

        Args:
            boundary: Agent boundary.

        Returns:
            List of routing keywords.
        """
        keywords: set[str] = set()

        # Add folder name parts as keywords
        try:
            # Get relative path from project root if possible
            for part in boundary.root_path.parts[-3:]:  # Last 3 parts
                if part and not part.startswith("."):
                    keywords.add(part.lower())
        except Exception:
            keywords.add(boundary.root_path.name.lower())

        # Scan files for names
        for pattern in boundary.include_patterns:
            try:
                for file_path in boundary.root_path.glob(pattern):
                    if file_path.is_file():
                        stem = file_path.stem.lower()
                        # Skip common names
                        if stem not in ("__init__", "index", "main", "app"):
                            keywords.add(stem)
            except Exception:
                pass

        # Limit keywords
        return list(keywords)[:20]

    def _generate_system_prompt(self, boundary: AgentBoundary) -> str:
        """Generate system prompt for code agent.

        Args:
            boundary: Agent boundary.

        Returns:
            System prompt for the agent.
        """
        return f"""You are an expert on the {boundary.root_path.name}/ module.

You know:
- All {boundary.file_count} files in this module
- The patterns and conventions used
- How this module interacts with others

When answering:
- Reference specific files and line numbers
- Explain the "why" behind code decisions
- Suggest improvements when relevant

Your scope is LIMITED to {boundary.root_path.name}/. If asked about code outside
your module, say so and suggest which module might handle it."""
