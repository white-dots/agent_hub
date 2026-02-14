from __future__ import annotations
"""Configuration for auto-agent generation."""

from pydantic import BaseModel, Field


class AutoAgentConfig(BaseModel):
    """Configuration for auto-agent generation.

    Controls how the codebase is analyzed and when agents are created.

    Example:
        >>> config = AutoAgentConfig(
        ...     min_folder_size_kb=30,
        ...     max_agent_context_kb=80,
        ... )
        >>> enable_auto_agents(hub, "./project", config=config)
    """

    # Size thresholds
    min_folder_size_kb: int = Field(
        default=50,
        description="Folder must be > this size to warrant an agent",
    )
    max_agent_context_kb: int = Field(
        default=60,  # Reduced from 100KB to save tokens (~24K tokens max)
        description="Split if agent context exceeds this size",
    )

    # Count thresholds
    min_files_per_folder: int = Field(
        default=5,
        description="Folder must have > this many files",
    )
    max_files_per_agent: int = Field(
        default=20,
        description="Split if agent has > this many files",
    )

    # Depth settings
    max_depth: int = Field(
        default=3,
        description="Maximum folder depth to analyze",
    )

    # Agent limits (for dynamic domain detection)
    max_agents: int = Field(
        default=10,
        description="Maximum number of Tier B agents to create",
    )

    # File patterns
    include_extensions: list[str] = Field(
        default_factory=lambda: [".py", ".sql", ".js", ".ts", ".jsx", ".tsx"],
        description="File extensions to include in analysis",
    )

    # Ignore patterns
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            "node_modules",
            "*.pyc",
            ".venv",
            "venv",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.egg-info",
            ".tox",
            "htmlcov",
            "coverage",
        ],
        description="Patterns to ignore during analysis",
    )


class Presets:
    """Pre-configured settings for common scenarios."""

    @staticmethod
    def small_project() -> AutoAgentConfig:
        """For projects < 500KB - minimal splitting.

        Returns:
            Config suitable for small projects.
        """
        return AutoAgentConfig(
            min_folder_size_kb=100,
            max_agent_context_kb=200,
            min_files_per_folder=10,
            max_depth=2,
        )

    @staticmethod
    def medium_project() -> AutoAgentConfig:
        """For projects 500KB - 5MB - balanced.

        Returns:
            Config suitable for medium projects.
        """
        return AutoAgentConfig(
            min_folder_size_kb=50,
            max_agent_context_kb=100,
            min_files_per_folder=5,
            max_depth=3,
        )

    @staticmethod
    def large_project() -> AutoAgentConfig:
        """For projects > 5MB - aggressive splitting.

        Returns:
            Config suitable for large projects.
        """
        return AutoAgentConfig(
            min_folder_size_kb=30,
            max_agent_context_kb=60,
            min_files_per_folder=3,
            max_depth=4,
        )

    @staticmethod
    def monorepo() -> AutoAgentConfig:
        """For monorepos - treat each package as boundary.

        Returns:
            Config suitable for monorepos.
        """
        return AutoAgentConfig(
            min_folder_size_kb=20,
            max_agent_context_kb=80,
            min_files_per_folder=3,
            max_depth=5,
            ignore_patterns=[
                "__pycache__",
                ".git",
                "node_modules",
                "dist",
                "build",
                ".venv",
                "venv",
                "coverage",
                ".pytest_cache",
                ".mypy_cache",
                "*.egg-info",
            ],
        )
