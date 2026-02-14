from __future__ import annotations
"""Configuration for DAG team execution.

This module provides configuration options for controlling how
multi-agent queries are classified and executed.
"""

from dataclasses import dataclass


@dataclass
class TeamConfig:
    """Configuration for DAG team execution.

    Attributes:
        complexity_threshold: Minimum confidence score (0.0-1.0) to trigger
            team execution. Lower values = more queries go to DAG teams.
            Default: 0.4

        min_agents_for_team: Minimum number of matched agents to consider
            team execution. Default: 2

        max_agents_per_team: Maximum agents to include in a single team
            execution. Agents beyond this limit are excluded. Default: 6

        max_parallel: Maximum agents to run in parallel per execution layer.
            Higher values can improve latency but use more resources. Default: 4

        agent_timeout_seconds: Timeout per individual agent execution.
            Agents that exceed this are marked as failed. Default: 30.0

        total_timeout_seconds: Total timeout for the entire team execution.
            Execution stops if this is exceeded. Default: 120.0

        decomposer_model: Model to use for query decomposition.
            Should be fast/cheap since decomposition is simple. Default: claude-haiku-4-5-20251001

        synthesizer_model: Model to use for response synthesis.
            Should be capable since synthesis requires reasoning. Default: claude-sonnet-4-20250514

        max_total_tokens: Kill switch - stop execution if total tokens exceed this.
            Prevents runaway costs. Default: 50000

        skip_synthesis_if_single: If only one agent ends up running, skip
            the synthesis step and return its output directly. Default: True

    Example:
        >>> from agenthub.teams.config import TeamConfig
        >>> config = TeamConfig(
        ...     max_parallel=2,  # More conservative parallelism
        ...     complexity_threshold=0.6,  # Trigger teams less often
        ... )
        >>> hub.enable_teams(config=config)
    """

    # Complexity classification
    complexity_threshold: float = 0.4
    min_agents_for_team: int = 2
    max_agents_per_team: int = 6

    # Execution
    max_parallel: int = 4
    agent_timeout_seconds: float = 120.0
    total_timeout_seconds: float = 300.0

    # Models
    decomposer_model: str = "claude-opus-4-20250514"
    synthesizer_model: str = "claude-opus-4-20250514"

    # Cost control
    max_total_tokens: int = 200000
    skip_synthesis_if_single: bool = True
