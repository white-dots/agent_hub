from __future__ import annotations
"""DAG Teams - Multi-agent collaboration for cross-cutting queries.

This module enables AgentHub to execute complex queries across multiple
specialized agents in topological order based on their dependencies.

Key Components:
    - DAGNode: Represents an agent in the execution graph
    - ExecutionDAG: Manages agent dependencies and execution order
    - ComplexityClassifier: Determines if a query needs team execution
    - DAGTeamExecutor: Orchestrates multi-agent query execution
    - QueryDecomposer: Breaks queries into agent-specific sub-questions
    - ResponseSynthesizer: Combines agent responses into a unified answer

Example:
    >>> from agenthub import AgentHub
    >>> hub = AgentHub()
    >>> hub.enable_auto_agents("./my-project")
    >>> hub.enable_teams()
    >>>
    >>> # Simple query -> single agent
    >>> response = hub.run("What does parse_config do?")
    >>>
    >>> # Complex query -> DAG team
    >>> response = hub.run("How does data flow from API to database?")
"""

from agenthub.teams.classifier import ComplexityClassifier, ComplexityResult
from agenthub.teams.config import TeamConfig
from agenthub.teams.dag import DAGNode, ExecutionDAG
from agenthub.teams.decomposer import QueryDecomposer
from agenthub.teams.executor import DAGTeamExecutor
from agenthub.teams.synthesizer import ResponseSynthesizer

__all__ = [
    "DAGNode",
    "ExecutionDAG",
    "ComplexityClassifier",
    "ComplexityResult",
    "DAGTeamExecutor",
    "QueryDecomposer",
    "ResponseSynthesizer",
    "TeamConfig",
]
