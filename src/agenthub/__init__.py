"""AgentHub: Lightweight agent orchestration for context-efficient LLM applications.

AgentHub lets you build and orchestrate specialized AI agents that maintain
focused context. Instead of one monolithic agent processing everything,
compose lightweight agents that know their domain.

Supports multiple LLM providers:
- Claude (Anthropic) - default
- ChatGPT (OpenAI)

Basic usage:
    >>> from agenthub import AgentHub
    >>> hub = AgentHub()
    >>> hub.register(my_agent)
    >>> response = hub.run("How does authentication work?")
    >>> print(response.content)

Auto-discovery (Tier A + Tier B agents) - RECOMMENDED:
    >>> from agenthub.auto import discover_all_agents
    >>>
    >>> # One line to get ALL agents (business + code)
    >>> hub, summary = discover_all_agents("./my-project")
    >>> print(summary)
    >>> response = hub.run("What's the optimal discount?")  # -> Tier A agent
    >>> response = hub.run("How does the API work?")        # -> Tier B agent

Multi-LLM support:
    >>> from agenthub.llm import create_client, ClaudeClient, OpenAIClient
    >>>
    >>> # Auto-detect from environment
    >>> client = create_client()
    >>>
    >>> # Or explicit
    >>> claude = create_client("claude")
    >>> openai = create_client("openai")

CLI commands (Docker-like interface):
    $ agenthub build ./my-project   # Discover Tier A, generate Tier B
    $ agenthub up --port 3001       # Start dashboard
    $ agenthub watch ./my-project   # Watch for file changes
    $ agenthub status               # Show agent status
"""

from agenthub.agents import APIAgent, BaseAgent, CodeAgent, DBAgent
from agenthub.config import AgentHubConfig, get_config, load_env_files, set_config
from agenthub.context import ContextBuilder, FileContext, SQLContext
from agenthub.hub import AgentHub
from agenthub.models import (
    AgentCapability,
    AgentResponse,
    AgentSpec,
    Artifact,
    Message,
    Session,
)
from agenthub.routing import KeywordRouter, LLMRouter, Router, TierAwareRouter

__version__ = "0.1.0"

__all__ = [
    # Core
    "AgentHub",
    # Models
    "AgentSpec",
    "AgentResponse",
    "AgentCapability",
    "Message",
    "Session",
    "Artifact",
    # Agents
    "BaseAgent",
    "CodeAgent",
    "DBAgent",
    "APIAgent",
    # Context
    "ContextBuilder",
    "FileContext",
    "SQLContext",
    # Routing
    "Router",
    "KeywordRouter",
    "LLMRouter",
    "TierAwareRouter",
    # Config
    "AgentHubConfig",
    "get_config",
    "set_config",
    "load_env_files",
    # Version
    "__version__",
]

# Convenience re-export from auto submodule
# Users can do: from agenthub.auto import enable_auto_agents
