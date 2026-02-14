from __future__ import annotations
"""Configuration for AgentHub via environment variables."""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


def load_env_files(verbose: bool = False) -> list[str]:
    """Load .env files from multiple locations.

    This should be called early in application startup to ensure
    environment variables are available before any config is loaded.

    Priority (later overrides earlier):
    1. Home directory ~/.env
    2. Project directory .env (current working directory)

    Args:
        verbose: If True, print which files were loaded to stderr.

    Returns:
        List of paths that were loaded.
    """
    loaded = []

    # Load from home directory first
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env)
        loaded.append(str(home_env))
        if verbose:
            print(f"Loaded .env from: {home_env}", file=sys.stderr)

    # Load from current working directory (project) - this overrides home
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=True)
        loaded.append(str(cwd_env))
        if verbose:
            print(f"Loaded .env from: {cwd_env}", file=sys.stderr)

    return loaded


class AgentHubConfig(BaseSettings):
    """Configuration via environment variables.

    All settings can be overridden via environment variables with the
    AGENTHUB_ prefix. For example: AGENTHUB_DEFAULT_MODEL=claude-opus-4-20250514
    """

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None  # For embeddings if needed

    # Model defaults
    default_model: str = "claude-opus-4-20250514"
    opus_model: str = "claude-opus-4-20250514"
    haiku_model: str = "claude-haiku-3-5-20241022"

    # Limits
    max_tokens_per_session: int = 100000
    max_context_per_agent: int = 50000
    default_max_tokens: int = 4096

    # Storage
    session_storage_path: str = "./sessions"
    context_cache_path: str = "./context_cache"

    model_config = {
        "env_file": ".env",
        "env_prefix": "AGENTHUB_",
        "extra": "ignore",
    }


# Global config instance (lazy loaded)
_config: Optional[AgentHubConfig] = None


def get_config() -> AgentHubConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = AgentHubConfig()
    return _config


def set_config(config: AgentHubConfig) -> None:
    """Set the global config instance."""
    global _config
    _config = config
