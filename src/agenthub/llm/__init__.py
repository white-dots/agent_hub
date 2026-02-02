"""Multi-LLM support for AgentHub.

This module provides an abstraction layer that allows agents to work
with different LLM providers (Claude, ChatGPT, etc.).

Usage:
    >>> from agenthub.llm import create_client
    >>>
    >>> # Create Claude client (default)
    >>> claude = create_client("claude")
    >>>
    >>> # Create OpenAI client
    >>> openai = create_client("openai")
    >>>
    >>> # Auto-detect from environment
    >>> client = create_client()  # Uses ANTHROPIC_API_KEY or OPENAI_API_KEY
"""

import os
from typing import Optional

from agenthub.llm.base import LLMClient, LLMResponse
from agenthub.llm.claude import ClaudeClient

# Optional imports
_openai_available = False
try:
    from agenthub.llm.openai import OpenAIClient

    _openai_available = True
except ImportError:
    OpenAIClient = None  # type: ignore


def create_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: LLM provider name ('claude', 'openai'). If None, auto-detects
                  based on available API keys.
        api_key: Optional API key. If None, reads from environment.

    Returns:
        LLMClient instance for the specified provider.

    Raises:
        ValueError: If provider is unknown or required package not installed.

    Example:
        >>> # Auto-detect provider
        >>> client = create_client()
        >>>
        >>> # Explicit provider
        >>> client = create_client("claude")
        >>> client = create_client("openai")
    """
    # Auto-detect provider from environment
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "claude"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            # Default to Claude
            provider = "claude"

    provider = provider.lower()

    if provider in ("claude", "anthropic"):
        return ClaudeClient(api_key=api_key)

    elif provider in ("openai", "chatgpt", "gpt"):
        if not _openai_available:
            raise ValueError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        return OpenAIClient(api_key=api_key)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: 'claude', 'openai'"
        )


def list_available_providers() -> list[str]:
    """List available LLM providers.

    Returns:
        List of provider names that can be used.
    """
    providers = ["claude"]
    if _openai_available:
        providers.append("openai")
    return providers


__all__ = [
    # Base classes
    "LLMClient",
    "LLMResponse",
    # Implementations
    "ClaudeClient",
    "OpenAIClient",
    # Factory
    "create_client",
    "list_available_providers",
]
