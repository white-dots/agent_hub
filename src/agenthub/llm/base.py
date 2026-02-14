from __future__ import annotations
"""Abstract LLM client interface.

This module provides an abstraction layer for different LLM providers,
allowing AgentHub to work with both Claude and ChatGPT-based agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    raw_response: Optional[object] = None  # Original provider response


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Implement this interface to add support for new LLM providers.

    Example:
        >>> class MyLLMClient(LLMClient):
        ...     def chat(self, messages, system, **kwargs):
        ...         # Call your LLM API
        ...         return LLMResponse(content="Hello", ...)
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'claude', 'openai')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt.
            model: Optional model override.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and token usage.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name}>"
