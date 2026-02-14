from __future__ import annotations
"""OpenAI/ChatGPT LLM client implementation."""

from typing import Optional

from agenthub.llm.base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """LLM client for ChatGPT (OpenAI API).

    Example:
        >>> from agenthub.llm import OpenAIClient
        >>> client = OpenAIClient()
        >>> response = client.chat(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     system="You are a helpful assistant."
        ... )
        >>> print(response.content)

    Note:
        Requires openai package: pip install openai
    """

    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            organization: Optional organization ID.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization

        self._client = OpenAI(**kwargs) if kwargs else OpenAI()

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def openai_client(self):
        """Get the underlying OpenAI client for direct access if needed."""
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat request to ChatGPT.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt.
            model: Optional model override.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and token usage.
        """
        # OpenAI uses system message in the messages array
        full_messages = [{"role": "system", "content": system}] + messages

        response = self._client.chat.completions.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            messages=full_messages,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            raw_response=response,
        )
