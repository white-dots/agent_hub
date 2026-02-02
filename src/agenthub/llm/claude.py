"""Claude/Anthropic LLM client implementation."""

from typing import Optional

from agenthub.llm.base import LLMClient, LLMResponse


class ClaudeClient(LLMClient):
    """LLM client for Claude (Anthropic API).

    Example:
        >>> from agenthub.llm import ClaudeClient
        >>> client = ClaudeClient()
        >>> response = client.chat(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     system="You are a helpful assistant."
        ... )
        >>> print(response.content)
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    @property
    def anthropic_client(self):
        """Get the underlying Anthropic client for direct access if needed."""
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat request to Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt.
            model: Optional model override.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and token usage.
        """
        response = self._client.messages.create(
            model=model or self.default_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            raw_response=response,
        )
