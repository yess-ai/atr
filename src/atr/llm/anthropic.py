"""Anthropic LLM adapter for ATR."""

from __future__ import annotations

import os
from typing import Any

from atr.core.exceptions import ConfigurationError, LLMError


class AnthropicLLM:
    """
    Anthropic LLM adapter for tool routing.

    Example:
        ```python
        from atr.llm import AnthropicLLM

        # Use default model (Claude 3 Haiku)
        llm = AnthropicLLM()

        # Use a specific model
        llm = AnthropicLLM(model="claude-3-5-sonnet-20241022")
        ```

    Attributes:
        model: The Anthropic model ID to use.
        api_key: Anthropic API key.
    """

    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic adapter.

        Args:
            model: Anthropic model ID (default: claude-3-haiku-20240307).
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            max_tokens: Maximum tokens in response (default: 1024).
            **kwargs: Additional arguments passed to Anthropic client.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._kwargs = kwargs
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the sync client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ConfigurationError(
                    "Anthropic adapter requires the 'anthropic' package. "
                    "Install with: pip install atr[anthropic]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._client = Anthropic(
                api_key=self.api_key,
                **self._kwargs,
            )
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily initialize the async client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ConfigurationError(
                    "Anthropic adapter requires the 'anthropic' package. "
                    "Install with: pip install atr[anthropic]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                **self._kwargs,
            )
        return self._async_client

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a completion using Anthropic.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's response text.
        """
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = client.messages.create(**kwargs)
            # Extract text from response content
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "".join(text_blocks)
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}") from e

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Async generate a completion using Anthropic.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's response text.
        """
        client = self._get_async_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await client.messages.create(**kwargs)
            # Extract text from response content
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "".join(text_blocks)
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}") from e
