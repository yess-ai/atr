"""OpenRouter LLM adapter for ATR (DEFAULT)."""

from __future__ import annotations

import os
from typing import Any

from atr.core.exceptions import ConfigurationError, LLMError


class OpenRouterLLM:
    """
    OpenRouter LLM adapter for tool routing.

    This is the DEFAULT and RECOMMENDED LLM adapter for ATR. OpenRouter
    provides access to many models through a unified API, allowing you
    to easily switch between providers.

    Example:
        ```python
        from atr.llm import OpenRouterLLM

        # Use default model (Claude 3 Haiku)
        llm = OpenRouterLLM()

        # Use a specific model
        llm = OpenRouterLLM(model="openai/gpt-4o-mini")
        ```

    Attributes:
        model: The model ID to use (OpenRouter format).
        api_key: OpenRouter API key.
    """

    DEFAULT_MODEL = "anthropic/claude-3-haiku"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs: Any,
    ):
        """
        Initialize the OpenRouter adapter.

        Args:
            model: Model ID in OpenRouter format (default: anthropic/claude-3-haiku).
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            **kwargs: Additional arguments passed to OpenAI client.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        self._kwargs = kwargs
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ConfigurationError(
                    "OpenRouter adapter requires the 'openai' package. "
                    "Install with: pip install atr[openrouter]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self._kwargs,
            )
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily initialize the async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ConfigurationError(
                    "OpenRouter adapter requires the 'openai' package. "
                    "Install with: pip install atr[openrouter]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self._kwargs,
            )
        return self._async_client

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a completion using OpenRouter.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's response text.
        """
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            if not response.choices:
                raise LLMError(
                    f"OpenRouter returned no choices. Response: {response}"
                )
            content = response.choices[0].message.content
            return content if content else ""
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"OpenRouter completion failed: {e}") from e

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Async generate a completion using OpenRouter.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's response text.
        """
        client = self._get_async_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            if not response.choices:
                raise LLMError(
                    f"OpenRouter returned no choices. Response: {response}"
                )
            content = response.choices[0].message.content
            return content if content else ""
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"OpenRouter completion failed: {e}") from e
