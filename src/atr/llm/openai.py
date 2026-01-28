"""OpenAI LLM adapter for ATR."""

from __future__ import annotations

import os
from typing import Any

from atr.core.exceptions import ConfigurationError, LLMError


class OpenAILLM:
    """
    OpenAI LLM adapter for tool routing.

    Example:
        ```python
        from atr.llm import OpenAILLM

        # Use default model (GPT-4o-mini)
        llm = OpenAILLM()

        # Use a specific model
        llm = OpenAILLM(model="gpt-4o")
        ```

    Attributes:
        model: The OpenAI model ID to use.
        api_key: OpenAI API key.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI adapter.

        Args:
            model: OpenAI model ID (default: gpt-4o-mini).
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            **kwargs: Additional arguments passed to OpenAI client.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
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
                    "OpenAI adapter requires the 'openai' package. "
                    "Install with: pip install atr[openai]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._client = OpenAI(
                api_key=self.api_key,
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
                    "OpenAI adapter requires the 'openai' package. "
                    "Install with: pip install atr[openai]"
                ) from e

            if not self.api_key:
                raise ConfigurationError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                **self._kwargs,
            )
        return self._async_client

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a completion using OpenAI.

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
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}") from e

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Async generate a completion using OpenAI.

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
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}") from e
