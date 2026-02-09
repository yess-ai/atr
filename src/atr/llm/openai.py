"""OpenAI-compatible LLM adapters for ATR."""

from __future__ import annotations

import os
from typing import Any

from atr.core.exceptions import ConfigurationError, LLMError


class _OpenAICompatibleLLM:
    """Base class for LLMs that use the OpenAI SDK (OpenAI, OpenRouter, etc.)."""

    _default_model: str = ""
    _env_var: str = ""
    _error_prefix: str = ""
    _install_extra: str = ""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model or self._default_model
        self.api_key = api_key or os.environ.get(self._env_var)
        self.base_url = base_url
        self._kwargs = kwargs
        self._client: Any = None
        self._async_client: Any = None

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self.api_key, **self._kwargs}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ConfigurationError(
                f"{self._error_prefix} API key not found. Set {self._env_var} "
                "environment variable or pass api_key parameter."
            )

    def _get_client(self) -> Any:
        """Lazily initialize the sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ConfigurationError(
                    f"{self._error_prefix} adapter requires the 'openai' package. "
                    f"Install with: pip install atr[{self._install_extra}]"
                ) from e
            self._ensure_api_key()
            self._client = OpenAI(**self._client_kwargs())
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily initialize the async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ConfigurationError(
                    f"{self._error_prefix} adapter requires the 'openai' package. "
                    f"Install with: pip install atr[{self._install_extra}]"
                ) from e
            self._ensure_api_key()
            self._async_client = AsyncOpenAI(**self._client_kwargs())
        return self._async_client

    def _build_messages(
        self, prompt: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a completion."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = client.chat.completions.create(
                model=self.model, messages=messages
            )
            if not response.choices:
                raise LLMError(f"{self._error_prefix} returned no choices.")
            return response.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"{self._error_prefix} completion failed: {e}") from e

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Async generate a completion."""
        client = self._get_async_client()
        messages = self._build_messages(prompt, system_prompt)
        try:
            response = await client.chat.completions.create(
                model=self.model, messages=messages
            )
            if not response.choices:
                raise LLMError(f"{self._error_prefix} returned no choices.")
            return response.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"{self._error_prefix} completion failed: {e}") from e


class OpenAILLM(_OpenAICompatibleLLM):
    """
    OpenAI LLM adapter for tool routing.

    Example:
        ```python
        from atr.llm import OpenAILLM

        llm = OpenAILLM()  # defaults to gpt-4o-mini
        llm = OpenAILLM(model="gpt-4o")
        ```
    """

    _default_model = "gpt-4o-mini"
    _env_var = "OPENAI_API_KEY"
    _error_prefix = "OpenAI"
    _install_extra = "openai"

    DEFAULT_MODEL = "gpt-4o-mini"
