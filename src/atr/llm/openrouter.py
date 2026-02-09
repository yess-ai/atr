"""OpenRouter LLM adapter for ATR (DEFAULT)."""

from __future__ import annotations

from typing import Any

from atr.llm.openai import _OpenAICompatibleLLM


class OpenRouterLLM(_OpenAICompatibleLLM):
    """
    OpenRouter LLM adapter for tool routing.

    This is the DEFAULT and RECOMMENDED adapter. OpenRouter provides access
    to many models through a unified API.

    Example:
        ```python
        from atr.llm import OpenRouterLLM

        llm = OpenRouterLLM()  # defaults to anthropic/claude-3-haiku
        llm = OpenRouterLLM(model="openai/gpt-4o-mini")
        ```
    """

    _default_model = "anthropic/claude-3-haiku"
    _env_var = "OPENROUTER_API_KEY"
    _error_prefix = "OpenRouter"
    _install_extra = "openrouter"

    DEFAULT_MODEL = "anthropic/claude-3-haiku"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs: Any,
    ):
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
