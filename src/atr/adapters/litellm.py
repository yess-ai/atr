"""LiteLLM tool adapter and custom hook for ATR.

Provides LiteLLMAdapter for tool conversion and ATRToolRoutingHook
for automatic tool routing in LiteLLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from atr.adapters.openai import OpenAIAdapter
from atr.core.router import ToolRouter
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.llm.base import RoutingLLM

logger = logging.getLogger(__name__)


class LiteLLMAdapter(OpenAIAdapter):
    """
    Adapter for LiteLLM tool definitions (OpenAI format).

    LiteLLM uses the same tool format as OpenAI, so this inherits
    all conversion logic from OpenAIAdapter with source="litellm".

    Example:
        ```python
        from atr.adapters.litellm import LiteLLMAdapter

        specs = LiteLLMAdapter.to_specs(tools)
        ```
    """

    _source = "litellm"


def filter_tools(
    tools: list[dict[str, Any]], filtered: ToolCollection
) -> list[dict[str, Any]]:
    """Convenience function: filter LiteLLM tools by a ToolCollection."""
    return LiteLLMAdapter.filter_tools(tools, filtered)


def extract_query_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Extract the last user query from a list of messages."""
    if not messages:
        return None

    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                if text_parts:
                    return " ".join(text_parts)
            return None

    return None


def _create_llm(provider: str, model: str | None = None) -> RoutingLLM:
    """Create a RoutingLLM instance for the given provider."""
    from atr.core.exceptions import ConfigurationError

    if provider == "openrouter":
        from atr.llm import OpenRouterLLM

        return OpenRouterLLM(model=model)
    elif provider == "openai":
        from atr.llm import OpenAILLM

        return OpenAILLM(model=model)
    elif provider == "anthropic":
        from atr.llm import AnthropicLLM

        return AnthropicLLM(model=model)
    else:
        raise ConfigurationError(
            f"Invalid LLM provider: {provider}. "
            "Supported: openrouter, openai, anthropic"
        )


class ATRToolRoutingHook:
    """
    LiteLLM custom hook for automatic tool routing with ATR.

    Intercepts LiteLLM requests and filters tools based on the user's
    query before the request reaches the model.

    Example (programmatic):
        ```python
        import litellm
        from atr.adapters.litellm import create_hook

        hook = create_hook(llm_provider="openrouter", max_tools=5)
        litellm.callbacks = [hook]
        ```
    """

    def __init__(
        self,
        enabled: bool = True,
        max_tools: int = 10,
        min_tools_threshold: int = 5,
        llm_provider: str = "openrouter",
        llm_model: str | None = None,
        llm: RoutingLLM | None = None,
    ):
        self.enabled = enabled
        self.max_tools = max_tools
        self.min_tools_threshold = min_tools_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm = llm
        self._router: ToolRouter | None = None

    def _get_router(self) -> ToolRouter:
        """Get or create the ToolRouter (lazy init)."""
        if self._router is None:
            if self._llm is None:
                self._llm = _create_llm(self.llm_provider, self.llm_model)
            self._router = ToolRouter(llm=self._llm, max_tools=self.max_tools)
        return self._router

    def _load_config_from_litellm(self) -> None:
        """Load configuration from litellm_settings.atr_config if available."""
        try:
            import litellm

            atr_config = getattr(litellm, "atr_config", None)
            if atr_config is None:
                litellm_settings = getattr(litellm, "litellm_settings", {})
                if isinstance(litellm_settings, dict):
                    atr_config = litellm_settings.get("atr_config")

            if atr_config and isinstance(atr_config, dict):
                self.enabled = atr_config.get("enabled", self.enabled)
                self.max_tools = atr_config.get("max_tools", self.max_tools)
                self.min_tools_threshold = atr_config.get(
                    "min_tools_threshold", self.min_tools_threshold
                )
                self.llm_provider = atr_config.get("llm_provider", self.llm_provider)
                self.llm_model = atr_config.get("llm_model", self.llm_model)
                # Reset router to pick up new config
                self._router = None
                self._llm = None
        except ImportError:
            pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict[str, Any],
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """LiteLLM pre-call hook that filters tools based on the user query."""
        self._load_config_from_litellm()

        if not self.enabled:
            return data

        if call_type not in ("completion", "acompletion"):
            return data

        tools = data.get("tools")
        if not tools or not isinstance(tools, list):
            return data

        if len(tools) < self.min_tools_threshold:
            logger.debug(
                f"ATR: Skipping routing, {len(tools)} tools below threshold "
                f"({self.min_tools_threshold})"
            )
            return data

        messages = data.get("messages", [])
        query = extract_query_from_messages(messages)
        if not query:
            logger.debug("ATR: No user query found in messages, skipping routing")
            return data

        try:
            specs = LiteLLMAdapter.to_specs(tools)

            router = self._get_router()
            router.clear_tools()
            router.add_tools(specs)

            filtered_specs = await router.aroute(query)
            filtered_tools = LiteLLMAdapter.filter_tools(tools, filtered_specs)

            logger.info(
                f"ATR: Filtered tools from {len(tools)} to {len(filtered_tools)} "
                f"for query: {query[:50]}..."
            )
            data["tools"] = filtered_tools

        except Exception as e:
            # Fail open - return original tools if routing fails
            logger.warning(f"ATR: Routing failed, using original tools: {e}")

        return data

    # LiteLLM CustomLogger interface methods (no-op for unused hooks)
    async def async_log_success_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def async_log_failure_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_success_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_failure_event(self, *args: Any, **kwargs: Any) -> None:
        pass


def create_hook(
    enabled: bool = True,
    max_tools: int = 10,
    min_tools_threshold: int = 5,
    llm_provider: str = "openrouter",
    llm_model: str | None = None,
    llm: RoutingLLM | None = None,
) -> ATRToolRoutingHook:
    """
    Factory function to create an ATR tool routing hook.

    Example:
        ```python
        import litellm
        from atr.adapters.litellm import create_hook

        hook = create_hook(llm_provider="openrouter", max_tools=5)
        litellm.callbacks = [hook]
        ```
    """
    return ATRToolRoutingHook(
        enabled=enabled,
        max_tools=max_tools,
        min_tools_threshold=min_tools_threshold,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm=llm,
    )
