"""LiteLLM tool adapter and custom hook for ATR.

This module provides both the LiteLLMAdapter class for tool conversion
and the ATRToolRoutingHook for automatic tool routing in LiteLLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from atr.core.router import ToolRouter
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.llm.base import RoutingLLM

logger = logging.getLogger(__name__)


# ========================================
# SECTION 1: ADAPTER CLASS (CONVERSION)
# ========================================


class LiteLLMAdapter:
    """
    Adapter for LiteLLM tool definitions (OpenAI format).

    Converts LiteLLM/OpenAI function definitions to ToolSpecs and filters
    based on routing results.

    Example:
        ```python
        from atr.adapters.litellm import LiteLLMAdapter

        # LiteLLM uses OpenAI format for tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {...}
                }
            }
        ]

        # Convert to ToolSpecs
        specs = LiteLLMAdapter.to_specs(tools)
        ```
    """

    @staticmethod
    def to_specs(tools: list[dict[str, Any]]) -> list[ToolSpec]:
        """
        Convert LiteLLM/OpenAI function definitions to ToolSpecs.

        Args:
            tools: List of tool definitions (OpenAI function format).

        Returns:
            List of ToolSpec objects.
        """
        specs = []
        for tool in tools:
            # Handle both wrapped and unwrapped formats
            if tool.get("type") == "function":
                func = tool.get("function", {})
            else:
                func = tool

            name = func.get("name")
            if not name:
                continue

            specs.append(
                ToolSpec(
                    name=name,
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                    source="litellm",
                    metadata={"original_tool": tool},
                )
            )

        return specs

    @staticmethod
    def filter_tools(
        tools: list[dict[str, Any]], filtered: ToolCollection
    ) -> list[dict[str, Any]]:
        """
        Filter tool definitions based on routing results.

        Args:
            tools: Original tool definitions.
            filtered: ToolCollection from routing.

        Returns:
            Filtered list of tool definitions.
        """
        filtered_names = filtered.names
        result = []

        for tool in tools:
            if tool.get("type") == "function":
                name = tool.get("function", {}).get("name")
            else:
                name = tool.get("name")

            if name in filtered_names:
                result.append(tool)

        return result


# ========================================
# SECTION 2: CONVENIENCE FUNCTIONS
# ========================================


def filter_tools(
    tools: list[dict[str, Any]], filtered: ToolCollection
) -> list[dict[str, Any]]:
    """
    Filter LiteLLM tools by a ToolCollection.

    Convenience function for filtering LiteLLM tools after routing.

    Args:
        tools: Original LiteLLM tool definitions.
        filtered: ToolCollection containing the filtered specs.

    Returns:
        Filtered list of tool definitions.

    Example:
        ```python
        from atr.adapters.litellm import filter_tools

        filtered_specs = router.route("What's the weather?")
        filtered_tools = filter_tools(all_tools, filtered_specs)
        ```
    """
    return LiteLLMAdapter.filter_tools(tools, filtered)


def extract_query_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """
    Extract the user query from a list of messages.

    Looks for the last user message in the conversation.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.

    Returns:
        The user's query text, or None if not found.
    """
    if not messages:
        return None

    # Find the last user message
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            # Handle content as list of parts (multimodal)
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
    """
    Create a RoutingLLM instance for the given provider.

    Args:
        provider: LLM provider name (openrouter, openai, anthropic).
        model: Optional model name.

    Returns:
        RoutingLLM instance.

    Raises:
        ConfigurationError: If provider is invalid or dependencies missing.
    """
    from atr.core.exceptions import ConfigurationError

    if provider == "openrouter":
        from atr.llm import OpenRouterLLM

        return OpenRouterLLM(model=model) if model else OpenRouterLLM()
    elif provider == "openai":
        from atr.llm import OpenAILLM

        return OpenAILLM(model=model) if model else OpenAILLM()
    elif provider == "anthropic":
        from atr.llm import AnthropicLLM

        return AnthropicLLM(model=model) if model else AnthropicLLM()
    else:
        raise ConfigurationError(
            f"Invalid LLM provider: {provider}. "
            "Supported providers: openrouter, openai, anthropic"
        )


# ========================================
# SECTION 3: HIGH-LEVEL HOOK CLASS
# ========================================


class ATRToolRoutingHook:
    """
    LiteLLM custom hook for automatic tool routing with ATR.

    This hook intercepts LiteLLM requests and filters tools based on
    the user's query before the request reaches the model.

    Example (proxy_config.yaml):
        ```yaml
        litellm_settings:
          callbacks:
            - atr.adapters.litellm.ATRToolRoutingHook
          atr_config:
            enabled: true
            max_tools: 10
            llm_provider: openrouter
            llm_model: anthropic/claude-3-haiku
        ```

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
        """
        Initialize the ATR tool routing hook.

        Args:
            enabled: Whether routing is enabled.
            max_tools: Maximum number of tools to return after routing.
            min_tools_threshold: Only route if tools >= this count.
            llm_provider: LLM provider for routing (openrouter/openai/anthropic).
            llm_model: Optional model name for the LLM provider.
            llm: Optional pre-configured RoutingLLM instance.
        """
        self.enabled = enabled
        self.max_tools = max_tools
        self.min_tools_threshold = min_tools_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._llm = llm
        self._router: ToolRouter | None = None

    def _get_router(self) -> ToolRouter:
        """Get or create the ToolRouter instance (lazy initialization)."""
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
                # Try getting from litellm_settings
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
        """
        LiteLLM pre-call hook that filters tools based on the user query.

        Args:
            user_api_key_dict: User API key information.
            cache: LiteLLM cache instance.
            data: Request data containing messages and tools.
            call_type: Type of call (completion, embedding, etc.).

        Returns:
            Modified data dict with filtered tools.
        """
        # Load config on first call (allows config from litellm_settings)
        self._load_config_from_litellm()

        # Skip if disabled
        if not self.enabled:
            return data

        # Skip non-completion calls
        if call_type not in ("completion", "acompletion"):
            return data

        # Get tools from request
        tools = data.get("tools")
        if not tools or not isinstance(tools, list):
            return data

        # Skip if below threshold
        if len(tools) < self.min_tools_threshold:
            logger.debug(
                f"ATR: Skipping routing, {len(tools)} tools below threshold "
                f"({self.min_tools_threshold})"
            )
            return data

        # Extract query from messages
        messages = data.get("messages", [])
        query = extract_query_from_messages(messages)
        if not query:
            logger.debug("ATR: No user query found in messages, skipping routing")
            return data

        try:
            # Convert tools to ToolSpecs
            specs = LiteLLMAdapter.to_specs(tools)

            # Get router and add tools
            router = self._get_router()
            router.clear_tools()
            router.add_tools(specs)

            # Route to get filtered tools
            filtered_specs = await router.aroute(query)

            # Filter original tools
            filtered_tools = LiteLLMAdapter.filter_tools(tools, filtered_specs)

            logger.info(
                f"ATR: Filtered tools from {len(tools)} to {len(filtered_tools)} "
                f"for query: {query[:50]}..."
            )

            # Update data with filtered tools
            data["tools"] = filtered_tools

        except Exception as e:
            # Fail open - return original tools if routing fails
            logger.warning(f"ATR: Routing failed, using original tools: {e}")

        return data

    # LiteLLM CustomLogger interface methods (no-op for unused hooks)
    async def async_log_success_event(self, *args: Any, **kwargs: Any) -> None:
        """No-op success logging."""
        pass

    async def async_log_failure_event(self, *args: Any, **kwargs: Any) -> None:
        """No-op failure logging."""
        pass

    def log_success_event(self, *args: Any, **kwargs: Any) -> None:
        """No-op success logging."""
        pass

    def log_failure_event(self, *args: Any, **kwargs: Any) -> None:
        """No-op failure logging."""
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

    This is the recommended way to create a hook for programmatic use.

    Args:
        enabled: Whether routing is enabled.
        max_tools: Maximum number of tools to return after routing.
        min_tools_threshold: Only route if tools >= this count.
        llm_provider: LLM provider for routing (openrouter/openai/anthropic).
        llm_model: Optional model name for the LLM provider.
        llm: Optional pre-configured RoutingLLM instance.

    Returns:
        Configured ATRToolRoutingHook instance.

    Example:
        ```python
        import litellm
        from atr.adapters.litellm import create_hook

        # Create hook with custom configuration
        hook = create_hook(
            llm_provider="openrouter",
            llm_model="anthropic/claude-3-haiku",
            max_tools=5,
        )

        # Add to LiteLLM callbacks
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
