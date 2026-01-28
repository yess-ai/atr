"""LLM adapters for ATR routing."""

from atr.llm.base import RoutingLLM

__all__ = ["RoutingLLM"]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "OpenRouterLLM":
        from atr.llm.openrouter import OpenRouterLLM
        return OpenRouterLLM
    elif name == "OpenAILLM":
        from atr.llm.openai import OpenAILLM
        return OpenAILLM
    elif name == "AnthropicLLM":
        from atr.llm.anthropic import AnthropicLLM
        return AnthropicLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
