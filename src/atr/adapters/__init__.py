"""Tool adapters for converting framework tools to ToolSpec."""

from atr.adapters.base import ToolAdapter, get_tool_name

__all__ = ["ToolAdapter", "get_tool_name"]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    # MCP Adapter
    if name == "MCPAdapter":
        from atr.adapters.mcp import MCPAdapter

        return MCPAdapter

    # LangChain
    elif name == "LangChainAdapter":
        from atr.adapters.langchain import LangChainAdapter

        return LangChainAdapter

    # Agno
    elif name == "AgnoAdapter":
        from atr.adapters.agno import AgnoAdapter

        return AgnoAdapter

    # OpenAI
    elif name == "OpenAIAdapter":
        from atr.adapters.openai import OpenAIAdapter

        return OpenAIAdapter

    # LiteLLM
    elif name == "LiteLLMAdapter":
        from atr.adapters.litellm import LiteLLMAdapter

        return LiteLLMAdapter
    elif name == "ATRToolRoutingHook":
        from atr.adapters.litellm import ATRToolRoutingHook

        return ATRToolRoutingHook
    elif name == "create_hook":
        from atr.adapters.litellm import create_hook

        return create_hook

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
