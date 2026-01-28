"""Tool adapters for converting framework tools to ToolSpec."""

from atr.adapters.base import ToolAdapter, get_tool_name

__all__ = ["ToolAdapter", "get_tool_name"]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "MCPAdapter":
        from atr.adapters.mcp import MCPAdapter

        return MCPAdapter
    elif name == "LangChainAdapter":
        from atr.adapters.langchain import LangChainAdapter

        return LangChainAdapter
    elif name == "AgnoAdapter":
        from atr.adapters.agno import AgnoAdapter

        return AgnoAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
