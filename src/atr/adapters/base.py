"""Base protocol for tool adapters."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from atr.core.tool import ToolCollection, ToolSpec

T = TypeVar("T")


@runtime_checkable
class ToolAdapter(Protocol[T]):
    """
    Protocol for converting framework tools to/from ToolSpec.

    Tool adapters handle the bidirectional conversion between ATR's
    ToolSpec format and framework-specific tool definitions (MCP schemas,
    LangChain BaseTool, OpenAI function definitions, etc.).

    Type Parameters:
        T: The framework's tool type (e.g., MCP Tool, LangChain BaseTool).
    """

    @staticmethod
    def to_specs(tools: list[T]) -> list[ToolSpec]:
        """
        Convert framework tools to ToolSpecs.

        Args:
            tools: List of framework-specific tool objects.

        Returns:
            List of ToolSpec objects.
        """
        ...

    @staticmethod
    def filter_tools(tools: list[T], filtered: ToolCollection) -> list[T]:
        """
        Filter original tools based on a ToolCollection.

        Args:
            tools: Original framework tools.
            filtered: ToolCollection containing the filtered specs.

        Returns:
            Filtered list of framework tools.
        """
        ...


def get_tool_name(tool: Any) -> str | None:
    """
    Extract tool name from various tool object types.

    Args:
        tool: A tool object from any framework.

    Returns:
        The tool name, or None if not found.
    """
    # Try common attribute names
    for attr in ("name", "function_name", "tool_name", "__name__"):
        if hasattr(tool, attr):
            name = getattr(tool, attr)
            if name and isinstance(name, str):
                return name

    # Try dict-like access
    if isinstance(tool, dict):
        for key in ("name", "function_name", "tool_name"):
            if key in tool:
                return str(tool[key])

    return None
