"""Base protocol for tool adapters."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from atr.core.tool import ToolCollection, ToolSpec

T = TypeVar("T")


@runtime_checkable
class ToolAdapter(Protocol[T]):
    """
    Protocol for converting framework tools to/from ToolSpec.

    Type Parameters:
        T: The framework's tool type (e.g., MCP Tool, LangChain BaseTool).
    """

    @staticmethod
    def to_specs(tools: list[T]) -> list[ToolSpec]:
        """Convert framework tools to ToolSpecs."""
        ...

    @staticmethod
    def filter_tools(tools: list[T], filtered: ToolCollection) -> list[T]:
        """Filter original tools based on a ToolCollection."""
        ...
