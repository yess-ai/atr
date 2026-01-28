"""OpenAI tool adapter for ATR.

This module provides both the OpenAIAdapter class for tool conversion
and high-level utilities for OpenAI Agents SDK integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.core.router import ToolRouter
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.llm.base import RoutingLLM


# ========================================
# SECTION 1: ADAPTER CLASS (CONVERSION)
# ========================================


class OpenAIAdapter:
    """
    Adapter for OpenAI function definitions.

    Converts OpenAI function definitions to ToolSpecs and filters
    based on routing results.

    Example:
        ```python
        from atr.adapters.openai import OpenAIAdapter

        # OpenAI function definitions
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
        specs = OpenAIAdapter.to_specs(tools)
        ```
    """

    @staticmethod
    def to_specs(tools: list[dict[str, Any]]) -> list[ToolSpec]:
        """
        Convert OpenAI function definitions to ToolSpecs.

        Args:
            tools: List of OpenAI tool definitions (function type).

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
                    source="openai",
                    metadata={"original_tool": tool},
                )
            )

        return specs

    @staticmethod
    def filter_tools(
        tools: list[dict[str, Any]], filtered: ToolCollection
    ) -> list[dict[str, Any]]:
        """
        Filter OpenAI function definitions based on routing results.

        Args:
            tools: Original OpenAI tool definitions.
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
# SECTION 2: HIGH-LEVEL ROUTER CLASS
# ========================================


class OpenAIRouter:
    """
    High-level OpenAI integration for ATR.

    Example:
        ```python
        from atr.adapters.openai import OpenAIRouter
        from atr.llm import OpenRouterLLM

        router = OpenAIRouter(
            llm=OpenRouterLLM(),
            tools=all_tools,
        )

        filtered = router.route("What's the weather?")
        # filtered is a list of OpenAI tool definitions
        ```
    """

    def __init__(
        self,
        llm: RoutingLLM,
        tools: list[dict[str, Any]] | None = None,
        max_tools: int = 10,
    ):
        """
        Initialize the OpenAI router.

        Args:
            llm: RoutingLLM for making routing decisions.
            tools: Initial list of tool definitions (optional).
            max_tools: Maximum tools to return from routing.
        """
        self._router = ToolRouter(llm=llm, max_tools=max_tools)
        self._tools: list[dict[str, Any]] = []

        if tools:
            self.add_tools(tools)

    def add_tools(self, tools: list[dict[str, Any]]) -> None:
        """
        Add OpenAI tool definitions to the router.

        Args:
            tools: List of OpenAI tool definitions.
        """
        self._tools.extend(tools)
        specs = OpenAIAdapter.to_specs(tools)
        self._router.add_tools(specs)

    def route(self, query: str) -> list[dict[str, Any]]:
        """
        Route a query and return filtered tool definitions.

        Args:
            query: The user's query.

        Returns:
            Filtered list of OpenAI tool definitions.
        """
        filtered_specs = self._router.route(query)
        return OpenAIAdapter.filter_tools(self._tools, filtered_specs)

    async def aroute(self, query: str) -> list[dict[str, Any]]:
        """
        Async route a query and return filtered tool definitions.

        Args:
            query: The user's query.

        Returns:
            Filtered list of OpenAI tool definitions.
        """
        filtered_specs = await self._router.aroute(query)
        return OpenAIAdapter.filter_tools(self._tools, filtered_specs)

    @property
    def all_tools(self) -> list[dict[str, Any]]:
        """Get all registered tool definitions."""
        return self._tools.copy()
