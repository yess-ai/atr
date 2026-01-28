"""OpenAI tool adapter for ATR.

This module provides the OpenAIAdapter class for converting OpenAI-format
tool definitions to ToolSpecs and filtering them after routing.
"""

from __future__ import annotations

from typing import Any

from atr.core.tool import ToolCollection, ToolSpec


class OpenAIAdapter:
    """
    Adapter for OpenAI function definitions.

    Converts OpenAI function definitions to ToolSpecs and filters
    based on routing results.

    Example:
        ```python
        from atr import ToolRouter
        from atr.adapters.openai import OpenAIAdapter, filter_tools
        from atr.llm import OpenRouterLLM

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

        # Convert and route
        router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
        router.add_tools(OpenAIAdapter.to_specs(tools))
        filtered_specs = router.route("What's the weather?")

        # Filter original tools
        filtered_tools = filter_tools(tools, filtered_specs)
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


def filter_tools(
    tools: list[dict[str, Any]], filtered: ToolCollection
) -> list[dict[str, Any]]:
    """
    Filter OpenAI tools by a ToolCollection.

    Convenience function for filtering OpenAI tools after routing.

    Args:
        tools: Original OpenAI tool definitions.
        filtered: ToolCollection containing the filtered specs.

    Returns:
        Filtered list of OpenAI tool definitions.

    Example:
        ```python
        from atr.adapters.openai import filter_tools

        filtered_specs = router.route("What's the weather?")
        filtered_tools = filter_tools(all_tools, filtered_specs)
        ```
    """
    return OpenAIAdapter.filter_tools(tools, filtered)
