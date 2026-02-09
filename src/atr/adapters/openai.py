"""OpenAI tool adapter for ATR.

Converts OpenAI-format tool definitions to ToolSpecs and filters them after routing.
Also serves as the base for LiteLLM adapter (same format).
"""

from __future__ import annotations

from typing import Any

from atr.core.tool import ToolCollection, ToolSpec


class OpenAIAdapter:
    """
    Adapter for OpenAI function definitions.

    Example:
        ```python
        from atr import ToolRouter
        from atr.adapters.openai import OpenAIAdapter, filter_tools
        from atr.llm import OpenRouterLLM

        router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
        router.add_tools(OpenAIAdapter.to_specs(tools))
        filtered_specs = router.route("What's the weather?")
        filtered_tools = filter_tools(tools, filtered_specs)
        ```
    """

    _source = "openai"

    @classmethod
    def to_specs(cls, tools: list[dict[str, Any]]) -> list[ToolSpec]:
        """Convert OpenAI function definitions to ToolSpecs."""
        specs = []
        for tool in tools:
            # Handle both wrapped {"type": "function", "function": {...}} and unwrapped
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
                    source=cls._source,
                    metadata={"original_tool": tool},
                )
            )
        return specs

    @staticmethod
    def filter_tools(
        tools: list[dict[str, Any]], filtered: ToolCollection
    ) -> list[dict[str, Any]]:
        """Filter OpenAI function definitions based on routing results."""
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
    """Convenience function: filter OpenAI tools by a ToolCollection."""
    return OpenAIAdapter.filter_tools(tools, filtered)
