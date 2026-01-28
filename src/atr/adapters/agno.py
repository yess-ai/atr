"""Agno tool adapter for ATR.

This module provides the AgnoAdapter class for converting Agno Function
objects to ToolSpecs and filtering them after routing.
"""

from __future__ import annotations

from typing import Any

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec


class AgnoAdapter:
    """
    Adapter for Agno tools.

    Converts Agno Function objects to ToolSpecs and filters tools
    based on routing results.

    Example:
        ```python
        from atr import ToolRouter
        from atr.adapters.agno import AgnoAdapter, filter_tools
        from atr.llm import OpenRouterLLM

        # Get tools from Agno toolkit
        toolkit = YFinanceTools()
        agno_functions = toolkit.functions

        # Convert and route
        router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
        router.add_tools(AgnoAdapter.to_specs(agno_functions))
        filtered_specs = router.route("What's AAPL's price?")

        # Filter original tools
        filtered_funcs = filter_tools(agno_functions, filtered_specs)
        ```
    """

    @staticmethod
    def to_specs(tools: list[Any]) -> list[ToolSpec]:
        """
        Convert Agno Function objects to ToolSpecs.

        Args:
            tools: List of Agno Function objects.

        Returns:
            List of ToolSpec objects.
        """
        specs = []
        for tool in tools:
            try:
                # Agno Function has: name, description, parameters
                name = getattr(tool, "name", None)
                if not name:
                    continue

                description = getattr(tool, "description", "") or ""

                # Get parameters if available
                parameters = None
                params = getattr(tool, "parameters", None)
                if params is not None:
                    if isinstance(params, dict):
                        parameters = params
                    elif hasattr(params, "model_dump"):
                        parameters = params.model_dump()

                specs.append(
                    ToolSpec(
                        name=name,
                        description=description,
                        parameters=parameters,
                        source="agno",
                        metadata={"original_tool": tool},
                    )
                )
            except Exception as e:
                raise AdapterError(f"Failed to convert Agno tool: {e}") from e

        return specs

    @staticmethod
    def filter_tools(tools: list[Any], filtered: ToolCollection) -> list[Any]:
        """
        Filter Agno tools based on routing results.

        Args:
            tools: Original Agno Function objects.
            filtered: ToolCollection from routing.

        Returns:
            Filtered list of Agno Function objects.
        """
        filtered_names = filtered.names
        return [t for t in tools if getattr(t, "name", None) in filtered_names]


def filter_tools(tools: Any, filtered: ToolCollection) -> list[Any]:
    """
    Filter Agno tools by a ToolCollection.

    Convenience function for filtering Agno tools after routing.
    Handles both raw Function lists and MCPTools/Toolkit instances.

    Args:
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        filtered: ToolCollection containing the filtered specs.

    Returns:
        Filtered list of Agno Function objects.

    Example:
        ```python
        from atr.adapters.agno import filter_tools

        filtered_specs = router.route("What's AAPL's price?")
        filtered_funcs = filter_tools(toolkit.functions, filtered_specs)
        ```
    """
    # Extract functions if given a toolkit/MCPTools instance
    if hasattr(tools, "functions"):
        functions = tools.functions
    else:
        functions = tools

    return AgnoAdapter.filter_tools(functions, filtered)
