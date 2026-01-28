"""Agno tool adapter for ATR.

This module provides both the AgnoAdapter class for tool conversion
and convenience functions for Agno agent integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.core.router import ToolRouter
    from atr.llm.base import RoutingLLM


class AgnoAdapter:
    """
    Adapter for Agno tools.

    Converts Agno Function objects to ToolSpecs and filters tools
    based on routing results.

    Example:
        ```python
        from agno.tools.yfinance import YFinanceTools
        from atr.adapters import AgnoAdapter

        # Get tools from Agno toolkit
        toolkit = YFinanceTools()
        agno_functions = toolkit.functions  # List of Function objects

        # Convert to ToolSpecs for routing
        specs = AgnoAdapter.to_specs(agno_functions)

        # After routing, filter original tools
        filtered_funcs = AgnoAdapter.filter_tools(agno_functions, filtered_specs)
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

    @staticmethod
    def to_specs_from_toolkit(toolkit: Any) -> list[ToolSpec]:
        """
        Convert an Agno Toolkit to ToolSpecs.

        Args:
            toolkit: An Agno Toolkit instance.

        Returns:
            List of ToolSpec objects.
        """
        functions = getattr(toolkit, "functions", [])
        return AgnoAdapter.to_specs(functions)


# ========================================
# SECTION 2: CONVENIENCE FUNCTIONS
# ========================================


def to_specs(tools: Any, source: str | None = None) -> list[ToolSpec]:
    """
    Convert Agno tools (MCPTools, Toolkit, or Function list) to ToolSpecs.

    This is a convenience wrapper around AgnoAdapter.to_specs() that also
    handles extracting functions from MCPTools/Toolkit instances.

    Args:
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        source: Optional source identifier (e.g., "mcp:filesystem").

    Returns:
        List of ToolSpec objects.

    Example:
        ```python
        from agno.tools.mcp import MCPTools
        from atr.adapters.agno import to_specs

        async with MCPTools(command="...") as mcp:
            specs = to_specs(mcp, source="mcp:filesystem")
        ```
    """
    # Extract functions if given a toolkit/MCPTools instance
    if hasattr(tools, "functions"):
        functions = tools.functions
    else:
        functions = tools

    specs = AgnoAdapter.to_specs(functions)

    # Add source if provided
    if source:
        for spec in specs:
            spec.source = source

    return specs


def filter_tools(tools: Any, filtered: ToolCollection) -> list[Any]:
    """
    Filter Agno tools based on ATR routing result.

    This is a convenience wrapper around AgnoAdapter.filter_tools() that also
    handles extracting functions from MCPTools/Toolkit instances.

    Args:
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        filtered: ToolCollection from router.route().

    Returns:
        Filtered list of Agno Function objects.

    Example:
        ```python
        filtered_collection = router.route("List files")
        filtered_funcs = filter_tools(mcp, filtered_collection)
        agent = Agent(tools=filtered_funcs)
        ```
    """
    # Extract functions if given a toolkit/MCPTools instance
    if hasattr(tools, "functions"):
        functions = tools.functions
    else:
        functions = tools

    return AgnoAdapter.filter_tools(functions, filtered)


def create_router(
    tools: Any,
    llm: RoutingLLM,
    max_tools: int = 10,
    source: str | None = None,
) -> ToolRouter:
    """
    Create a ToolRouter pre-configured with Agno tools.

    Convenience function that combines to_specs() and router creation.

    Args:
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        llm: RoutingLLM for making routing decisions.
        max_tools: Maximum tools to return from routing.
        source: Optional source identifier.

    Returns:
        Configured ToolRouter instance.

    Example:
        ```python
        from atr.adapters.agno import create_router

        async with MCPTools(command="...") as mcp:
            router = create_router(mcp, llm=OpenRouterLLM())
            filtered = router.route("List files")
            filtered_funcs = filter_tools(mcp, filtered)
        ```
    """
    from atr.core.router import ToolRouter

    specs = to_specs(tools, source=source)
    router = ToolRouter(llm=llm, max_tools=max_tools)
    router.add_tools(specs)
    return router


async def route_and_filter(
    router: ToolRouter,
    tools: Any,
    query: str,
) -> list[Any]:
    """
    Route a query and return filtered Agno Function objects.

    Convenience function that combines routing and filtering in one call.

    Args:
        router: ToolRouter instance.
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        query: User's query.

    Returns:
        Filtered list of Agno Function objects.

    Example:
        ```python
        filtered_funcs = await route_and_filter(router, mcp, "List files")
        agent = Agent(tools=filtered_funcs)
        ```
    """
    filtered_collection = await router.aroute(query)
    return filter_tools(tools, filtered_collection)


def route_and_filter_sync(
    router: ToolRouter,
    tools: Any,
    query: str,
) -> list[Any]:
    """
    Sync version of route_and_filter.

    Args:
        router: ToolRouter instance.
        tools: Agno MCPTools, Toolkit, or list of Function objects.
        query: User's query.

    Returns:
        Filtered list of Agno Function objects.
    """
    filtered_collection = router.route(query)
    return filter_tools(tools, filtered_collection)
