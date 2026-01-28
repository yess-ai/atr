"""Agno integration for ATR.

This module provides thin helpers for using ATR with Agno agents.
The heavy lifting is done by the adapters - this just provides convenience.

Example:
    ```python
    from agno.tools.mcp import MCPTools
    from agno.agent import Agent
    from atr import ToolRouter
    from atr.adapters import AgnoAdapter

    async with MCPTools(command="npx -y @anthropic/mcp-server-filesystem /tmp") as mcp:
        # Convert to specs and create router
        specs = AgnoAdapter.to_specs(mcp.functions)
        router = ToolRouter(llm=OpenRouterLLM())
        router.add_tools(specs)

        # Route and filter
        filtered = router.route("List files")
        filtered_funcs = AgnoAdapter.filter_tools(mcp.functions, filtered)

        # Create agent with filtered tools only
        agent = Agent(model=model, tools=filtered_funcs)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.adapters.agno import AgnoAdapter
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.llm.base import RoutingLLM


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
        from atr.integrations.agno import to_specs

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
):
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
        from atr.integrations.agno import create_router

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
    router,
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
    router,
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
