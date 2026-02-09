"""LangChain tool adapter for ATR.

This module provides the LangChainAdapter class for tool conversion
and convenience functions for LangGraph integration.

Supports the same tool formats that LangChain agents accept:
- BaseTool objects (including @tool decorated functions)
- StructuredTool objects
- Plain callables (functions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.core.router import ToolRouter


def _tool_to_spec(tool: Any) -> ToolSpec | None:
    """Convert a single LangChain tool to ToolSpec."""
    # Handle BaseTool/StructuredTool (has name attribute)
    name = getattr(tool, "name", None)

    # Handle plain callable
    if not name and callable(tool):
        name = getattr(tool, "__name__", None)
        if name:
            return ToolSpec(
                name=name,
                description=getattr(tool, "__doc__", "") or "",
                parameters=None,
                source="langchain",
                metadata={"original_tool": tool},
            )
        return None

    if not name:
        return None

    description = getattr(tool, "description", "") or ""

    # Get parameters from args_schema if available
    parameters = None
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        # Pydantic model - convert to JSON schema
        if hasattr(args_schema, "model_json_schema"):
            parameters = args_schema.model_json_schema()
        elif hasattr(args_schema, "schema"):
            parameters = args_schema.schema()

    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        source="langchain",
        metadata={"original_tool": tool},
    )


class LangChainAdapter:
    """
    Adapter for LangChain tools.

    Accepts the same tool formats that LangChain agents accept:
    - BaseTool objects (from subclassing or @tool decorator)
    - StructuredTool objects
    - Plain callables (functions)

    Example:
        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from atr.adapters import LangChainAdapter

        # Get tools from LangChain MCP client
        client = MultiServerMCPClient({...})
        lc_tools = await client.get_tools()

        # Convert to ToolSpecs for routing
        specs = LangChainAdapter.to_specs(lc_tools)

        # After routing, filter original tools
        filtered_lc = LangChainAdapter.filter_tools(lc_tools, filtered_specs)
        agent = create_react_agent(model, filtered_lc)
        ```
    """

    @staticmethod
    def to_specs(tools: Sequence[Any]) -> list[ToolSpec]:
        """
        Convert LangChain tools to ToolSpecs.

        Args:
            tools: Sequence of LangChain tools (BaseTool, StructuredTool, or callables).

        Returns:
            List of ToolSpec objects.
        """
        specs = []
        for tool in tools:
            try:
                spec = _tool_to_spec(tool)
                if spec:
                    specs.append(spec)
            except Exception as e:
                raise AdapterError(f"Failed to convert LangChain tool: {e}") from e

        return specs

    @staticmethod
    def filter_tools(tools: Sequence[Any], filtered: ToolCollection) -> list[Any]:
        """
        Filter LangChain tools based on routing results.

        Args:
            tools: Original LangChain tools.
            filtered: ToolCollection from routing.

        Returns:
            Filtered list of LangChain tools.
        """
        filtered_names = filtered.names
        result = []
        for t in tools:
            # Check for name attribute (BaseTool/StructuredTool)
            name = getattr(t, "name", None)
            # Fall back to __name__ for plain callables
            if not name and callable(t):
                name = getattr(t, "__name__", None)
            if name in filtered_names:
                result.append(t)
        return result


def filter_tools(tools: Sequence[Any], filtered: ToolCollection) -> list[Any]:
    """
    Filter LangChain tools by a ToolCollection.

    Convenience function for filtering LangChain tools after routing.

    Args:
        tools: Original LangChain tool objects.
        filtered: ToolCollection containing the filtered specs.

    Returns:
        Filtered list of LangChain tools.

    Example:
        ```python
        from atr.adapters.langchain import filter_tools

        # After routing
        filtered_specs = router.route("Read the README")
        filtered_tools = filter_tools(all_tools, filtered_specs)
        ```
    """
    return LangChainAdapter.filter_tools(tools, filtered)


def create_router_node(
    router: ToolRouter,
    tools: Sequence[Any],
    state_key: str = "tools",
    query_key: str = "query",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a LangGraph node that performs tool routing.

    Returns a node function that can be added to a LangGraph graph.
    The node reads the query from state, routes to select tools,
    and updates state with filtered tools.

    Args:
        router: ToolRouter with tools already added.
        tools: Original LangChain tools to filter.
        state_key: State key to store filtered tools (default: "tools").
        query_key: State key containing the query (default: "query").

    Returns:
        A node function for LangGraph.

    Example:
        ```python
        from langgraph.graph import StateGraph
        from atr.adapters.langchain import create_router_node

        # Create router and add tools
        router = ToolRouter(llm=OpenRouterLLM())
        router.add_tools(LangChainAdapter.to_specs(all_tools))

        # Create routing node
        routing_node = create_router_node(router, all_tools)

        # Add to graph
        graph = StateGraph(AgentState)
        graph.add_node("route_tools", routing_node)
        ```
    """

    def route_node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get(query_key, "")
        if not query:
            # No query, return all tools
            return {state_key: tools}

        filtered_specs = router.route(query)
        filtered_tools = filter_tools(tools, filtered_specs)
        return {state_key: filtered_tools}

    return route_node


def create_async_router_node(
    router: ToolRouter,
    tools: Sequence[Any],
    state_key: str = "tools",
    query_key: str = "query",
) -> Callable[[dict[str, Any]], Any]:
    """
    Create an async LangGraph node that performs tool routing.

    Args:
        router: ToolRouter with tools already added.
        tools: Original LangChain tools to filter.
        state_key: State key to store filtered tools (default: "tools").
        query_key: State key containing the query (default: "query").

    Returns:
        An async node function for LangGraph.
    """

    async def route_node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get(query_key, "")
        if not query:
            return {state_key: tools}

        filtered_specs = await router.aroute(query)
        filtered_tools = filter_tools(tools, filtered_specs)
        return {state_key: filtered_tools}

    return route_node
