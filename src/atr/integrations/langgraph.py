"""LangGraph integration for ATR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from atr.adapters.langchain import LangChainAdapter
from atr.core.router import ToolRouter
from atr.core.tool import ToolCollection

if TYPE_CHECKING:
    pass


def filter_tools(tools: list[Any], filtered: ToolCollection) -> list[Any]:
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
        from atr.integrations.langgraph import filter_tools

        # After routing
        filtered_specs = router.route("Read the README")
        filtered_tools = filter_tools(all_tools, filtered_specs)
        ```
    """
    return LangChainAdapter.filter_tools(tools, filtered)


def create_router_node(
    router: ToolRouter,
    tools: list[Any],
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
        from atr.integrations.langgraph import create_router_node

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


async def create_async_router_node(
    router: ToolRouter,
    tools: list[Any],
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


class LangGraphRouter:
    """
    High-level LangGraph integration for ATR.

    Combines tool loading, conversion, and routing into a single class.

    Example:
        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from atr.integrations.langgraph import LangGraphRouter
        from atr.llm import OpenRouterLLM

        async with MultiServerMCPClient(server_configs) as client:
            all_tools = await client.get_tools()

            router = LangGraphRouter(
                llm=OpenRouterLLM(),
                tools=all_tools,
            )

            filtered = await router.route("Read the README")
            # filtered is a list of LangChain tools
        ```
    """

    def __init__(
        self,
        llm: Any,
        tools: list[Any] | None = None,
        max_tools: int = 10,
    ):
        """
        Initialize the LangGraph router.

        Args:
            llm: RoutingLLM for making routing decisions.
            tools: Initial list of LangChain tools (optional).
            max_tools: Maximum tools to return from routing.
        """
        self._router = ToolRouter(llm=llm, max_tools=max_tools)
        self._tools: list[Any] = []

        if tools:
            self.add_tools(tools)

    def add_tools(self, tools: list[Any]) -> None:
        """
        Add LangChain tools to the router.

        Args:
            tools: List of LangChain BaseTool objects.
        """
        self._tools.extend(tools)
        specs = LangChainAdapter.to_specs(tools)
        self._router.add_tools(specs)

    def route(self, query: str) -> list[Any]:
        """
        Route a query and return filtered LangChain tools.

        Args:
            query: The user's query.

        Returns:
            Filtered list of LangChain tools.
        """
        filtered_specs = self._router.route(query)
        return LangChainAdapter.filter_tools(self._tools, filtered_specs)

    async def aroute(self, query: str) -> list[Any]:
        """
        Async route a query and return filtered LangChain tools.

        Args:
            query: The user's query.

        Returns:
            Filtered list of LangChain tools.
        """
        filtered_specs = await self._router.aroute(query)
        return LangChainAdapter.filter_tools(self._tools, filtered_specs)

    @property
    def all_tools(self) -> list[Any]:
        """Get all registered LangChain tools."""
        return self._tools.copy()

    def create_node(
        self,
        state_key: str = "tools",
        query_key: str = "query",
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Create a LangGraph routing node.

        Args:
            state_key: State key to store filtered tools.
            query_key: State key containing the query.

        Returns:
            A node function for LangGraph.
        """
        return create_router_node(
            self._router,
            self._tools,
            state_key=state_key,
            query_key=query_key,
        )
