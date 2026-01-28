"""Main ToolRouter class for ATR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atr.core.strategies import LLMFilterStrategy, PassthroughStrategy
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.core.strategies import FilterStrategy
    from atr.llm.base import RoutingLLM


class ToolRouter:
    """
    Main router class for Adaptive Tool Routing.

    ToolRouter manages a collection of tools and routes user queries to
    select only the relevant tools. This reduces context size and improves
    tool selection accuracy.

    Example:
        ```python
        from atr import ToolRouter, ToolSpec
        from atr.llm import OpenRouterLLM

        router = ToolRouter(llm=OpenRouterLLM())
        router.add_tools([
            ToolSpec(name="get_price", description="Get stock price"),
            ToolSpec(name="get_news", description="Get company news"),
        ])

        filtered = router.route("What is AAPL's price?")
        print(filtered.names)  # {'get_price'}
        ```

    Attributes:
        tools: Collection of all registered tools.
        strategy: The filtering strategy to use.
    """

    def __init__(
        self,
        llm: RoutingLLM | None = None,
        strategy: FilterStrategy | None = None,
        max_tools: int = 10,
    ):
        """
        Initialize the ToolRouter.

        Args:
            llm: LLM to use for routing (creates LLMFilterStrategy).
            strategy: Custom filter strategy (overrides llm parameter).
            max_tools: Maximum tools to return (used with LLMFilterStrategy).

        Note:
            If neither llm nor strategy is provided, PassthroughStrategy is used.
        """
        self._tools = ToolCollection()

        if strategy is not None:
            self._strategy = strategy
        elif llm is not None:
            self._strategy = LLMFilterStrategy(llm=llm, max_tools=max_tools)
        else:
            self._strategy = PassthroughStrategy()

    @property
    def tools(self) -> ToolCollection:
        """Get all registered tools."""
        return self._tools

    @property
    def strategy(self) -> FilterStrategy:
        """Get the current filter strategy."""
        return self._strategy

    def add_tool(self, tool: ToolSpec) -> None:
        """
        Add a single tool to the router.

        Args:
            tool: The tool specification to add.
        """
        self._tools.add(tool)

    def add_tools(self, tools: list[ToolSpec]) -> None:
        """
        Add multiple tools to the router.

        Args:
            tools: List of tool specifications to add.
        """
        self._tools.extend(tools)

    def clear_tools(self) -> None:
        """Remove all tools from the router."""
        self._tools = ToolCollection()

    def route(self, query: str) -> ToolCollection:
        """
        Route a query to select relevant tools.

        Args:
            query: The user's query/request.

        Returns:
            ToolCollection containing only the relevant tools.
        """
        return self._strategy.filter(query, self._tools)

    async def aroute(self, query: str) -> ToolCollection:
        """
        Async route a query to select relevant tools.

        Args:
            query: The user's query/request.

        Returns:
            ToolCollection containing only the relevant tools.
        """
        return await self._strategy.afilter(query, self._tools)

    def set_strategy(self, strategy: FilterStrategy) -> None:
        """
        Set a new filter strategy.

        Args:
            strategy: The new strategy to use.
        """
        self._strategy = strategy
