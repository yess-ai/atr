"""OpenAI Agents SDK integration for ATR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.core.router import ToolRouter
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    from atr.llm.base import RoutingLLM


class OpenAIAgentsAdapter:
    """
    Adapter for OpenAI Agents SDK function definitions.

    Converts OpenAI function definitions to ToolSpecs and filters
    based on routing results.

    Example:
        ```python
        from atr.integrations.openai_agents import OpenAIAgentsAdapter

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
        specs = OpenAIAgentsAdapter.to_specs(tools)
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
                    source="openai_agents",
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


class FilteredRunner:
    """
    Wrapper for OpenAI Agents SDK Runner with pre-routing.

    FilteredRunner automatically routes queries to filter tools
    before passing them to the agent.

    Example:
        ```python
        from openai_agents import Agent, Runner
        from atr.integrations.openai_agents import FilteredRunner
        from atr.llm import OpenRouterLLM

        # Create agent with all tools
        agent = Agent(
            name="assistant",
            instructions="You are a helpful assistant",
            tools=all_tools,
        )

        # Create router
        router = ToolRouter(llm=OpenRouterLLM())
        router.add_tools(OpenAIAgentsAdapter.to_specs(all_tools))

        # Create filtered runner
        runner = FilteredRunner(agent, router, all_tools)

        # Run with automatic routing
        result = await runner.run("What's the weather in NYC?")
        ```
    """

    def __init__(
        self,
        agent: Any,
        router: ToolRouter,
        tools: list[dict[str, Any]],
    ):
        """
        Initialize the filtered runner.

        Args:
            agent: OpenAI Agents SDK Agent instance.
            router: ToolRouter with tools added.
            tools: All tool definitions to filter from.
        """
        self._agent = agent
        self._router = router
        self._tools = tools

    def _create_filtered_agent(self, filtered_tools: list[dict[str, Any]]) -> Any:
        """Create a copy of the agent with filtered tools."""
        # Create a new agent with filtered tools
        # Note: This depends on the OpenAI Agents SDK API
        try:
            from openai_agents import Agent

            return Agent(
                name=self._agent.name,
                instructions=self._agent.instructions,
                tools=filtered_tools,
                model=getattr(self._agent, "model", None),
            )
        except ImportError:
            # Fallback: return original agent
            return self._agent

    async def run(self, query: str, **kwargs: Any) -> Any:
        """
        Run the agent with filtered tools.

        Args:
            query: The user's query.
            **kwargs: Additional arguments passed to Runner.run().

        Returns:
            The agent's response.
        """
        try:
            from openai_agents import Runner
        except ImportError as e:
            from atr.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "OpenAI Agents integration requires the 'openai-agents' package. "
                "Install with: pip install atr[openai-agents]"
            ) from e

        # Route to get filtered tools
        filtered_specs = await self._router.aroute(query)
        filtered_tools = OpenAIAgentsAdapter.filter_tools(self._tools, filtered_specs)

        # Create agent with filtered tools
        filtered_agent = self._create_filtered_agent(filtered_tools)

        # Run the agent
        return await Runner.run(filtered_agent, query, **kwargs)


class OpenAIAgentsRouter:
    """
    High-level OpenAI Agents SDK integration for ATR.

    Example:
        ```python
        from atr.integrations.openai_agents import OpenAIAgentsRouter
        from atr.llm import OpenRouterLLM

        router = OpenAIAgentsRouter(
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
        Initialize the OpenAI Agents router.

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
        specs = OpenAIAgentsAdapter.to_specs(tools)
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
        return OpenAIAgentsAdapter.filter_tools(self._tools, filtered_specs)

    async def aroute(self, query: str) -> list[dict[str, Any]]:
        """
        Async route a query and return filtered tool definitions.

        Args:
            query: The user's query.

        Returns:
            Filtered list of OpenAI tool definitions.
        """
        filtered_specs = await self._router.aroute(query)
        return OpenAIAgentsAdapter.filter_tools(self._tools, filtered_specs)

    @property
    def all_tools(self) -> list[dict[str, Any]]:
        """Get all registered tool definitions."""
        return self._tools.copy()
