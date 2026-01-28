"""Filter strategies for tool routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from atr.core.tool import ToolCollection

if TYPE_CHECKING:
    from atr.core.tool import ToolSpec
    from atr.llm.base import RoutingLLM


@runtime_checkable
class FilterStrategy(Protocol):
    """
    Protocol for tool filtering strategies.

    FilterStrategy defines the interface for selecting relevant tools
    based on a user query. Implementations can use LLMs, embeddings,
    rule-based systems, or any other approach.
    """

    def filter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """
        Filter tools based on the query.

        Args:
            query: The user's query/request.
            tools: Collection of all available tools.

        Returns:
            Filtered collection containing only relevant tools.
        """
        ...

    async def afilter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """
        Async version of filter.

        Args:
            query: The user's query/request.
            tools: Collection of all available tools.

        Returns:
            Filtered collection containing only relevant tools.
        """
        ...


class BaseFilterStrategy(ABC):
    """Base class for filter strategies with common functionality."""

    @abstractmethod
    def filter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """Filter tools based on the query."""
        ...

    @abstractmethod
    async def afilter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """Async version of filter."""
        ...


class LLMFilterStrategy(BaseFilterStrategy):
    """
    Filter strategy that uses an LLM to select relevant tools.

    This is the default and recommended strategy for ATR. It sends the
    user query and tool summaries to a lightweight LLM (e.g., GPT-4o-mini)
    which returns the names of relevant tools.

    Attributes:
        llm: The LLM to use for routing decisions.
        max_tools: Maximum number of tools to return.
        system_prompt: Custom system prompt for the routing LLM.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a tool selector assistant.
Given a user query and a list of available tools, your job is to select ONLY the tools that are directly relevant to answering the query.
Be conservative - select only tools that will definitely be needed.
Return ONLY the tool names, one per line, no explanations or numbering.
If unsure between similar tools, include both."""

    def __init__(
        self,
        llm: RoutingLLM,
        max_tools: int = 10,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LLM filter strategy.

        Args:
            llm: The LLM to use for routing decisions.
            max_tools: Maximum number of tools to return (default: 10).
            system_prompt: Custom system prompt (optional).
        """
        self.llm = llm
        self.max_tools = max_tools
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _build_prompt(self, query: str, tools: ToolCollection) -> str:
        """Build the routing prompt."""
        tools_summary = tools.to_summaries()
        return f"""User query: "{query}"

Available tools:
{tools_summary}

Select the tools needed for this query. Return only tool names, one per line."""

    def _parse_response(self, response: str, tools: ToolCollection) -> ToolCollection:
        """Parse the LLM response and filter tools."""
        # Parse selected tool names from response
        selected_names = set()
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Handle various formats: "- tool_name", "1. tool_name", "tool_name"
            name = line.lstrip("-").lstrip("0123456789.").strip()
            if name:
                selected_names.add(name)

        # Filter to only valid, selected tools
        valid_selected = selected_names & tools.names

        if not valid_selected:
            # Fallback to all tools if none were selected
            return tools

        filtered = tools.filter_by_names(valid_selected)

        # Apply max_tools limit
        if len(filtered) > self.max_tools:
            filtered = ToolCollection(tools=filtered.tools[: self.max_tools])

        return filtered

    def filter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """
        Filter tools using the LLM.

        Args:
            query: The user's query/request.
            tools: Collection of all available tools.

        Returns:
            Filtered collection containing only relevant tools.
        """
        if not tools.tools:
            return tools

        prompt = self._build_prompt(query, tools)
        response = self.llm.complete(prompt, system_prompt=self.system_prompt)
        return self._parse_response(response, tools)

    async def afilter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """
        Async filter tools using the LLM.

        Args:
            query: The user's query/request.
            tools: Collection of all available tools.

        Returns:
            Filtered collection containing only relevant tools.
        """
        if not tools.tools:
            return tools

        prompt = self._build_prompt(query, tools)
        response = await self.llm.acomplete(prompt, system_prompt=self.system_prompt)
        return self._parse_response(response, tools)


class PassthroughStrategy(BaseFilterStrategy):
    """
    Strategy that returns all tools without filtering.

    Useful for testing or when routing should be disabled.
    """

    def filter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """Return all tools without filtering."""
        return tools

    async def afilter(
        self,
        query: str,
        tools: ToolCollection,
    ) -> ToolCollection:
        """Return all tools without filtering."""
        return tools
