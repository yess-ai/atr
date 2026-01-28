"""LangChain tool adapter for ATR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    pass


class LangChainAdapter:
    """
    Adapter for LangChain tools.

    Converts LangChain BaseTool objects to ToolSpecs and filters tools
    based on routing results.

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
        ```
    """

    @staticmethod
    def to_specs(tools: list[Any]) -> list[ToolSpec]:
        """
        Convert LangChain tools to ToolSpecs.

        Args:
            tools: List of LangChain BaseTool objects.

        Returns:
            List of ToolSpec objects.
        """
        specs = []
        for tool in tools:
            try:
                # LangChain BaseTool has: name, description, args_schema
                name = getattr(tool, "name", None)
                if not name:
                    continue

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

                specs.append(
                    ToolSpec(
                        name=name,
                        description=description,
                        parameters=parameters,
                        source="langchain",
                        metadata={"original_tool": tool},
                    )
                )
            except Exception as e:
                raise AdapterError(f"Failed to convert LangChain tool: {e}") from e

        return specs

    @staticmethod
    def filter_tools(tools: list[Any], filtered: ToolCollection) -> list[Any]:
        """
        Filter LangChain tools based on routing results.

        Args:
            tools: Original LangChain tools.
            filtered: ToolCollection from routing.

        Returns:
            Filtered list of LangChain tools.
        """
        filtered_names = filtered.names
        return [t for t in tools if getattr(t, "name", None) in filtered_names]
