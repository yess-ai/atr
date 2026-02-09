"""MCP tool adapter for ATR."""

from __future__ import annotations

from typing import Any

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec


class MCPAdapter:
    """
    Adapter for MCP (Model Context Protocol) tools.

    Example:
        ```python
        from mcp import ClientSession
        from atr.adapters import MCPAdapter

        mcp_tools = await session.list_tools()
        specs = MCPAdapter.to_specs(mcp_tools.tools)
        filtered_mcp = MCPAdapter.filter_tools(mcp_tools.tools, filtered_specs)
        ```
    """

    @staticmethod
    def to_specs(tools: list[Any]) -> list[ToolSpec]:
        """Convert MCP Tool objects to ToolSpecs."""
        specs = []
        for tool in tools:
            try:
                name = getattr(tool, "name", None)
                if not name:
                    continue

                description = getattr(tool, "description", "") or ""
                input_schema = getattr(tool, "inputSchema", None)

                parameters = None
                if input_schema is not None:
                    if hasattr(input_schema, "model_dump"):
                        parameters = input_schema.model_dump()
                    elif isinstance(input_schema, dict):
                        parameters = input_schema

                specs.append(
                    ToolSpec(
                        name=name,
                        description=description,
                        parameters=parameters,
                        source="mcp",
                        metadata={"original_tool": tool},
                    )
                )
            except Exception as e:
                raise AdapterError(f"Failed to convert MCP tool: {e}") from e

        return specs

    @staticmethod
    def filter_tools(tools: list[Any], filtered: ToolCollection) -> list[Any]:
        """Filter MCP tools based on routing results."""
        filtered_names = filtered.names
        return [t for t in tools if getattr(t, "name", None) in filtered_names]

    @staticmethod
    def to_specs_from_dict(tools: list[dict[str, Any]]) -> list[ToolSpec]:
        """Convert MCP tools from dict format to ToolSpecs."""
        specs = []
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue

            specs.append(
                ToolSpec(
                    name=name,
                    description=tool.get("description", ""),
                    parameters=tool.get("inputSchema"),
                    source="mcp",
                    metadata={"original_dict": tool},
                )
            )

        return specs
