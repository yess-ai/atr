"""Agno tool adapter for ATR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec

if TYPE_CHECKING:
    pass


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
