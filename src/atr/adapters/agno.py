"""Agno tool adapter for ATR.

This module provides the AgnoAdapter class for converting Agno tools
to ToolSpecs and filtering them after routing.

Supports the same tool formats as Agno's Agent.tools parameter:
- Toolkit instances (e.g., YFinanceTools)
- Function objects
- Callables (plain functions)
- Dict (OpenAI-style function definitions)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec


# Type alias matching Agno's Agent.tools type
AgnoToolType = Union[Any, Callable, dict]  # Toolkit | Function | Callable | Dict


def _function_to_spec(func: Any) -> ToolSpec:
    """Convert an Agno Function object to ToolSpec."""
    name = getattr(func, "name", None)
    if not name:
        raise AdapterError("Function object missing 'name' attribute")

    description = getattr(func, "description", "") or ""

    # Get parameters if available
    parameters = None
    params = getattr(func, "parameters", None)
    if params is not None:
        if isinstance(params, dict):
            parameters = params
        elif hasattr(params, "model_dump"):
            parameters = params.model_dump()

    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        source="agno",
        metadata={"original_tool": func},
    )


def _callable_to_spec(func: Callable) -> ToolSpec:
    """Convert a plain callable to ToolSpec."""
    name = getattr(func, "__name__", None)
    if not name:
        raise AdapterError("Callable missing '__name__' attribute")

    description = getattr(func, "__doc__", "") or ""

    return ToolSpec(
        name=name,
        description=description,
        parameters=None,
        source="agno",
        metadata={"original_tool": func},
    )


def _dict_to_spec(tool_dict: dict) -> ToolSpec:
    """Convert an OpenAI-style dict to ToolSpec."""
    # Handle wrapped format: {"type": "function", "function": {...}}
    if tool_dict.get("type") == "function":
        func = tool_dict.get("function", {})
    else:
        func = tool_dict

    name = func.get("name")
    if not name:
        raise AdapterError("Dict tool missing 'name' field")

    return ToolSpec(
        name=name,
        description=func.get("description", ""),
        parameters=func.get("parameters"),
        source="agno",
        metadata={"original_tool": tool_dict},
    )


def _normalize_tool(tool: AgnoToolType) -> list[ToolSpec]:
    """
    Normalize any Agno tool type to a list of ToolSpecs.

    Handles: Toolkit, Function, Callable, Dict
    """
    # Check for Toolkit (has functions dict or get_functions method)
    if hasattr(tool, "get_functions"):
        funcs = tool.get_functions()
        return [_function_to_spec(f) for f in funcs.values()]
    elif hasattr(tool, "functions") and isinstance(getattr(tool, "functions"), dict):
        funcs = tool.functions
        return [_function_to_spec(f) for f in funcs.values()]

    # Check for Function object (has name and description attributes)
    if hasattr(tool, "name") and hasattr(tool, "description"):
        return [_function_to_spec(tool)]

    # Check for dict (OpenAI-style)
    if isinstance(tool, dict):
        return [_dict_to_spec(tool)]

    # Plain callable
    if callable(tool):
        return [_callable_to_spec(tool)]

    raise AdapterError(f"Unsupported tool type: {type(tool)}")


def _extract_all_functions(tools: Sequence[AgnoToolType]) -> dict[str, Any]:
    """
    Extract all Function objects from a sequence of tools.

    Returns a dict mapping function name to Function object.
    """
    all_funcs: dict[str, Any] = {}

    for tool in tools:
        # Toolkit
        if hasattr(tool, "get_functions"):
            funcs = tool.get_functions()
            all_funcs.update(funcs)
        elif hasattr(tool, "functions") and isinstance(getattr(tool, "functions"), dict):
            all_funcs.update(tool.functions)
        # Function object
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            all_funcs[tool.name] = tool
        # Dict
        elif isinstance(tool, dict):
            name = tool.get("function", {}).get("name") if tool.get("type") == "function" else tool.get("name")
            if name:
                all_funcs[name] = tool
        # Callable
        elif callable(tool):
            name = getattr(tool, "__name__", None)
            if name:
                all_funcs[name] = tool

    return all_funcs


class AgnoAdapter:
    """
    Adapter for Agno tools.

    Accepts the same tool formats as Agno's Agent.tools parameter:
    - Toolkit instances (e.g., YFinanceTools, MCPTools)
    - Function objects
    - Callables (plain functions)
    - Dict (OpenAI-style function definitions)

    Example:
        ```python
        from atr import ToolRouter
        from atr.adapters.agno import AgnoAdapter, filter_tools
        from atr.llm import OpenRouterLLM

        # Pass toolkit directly - same as you'd pass to Agent
        toolkit = YFinanceTools()

        # Convert and route
        router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
        router.add_tools(AgnoAdapter.to_specs([toolkit]))
        filtered_specs = router.route("What's AAPL's price?")

        # Filter and use directly with Agent
        filtered_tools = filter_tools([toolkit], filtered_specs)
        agent = Agent(model=..., tools=filtered_tools)
        ```
    """

    @staticmethod
    def to_specs(tools: Sequence[AgnoToolType] | dict[str, Any]) -> list[ToolSpec]:
        """
        Convert Agno tools to ToolSpecs.

        Accepts the same formats as Agno's Agent.tools:
        - Sequence of Toolkit, Function, Callable, or Dict
        - Dict[str, Function] (legacy format from toolkit.get_functions())

        Args:
            tools: Agno tools in any supported format.

        Returns:
            List of ToolSpec objects.
        """
        specs = []

        # Handle legacy dict format (from toolkit.get_functions())
        if isinstance(tools, dict):
            for tool in tools.values():
                try:
                    specs.append(_function_to_spec(tool))
                except Exception as e:
                    raise AdapterError(f"Failed to convert Agno tool: {e}") from e
            return specs

        # Handle sequence of tools (new format)
        for tool in tools:
            try:
                specs.extend(_normalize_tool(tool))
            except Exception as e:
                raise AdapterError(f"Failed to convert Agno tool: {e}") from e

        return specs

    @staticmethod
    def filter_tools(
        tools: Sequence[AgnoToolType] | dict[str, Any],
        filtered: ToolCollection,
    ) -> list[Any]:
        """
        Filter Agno tools based on routing results.

        Returns a list that can be passed directly to Agent(tools=...).

        Args:
            tools: Original Agno tools (same formats as to_specs).
            filtered: ToolCollection from routing.

        Returns:
            List of filtered tools (Function objects, callables, or dicts).
        """
        filtered_names = filtered.names

        # Handle legacy dict format
        if isinstance(tools, dict):
            return [t for t in tools.values() if getattr(t, "name", None) in filtered_names]

        # Extract all functions and filter
        all_funcs = _extract_all_functions(tools)
        return [t for name, t in all_funcs.items() if name in filtered_names]


def filter_tools(
    tools: Sequence[AgnoToolType] | dict[str, Any] | Any,
    filtered: ToolCollection,
) -> list[Any]:
    """
    Filter Agno tools by a ToolCollection.

    Convenience function for filtering Agno tools after routing.
    Accepts the same formats as Agent.tools and returns a list
    that can be passed directly to Agent(tools=...).

    Args:
        tools: Agno tools - Toolkit, Function list, or dict.
        filtered: ToolCollection containing the filtered specs.

    Returns:
        Filtered list of tools for Agent(tools=...).

    Example:
        ```python
        from atr.adapters.agno import AgnoAdapter, filter_tools

        toolkit = YFinanceTools()
        router.add_tools(AgnoAdapter.to_specs([toolkit]))
        filtered_specs = router.route("What's AAPL's price?")

        # Filter and pass directly to Agent
        filtered_tools = filter_tools([toolkit], filtered_specs)
        agent = Agent(model=..., tools=filtered_tools)
        ```
    """
    # Handle single toolkit/MCPTools instance (not in a list)
    if not isinstance(tools, (list, tuple, dict)) and hasattr(tools, "functions"):
        tools = [tools]

    return AgnoAdapter.filter_tools(tools, filtered)
