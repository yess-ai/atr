"""Tests for Agno adapter."""

from __future__ import annotations

import pytest

from atr.adapters.agno import AgnoAdapter, filter_tools, _normalize_tool
from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec


class MockAgnoFunction:
    """Mock Agno Function object."""

    def __init__(self, name: str, description: str = "", parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters


class MockAgnoToolkit:
    """Mock Agno Toolkit with get_functions."""

    def __init__(self, functions: dict):
        self._functions = functions

    def get_functions(self) -> dict:
        return self._functions


class MockAgnoToolkitLegacy:
    """Mock Agno Toolkit with functions dict (no get_functions)."""

    def __init__(self, functions: dict):
        self.functions = functions


class TestAgnoAdapterToSpecs:
    """Tests for AgnoAdapter.to_specs."""

    def test_to_specs_function_objects(self):
        """Convert Function objects."""
        funcs = [
            MockAgnoFunction("get_price", "Get stock price"),
            MockAgnoFunction("get_news", "Get news"),
        ]

        specs = AgnoAdapter.to_specs(funcs)

        assert len(specs) == 2
        assert specs[0].name == "get_price"
        assert specs[0].source == "agno"

    def test_to_specs_toolkit_get_functions(self):
        """Convert Toolkit with get_functions."""
        toolkit = MockAgnoToolkit({
            "get_price": MockAgnoFunction("get_price", "Get price"),
            "get_news": MockAgnoFunction("get_news", "Get news"),
        })

        specs = AgnoAdapter.to_specs([toolkit])

        assert len(specs) == 2
        names = {s.name for s in specs}
        assert "get_price" in names
        assert "get_news" in names

    def test_to_specs_toolkit_functions_dict(self):
        """Convert Toolkit with functions dict attribute."""
        toolkit = MockAgnoToolkitLegacy({
            "get_price": MockAgnoFunction("get_price", "Get price"),
        })

        specs = AgnoAdapter.to_specs([toolkit])

        assert len(specs) == 1
        assert specs[0].name == "get_price"

    def test_to_specs_callable(self):
        """Convert plain callables."""

        def my_func():
            """A function."""
            pass

        specs = AgnoAdapter.to_specs([my_func])

        assert len(specs) == 1
        assert specs[0].name == "my_func"
        assert specs[0].description == "A function."

    def test_to_specs_dict_wrapped(self):
        """Convert wrapped dict format."""
        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "Test"},
            }
        ]

        specs = AgnoAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "test"

    def test_to_specs_dict_unwrapped(self):
        """Convert unwrapped dict format."""
        tools = [{"name": "test", "description": "Test"}]

        specs = AgnoAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "test"

    def test_to_specs_legacy_dict_format(self):
        """Convert legacy dict format from toolkit.get_functions()."""
        funcs = {
            "get_price": MockAgnoFunction("get_price", "Get price"),
            "get_news": MockAgnoFunction("get_news", "Get news"),
        }

        specs = AgnoAdapter.to_specs(funcs)

        assert len(specs) == 2

    def test_to_specs_empty_list(self):
        """Handle empty list."""
        specs = AgnoAdapter.to_specs([])
        assert specs == []

    def test_to_specs_mixed_types(self):
        """Handle mixed tool types."""

        def my_func():
            """Callable func."""
            pass

        tools = [
            MockAgnoFunction("func_tool", "Function tool"),
            my_func,
            {"name": "dict_tool", "description": "Dict tool"},
        ]

        specs = AgnoAdapter.to_specs(tools)

        assert len(specs) == 3
        names = {s.name for s in specs}
        assert "func_tool" in names
        assert "my_func" in names
        assert "dict_tool" in names

    def test_to_specs_unsupported_type_raises(self):
        """Unsupported types raise AdapterError."""
        with pytest.raises(AdapterError):
            AgnoAdapter.to_specs([42])

    def test_to_specs_function_with_parameters(self):
        """Function with dict parameters."""
        func = MockAgnoFunction(
            "test", "Test", parameters={"type": "object", "properties": {}}
        )

        specs = AgnoAdapter.to_specs([func])

        assert specs[0].parameters == {"type": "object", "properties": {}}


class TestAgnoAdapterFilterTools:
    """Tests for AgnoAdapter.filter_tools."""

    def test_filter_toolkit(self):
        """Filter tools from toolkit."""
        toolkit = MockAgnoToolkit({
            "get_price": MockAgnoFunction("get_price", "Price"),
            "get_news": MockAgnoFunction("get_news", "News"),
            "get_weather": MockAgnoFunction("get_weather", "Weather"),
        })
        filtered = ToolCollection(tools=[
            ToolSpec(name="get_price", description="Price"),
        ])

        result = AgnoAdapter.filter_tools([toolkit], filtered)

        assert len(result) == 1
        assert result[0].name == "get_price"

    def test_filter_legacy_dict(self):
        """Filter legacy dict format."""
        funcs = {
            "get_price": MockAgnoFunction("get_price", "Price"),
            "get_news": MockAgnoFunction("get_news", "News"),
        }
        filtered = ToolCollection(tools=[
            ToolSpec(name="get_news", description="News"),
        ])

        result = AgnoAdapter.filter_tools(funcs, filtered)

        assert len(result) == 1
        assert result[0].name == "get_news"

    def test_filter_empty_collection(self):
        """Filter with empty collection."""
        toolkit = MockAgnoToolkit({
            "test": MockAgnoFunction("test", "Test"),
        })

        result = AgnoAdapter.filter_tools([toolkit], ToolCollection())

        assert result == []


class TestAgnoConvenienceFilterTools:
    """Tests for module-level filter_tools function."""

    def test_filter_tools_convenience(self):
        """Convenience function delegates to adapter."""
        toolkit = MockAgnoToolkit({
            "test": MockAgnoFunction("test", "Test"),
        })
        filtered = ToolCollection(tools=[ToolSpec(name="test", description="Test")])

        result = filter_tools([toolkit], filtered)

        assert len(result) == 1

    def test_filter_tools_single_toolkit(self):
        """Convenience function handles single toolkit (not in list)."""
        toolkit = MockAgnoToolkitLegacy({
            "test": MockAgnoFunction("test", "Test"),
        })
        filtered = ToolCollection(tools=[ToolSpec(name="test", description="Test")])

        result = filter_tools(toolkit, filtered)

        assert len(result) == 1
