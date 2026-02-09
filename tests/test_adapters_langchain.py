"""Tests for LangChain adapter."""

from __future__ import annotations

import pytest

from atr.adapters.langchain import (
    LangChainAdapter,
    filter_tools,
    create_router_node,
    create_async_router_node,
)
from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec

from .conftest import MockRoutingLLM


class MockBaseTool:
    """Mock LangChain BaseTool."""

    def __init__(self, name: str, description: str = "", args_schema=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema


class MockPydanticModel:
    """Mock Pydantic model with model_json_schema."""

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"arg1": {"type": "string"}}}


class MockLegacyPydanticModel:
    """Mock old-style Pydantic model with schema()."""

    @classmethod
    def schema(cls):
        return {"type": "object", "properties": {"arg1": {"type": "string"}}}


class TestLangChainAdapterToSpecs:
    """Tests for LangChainAdapter.to_specs."""

    def test_to_specs_basetool(self):
        """Convert BaseTool-like objects."""
        tools = [
            MockBaseTool("read_file", "Read a file"),
            MockBaseTool("write_file", "Write a file"),
        ]

        specs = LangChainAdapter.to_specs(tools)

        assert len(specs) == 2
        assert specs[0].name == "read_file"
        assert specs[0].description == "Read a file"
        assert specs[0].source == "langchain"

    def test_to_specs_with_pydantic_schema(self):
        """Convert tools with Pydantic v2 args_schema."""
        tool = MockBaseTool("test", "Test", args_schema=MockPydanticModel)

        specs = LangChainAdapter.to_specs([tool])

        assert specs[0].parameters == {"type": "object", "properties": {"arg1": {"type": "string"}}}

    def test_to_specs_with_legacy_schema(self):
        """Convert tools with Pydantic v1 schema."""
        tool = MockBaseTool("test", "Test", args_schema=MockLegacyPydanticModel)

        specs = LangChainAdapter.to_specs([tool])

        assert specs[0].parameters == {"type": "object", "properties": {"arg1": {"type": "string"}}}

    def test_to_specs_callable(self):
        """Convert plain callable functions."""

        def my_function():
            """A helpful function."""
            pass

        specs = LangChainAdapter.to_specs([my_function])

        assert len(specs) == 1
        assert specs[0].name == "my_function"
        assert specs[0].description == "A helpful function."

    def test_to_specs_callable_no_doc(self):
        """Convert callable without docstring."""

        def no_doc_func():
            pass

        # Remove docstring
        no_doc_func.__doc__ = None

        specs = LangChainAdapter.to_specs([no_doc_func])

        assert specs[0].description == ""

    def test_to_specs_skips_unknown(self):
        """Skip objects that can't be converted."""

        class NoName:
            pass

        specs = LangChainAdapter.to_specs([NoName()])

        assert len(specs) == 0

    def test_to_specs_empty_list(self):
        """Handle empty list."""
        specs = LangChainAdapter.to_specs([])
        assert specs == []

    def test_to_specs_preserves_original(self):
        """Original tool stored in metadata."""
        tool = MockBaseTool("test", "Test")

        specs = LangChainAdapter.to_specs([tool])

        assert specs[0].metadata["original_tool"] is tool

    def test_to_specs_none_description(self):
        """Handle None description."""
        tool = MockBaseTool("test", None)

        specs = LangChainAdapter.to_specs([tool])

        assert specs[0].description == ""


class TestLangChainAdapterFilterTools:
    """Tests for LangChainAdapter.filter_tools."""

    def test_filter_basetools(self):
        """Filter BaseTool-like objects."""
        tools = [
            MockBaseTool("read_file", "Read"),
            MockBaseTool("write_file", "Write"),
            MockBaseTool("delete_file", "Delete"),
        ]
        filtered = ToolCollection(tools=[
            ToolSpec(name="read_file", description="Read"),
            ToolSpec(name="delete_file", description="Delete"),
        ])

        result = LangChainAdapter.filter_tools(tools, filtered)

        assert len(result) == 2
        names = [t.name for t in result]
        assert "read_file" in names
        assert "delete_file" in names

    def test_filter_callables(self):
        """Filter plain callable functions."""

        def func_a():
            pass

        def func_b():
            pass

        tools = [func_a, func_b]
        filtered = ToolCollection(tools=[ToolSpec(name="func_a", description="")])

        result = LangChainAdapter.filter_tools(tools, filtered)

        assert len(result) == 1
        assert result[0].__name__ == "func_a"

    def test_filter_empty_collection(self):
        """Filter with empty collection."""
        tools = [MockBaseTool("test", "Test")]

        result = LangChainAdapter.filter_tools(tools, ToolCollection())

        assert result == []


class TestLangChainConvenienceFilterTools:
    """Tests for module-level filter_tools function."""

    def test_filter_tools_convenience(self):
        """Convenience function delegates to adapter."""
        tools = [MockBaseTool("test", "Test")]
        filtered = ToolCollection(tools=[ToolSpec(name="test", description="Test")])

        result = filter_tools(tools, filtered)

        assert len(result) == 1


class TestCreateRouterNode:
    """Tests for create_router_node."""

    def test_create_router_node_basic(self):
        """Creates a callable node function."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read"), MockBaseTool("write_file", "Write")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_router_node(router, tools)

        assert callable(node)

    def test_create_router_node_filters(self):
        """Node function filters tools based on query."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read"), MockBaseTool("write_file", "Write")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_router_node(router, tools)
        state = {"query": "Read a file"}
        result = node(state)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0].name == "read_file"

    def test_create_router_node_empty_query(self):
        """Node returns all tools when query is empty."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read"), MockBaseTool("write_file", "Write")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_router_node(router, tools)
        state = {"query": ""}
        result = node(state)

        assert len(result["tools"]) == 2

    def test_create_router_node_custom_keys(self):
        """Node uses custom state keys."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_router_node(router, tools, state_key="filtered", query_key="user_query")
        state = {"user_query": "Read file"}
        result = node(state)

        assert "filtered" in result


class TestCreateAsyncRouterNode:
    """Tests for create_async_router_node."""

    def test_create_async_router_node_is_sync_factory(self):
        """create_async_router_node is a sync function (not a coroutine)."""
        import inspect

        assert not inspect.iscoroutinefunction(create_async_router_node)

    def test_create_async_router_node_returns_coroutine_function(self):
        """Returns an async callable node."""
        import inspect
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_async_router_node(router, tools)

        assert inspect.iscoroutinefunction(node)

    @pytest.mark.asyncio
    async def test_create_async_router_node_filters(self):
        """Async node filters tools based on query."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read"), MockBaseTool("write_file", "Write")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_async_router_node(router, tools)
        state = {"query": "Read a file"}
        result = await node(state)

        assert len(result["tools"]) == 1
        assert result["tools"][0].name == "read_file"

    @pytest.mark.asyncio
    async def test_create_async_router_node_empty_query(self):
        """Async node returns all tools when query is empty."""
        from atr import ToolRouter

        llm = MockRoutingLLM(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        tools = [MockBaseTool("read_file", "Read"), MockBaseTool("write_file", "Write")]
        router.add_tools(LangChainAdapter.to_specs(tools))

        node = create_async_router_node(router, tools)
        state = {"query": ""}
        result = await node(state)

        assert len(result["tools"]) == 2
