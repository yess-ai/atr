"""Tests for MCP adapter."""

from __future__ import annotations

import pytest

from atr.adapters.mcp import MCPAdapter
from atr.core.exceptions import AdapterError
from atr.core.tool import ToolCollection, ToolSpec


class MockMCPTool:
    """Mock MCP Tool object."""

    def __init__(self, name: str, description: str = "", input_schema: dict | None = None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class MockPydanticSchema:
    """Mock Pydantic model with model_dump."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


class TestMCPAdapterToSpecs:
    """Tests for MCPAdapter.to_specs."""

    def test_to_specs_basic(self):
        """Convert basic MCP tools to ToolSpecs."""
        tools = [
            MockMCPTool("read_file", "Read a file"),
            MockMCPTool("write_file", "Write a file"),
        ]

        specs = MCPAdapter.to_specs(tools)

        assert len(specs) == 2
        assert specs[0].name == "read_file"
        assert specs[0].description == "Read a file"
        assert specs[0].source == "mcp"
        assert specs[1].name == "write_file"

    def test_to_specs_with_input_schema_dict(self):
        """Convert MCP tools with dict input schema."""
        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        tools = [MockMCPTool("read_file", "Read", schema)]

        specs = MCPAdapter.to_specs(tools)

        assert specs[0].parameters == schema

    def test_to_specs_with_pydantic_schema(self):
        """Convert MCP tools with Pydantic model_dump schema."""
        schema_data = {"type": "object", "properties": {}}
        pydantic_schema = MockPydanticSchema(schema_data)
        tool = MockMCPTool("test_tool", "Test")
        tool.inputSchema = pydantic_schema

        specs = MCPAdapter.to_specs([tool])

        assert specs[0].parameters == schema_data

    def test_to_specs_skips_nameless_tools(self):
        """Skip tools without names."""
        tool = MockMCPTool("", "No name tool")

        specs = MCPAdapter.to_specs([tool])

        assert len(specs) == 0

    def test_to_specs_empty_list(self):
        """Handle empty tool list."""
        specs = MCPAdapter.to_specs([])

        assert specs == []

    def test_to_specs_preserves_original(self):
        """Original tool stored in metadata."""
        tool = MockMCPTool("test", "Test")

        specs = MCPAdapter.to_specs([tool])

        assert specs[0].metadata["original_tool"] is tool

    def test_to_specs_none_description(self):
        """Handle None description."""
        tool = MockMCPTool("test", None)

        specs = MCPAdapter.to_specs([tool])

        assert specs[0].description == ""

    def test_to_specs_none_input_schema(self):
        """Handle None input schema."""
        tool = MockMCPTool("test", "Test", None)

        specs = MCPAdapter.to_specs([tool])

        assert specs[0].parameters is None


class TestMCPAdapterFilterTools:
    """Tests for MCPAdapter.filter_tools."""

    def test_filter_tools(self):
        """Filter MCP tools by ToolCollection."""
        tools = [
            MockMCPTool("read_file", "Read"),
            MockMCPTool("write_file", "Write"),
            MockMCPTool("delete_file", "Delete"),
        ]
        filtered = ToolCollection(tools=[
            ToolSpec(name="read_file", description="Read"),
            ToolSpec(name="delete_file", description="Delete"),
        ])

        result = MCPAdapter.filter_tools(tools, filtered)

        assert len(result) == 2
        names = [t.name for t in result]
        assert "read_file" in names
        assert "delete_file" in names
        assert "write_file" not in names

    def test_filter_tools_empty_collection(self):
        """Filter with empty collection returns nothing."""
        tools = [MockMCPTool("test", "Test")]

        result = MCPAdapter.filter_tools(tools, ToolCollection())

        assert result == []

    def test_filter_tools_no_match(self):
        """Filter when no tools match returns empty."""
        tools = [MockMCPTool("test", "Test")]
        filtered = ToolCollection(tools=[ToolSpec(name="other", description="Other")])

        result = MCPAdapter.filter_tools(tools, filtered)

        assert result == []


class TestMCPAdapterToSpecsFromDict:
    """Tests for MCPAdapter.to_specs_from_dict."""

    def test_to_specs_from_dict_basic(self):
        """Convert dict-format MCP tools."""
        tools = [
            {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}},
            {"name": "write_file", "description": "Write a file"},
        ]

        specs = MCPAdapter.to_specs_from_dict(tools)

        assert len(specs) == 2
        assert specs[0].name == "read_file"
        assert specs[0].parameters == {"type": "object"}
        assert specs[0].source == "mcp"
        assert specs[1].name == "write_file"
        assert specs[1].parameters is None

    def test_to_specs_from_dict_skips_nameless(self):
        """Skip dicts without names."""
        tools = [{"description": "No name"}]

        specs = MCPAdapter.to_specs_from_dict(tools)

        assert len(specs) == 0

    def test_to_specs_from_dict_empty_list(self):
        """Handle empty list."""
        specs = MCPAdapter.to_specs_from_dict([])

        assert specs == []

    def test_to_specs_from_dict_preserves_original(self):
        """Original dict stored in metadata."""
        tool = {"name": "test", "description": "Test"}

        specs = MCPAdapter.to_specs_from_dict([tool])

        assert specs[0].metadata["original_dict"] is tool
