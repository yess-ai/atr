"""Tests for OpenAI adapter."""

from __future__ import annotations

import pytest

from atr.adapters.openai import OpenAIAdapter, filter_tools
from atr.core.tool import ToolCollection, ToolSpec


class TestOpenAIAdapterToSpecs:
    """Tests for OpenAIAdapter.to_specs."""

    def test_to_specs_wrapped_format(self):
        """Convert wrapped function format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        specs = OpenAIAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "get_weather"
        assert specs[0].description == "Get weather for a location"
        assert specs[0].source == "openai"

    def test_to_specs_unwrapped_format(self):
        """Convert unwrapped function format."""
        tools = [
            {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {"type": "object"},
            }
        ]

        specs = OpenAIAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "send_email"
        assert specs[0].source == "openai"

    def test_to_specs_skips_invalid(self):
        """Skip tools without names."""
        tools = [
            {"type": "function", "function": {"description": "No name"}},
            {"type": "function", "function": {"name": "valid", "description": "Valid"}},
        ]

        specs = OpenAIAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "valid"

    def test_to_specs_empty_list(self):
        """Handle empty tool list."""
        specs = OpenAIAdapter.to_specs([])
        assert specs == []

    def test_to_specs_preserves_original(self):
        """Original tool stored in metadata."""
        tool = {"type": "function", "function": {"name": "test", "description": "Test"}}

        specs = OpenAIAdapter.to_specs([tool])

        assert specs[0].metadata["original_tool"] == tool

    def test_to_specs_missing_description(self):
        """Handle missing description."""
        tools = [{"type": "function", "function": {"name": "test"}}]

        specs = OpenAIAdapter.to_specs(tools)

        assert specs[0].description == ""

    def test_to_specs_missing_parameters(self):
        """Handle missing parameters."""
        tools = [{"type": "function", "function": {"name": "test", "description": "Test"}}]

        specs = OpenAIAdapter.to_specs(tools)

        assert specs[0].parameters is None


class TestOpenAIAdapterFilterTools:
    """Tests for OpenAIAdapter.filter_tools."""

    def test_filter_wrapped_format(self):
        """Filter wrapped format tools."""
        tools = [
            {"type": "function", "function": {"name": "get_weather", "description": "Weather"}},
            {"type": "function", "function": {"name": "send_email", "description": "Email"}},
            {"type": "function", "function": {"name": "search_web", "description": "Search"}},
        ]
        filtered = ToolCollection(tools=[
            ToolSpec(name="get_weather", description="Weather"),
        ])

        result = OpenAIAdapter.filter_tools(tools, filtered)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_filter_unwrapped_format(self):
        """Filter unwrapped format tools."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
        ]
        filtered = ToolCollection(tools=[ToolSpec(name="tool_b", description="B")])

        result = OpenAIAdapter.filter_tools(tools, filtered)

        assert len(result) == 1
        assert result[0]["name"] == "tool_b"

    def test_filter_empty_collection(self):
        """Filter with empty collection."""
        tools = [{"type": "function", "function": {"name": "test", "description": "Test"}}]

        result = OpenAIAdapter.filter_tools(tools, ToolCollection())

        assert result == []


class TestOpenAIConvenienceFilterTools:
    """Tests for module-level filter_tools function."""

    def test_filter_tools_convenience(self):
        """Convenience function delegates to adapter."""
        tools = [
            {"type": "function", "function": {"name": "test", "description": "Test"}},
        ]
        filtered = ToolCollection(tools=[ToolSpec(name="test", description="Test")])

        result = filter_tools(tools, filtered)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "test"
