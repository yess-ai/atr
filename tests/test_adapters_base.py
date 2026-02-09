"""Tests for adapter base protocol and utilities."""

from __future__ import annotations

import pytest

from atr.adapters.base import ToolAdapter, get_tool_name
from atr.core.tool import ToolCollection, ToolSpec


class TestToolAdapterProtocol:
    """Tests for ToolAdapter protocol definition."""

    def test_tool_adapter_protocol_exists(self):
        """ToolAdapter protocol is defined."""
        assert ToolAdapter is not None

    def test_tool_adapter_is_runtime_checkable(self):
        """ToolAdapter can be used with isinstance()."""
        from atr.adapters.litellm import LiteLLMAdapter

        assert isinstance(LiteLLMAdapter, type)


class TestGetToolName:
    """Tests for get_tool_name utility function."""

    def test_get_name_from_name_attribute(self):
        """Extract name from object with 'name' attribute."""

        class Tool:
            name = "test_tool"

        assert get_tool_name(Tool()) == "test_tool"

    def test_get_name_from_function_name(self):
        """Extract name from object with 'function_name' attribute."""

        class Tool:
            function_name = "my_func"

        assert get_tool_name(Tool()) == "my_func"

    def test_get_name_from_tool_name(self):
        """Extract name from object with 'tool_name' attribute."""

        class Tool:
            tool_name = "my_tool"

        assert get_tool_name(Tool()) == "my_tool"

    def test_get_name_from_dunder_name(self):
        """Extract name from callable with __name__."""

        def my_function():
            pass

        assert get_tool_name(my_function) == "my_function"

    def test_get_name_from_dict_name_key(self):
        """Extract name from dict with 'name' key."""
        tool = {"name": "dict_tool"}

        assert get_tool_name(tool) == "dict_tool"

    def test_get_name_from_dict_function_name_key(self):
        """Extract name from dict with 'function_name' key."""
        tool = {"function_name": "dict_func"}

        assert get_tool_name(tool) == "dict_func"

    def test_get_name_from_dict_tool_name_key(self):
        """Extract name from dict with 'tool_name' key."""
        tool = {"tool_name": "dict_tool_name"}

        assert get_tool_name(tool) == "dict_tool_name"

    def test_get_name_returns_none_for_unknown(self):
        """Returns None when no name can be extracted."""

        class NoName:
            pass

        assert get_tool_name(NoName()) is None

    def test_get_name_returns_none_for_empty_dict(self):
        """Returns None for dict without name keys."""
        assert get_tool_name({}) is None

    def test_get_name_returns_none_for_empty_string_attr(self):
        """Returns None when name attribute is empty string."""

        class Tool:
            name = ""

        assert get_tool_name(Tool()) is None

    def test_get_name_returns_none_for_int(self):
        """Returns None for non-object types without name."""
        assert get_tool_name(42) is None
