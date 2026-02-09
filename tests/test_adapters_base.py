"""Tests for adapter base protocol."""

from __future__ import annotations

import pytest

from atr.adapters.base import ToolAdapter
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
