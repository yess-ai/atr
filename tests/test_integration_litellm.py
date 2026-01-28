"""Tests for LiteLLM integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atr import ToolSpec
from atr.core.tool import ToolCollection
from atr.adapters.litellm import (
    ATRToolRoutingHook,
    LiteLLMAdapter,
    create_hook,
    extract_query_from_messages,
)


class TestLiteLLMAdapter:
    """Tests for LiteLLMAdapter."""

    def test_to_specs_wrapped_format(self):
        """Test converting wrapped function format."""
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

        specs = LiteLLMAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "get_weather"
        assert specs[0].description == "Get weather for a location"
        assert specs[0].source == "litellm"

    def test_to_specs_unwrapped_format(self):
        """Test converting unwrapped function format."""
        tools = [
            {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {"type": "object"},
            }
        ]

        specs = LiteLLMAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "send_email"
        assert specs[0].description == "Send an email"

    def test_to_specs_skips_invalid_tools(self):
        """Test that tools without names are skipped."""
        tools = [
            {"type": "function", "function": {"description": "No name"}},
            {
                "type": "function",
                "function": {"name": "valid_tool", "description": "Has name"},
            },
        ]

        specs = LiteLLMAdapter.to_specs(tools)

        assert len(specs) == 1
        assert specs[0].name == "valid_tool"

    def test_to_specs_empty_list(self):
        """Test converting empty tool list."""
        specs = LiteLLMAdapter.to_specs([])
        assert specs == []

    def test_to_specs_preserves_original_tool(self):
        """Test that original tool is preserved in metadata."""
        tools = [
            {
                "type": "function",
                "function": {"name": "test_tool", "description": "Test"},
            }
        ]

        specs = LiteLLMAdapter.to_specs(tools)

        assert specs[0].metadata["original_tool"] == tools[0]

    def test_filter_tools_wrapped_format(self):
        """Test filtering wrapped format tools."""
        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Weather"},
            },
            {
                "type": "function",
                "function": {"name": "send_email", "description": "Email"},
            },
            {
                "type": "function",
                "function": {"name": "search_web", "description": "Search"},
            },
        ]

        filtered_collection = ToolCollection()
        filtered_collection.add(ToolSpec(name="get_weather", description="Weather"))
        filtered_collection.add(ToolSpec(name="search_web", description="Search"))

        result = LiteLLMAdapter.filter_tools(tools, filtered_collection)

        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "get_weather" in names
        assert "search_web" in names
        assert "send_email" not in names

    def test_filter_tools_unwrapped_format(self):
        """Test filtering unwrapped format tools."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
        ]

        filtered_collection = ToolCollection()
        filtered_collection.add(ToolSpec(name="tool_a", description="A"))

        result = LiteLLMAdapter.filter_tools(tools, filtered_collection)

        assert len(result) == 1
        assert result[0]["name"] == "tool_a"

    def test_filter_tools_empty_filtered(self):
        """Test filtering with empty filtered collection."""
        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "Test"},
            }
        ]

        result = LiteLLMAdapter.filter_tools(tools, ToolCollection())

        assert result == []


class TestExtractQueryFromMessages:
    """Tests for extract_query_from_messages."""

    def test_extract_simple_user_message(self):
        """Test extracting from simple user message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is the weather?"},
        ]

        query = extract_query_from_messages(messages)

        assert query == "What is the weather?"

    def test_extract_last_user_message(self):
        """Test extracting the last user message in conversation."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up question"},
        ]

        query = extract_query_from_messages(messages)

        assert query == "Follow-up question"

    def test_extract_multimodal_text_part(self):
        """Test extracting from multimodal message with text part."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]

        query = extract_query_from_messages(messages)

        assert query == "What is in this image?"

    def test_extract_multimodal_multiple_text_parts(self):
        """Test extracting from multimodal message with multiple text parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this."},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ]

        query = extract_query_from_messages(messages)

        assert query == "Look at this. What do you see?"

    def test_extract_multimodal_string_parts(self):
        """Test extracting from content with string parts."""
        messages = [
            {
                "role": "user",
                "content": ["Part one", "Part two"],
            }
        ]

        query = extract_query_from_messages(messages)

        assert query == "Part one Part two"

    def test_extract_empty_messages(self):
        """Test extracting from empty messages list."""
        query = extract_query_from_messages([])
        assert query is None

    def test_extract_no_user_messages(self):
        """Test extracting when no user messages exist."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        query = extract_query_from_messages(messages)

        assert query is None

    def test_extract_none_content(self):
        """Test extracting from message with None content."""
        messages = [{"role": "user", "content": None}]

        query = extract_query_from_messages(messages)

        assert query is None


class TestATRToolRoutingHook:
    """Tests for ATRToolRoutingHook."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns specific tools."""
        llm = MagicMock()
        llm.acomplete = AsyncMock(return_value="get_weather\nsearch_web")
        llm.complete = MagicMock(return_value="get_weather\nsearch_web")
        return llm

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            },
            {
                "type": "function",
                "function": {"name": "send_email", "description": "Send email"},
            },
            {
                "type": "function",
                "function": {"name": "search_web", "description": "Search web"},
            },
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read file"},
            },
            {
                "type": "function",
                "function": {"name": "write_file", "description": "Write file"},
            },
        ]

    @pytest.mark.asyncio
    async def test_pre_call_hook_filters_tools(self, mock_llm, sample_tools):
        """Test that pre_call_hook filters tools based on query."""
        hook = ATRToolRoutingHook(llm=mock_llm, min_tools_threshold=3)

        data = {
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": sample_tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should be filtered to matched tools
        assert len(result["tools"]) == 2
        names = [t["function"]["name"] for t in result["tools"]]
        assert "get_weather" in names
        assert "search_web" in names

    @pytest.mark.asyncio
    async def test_pre_call_hook_skips_when_disabled(self, mock_llm, sample_tools):
        """Test that routing is skipped when disabled."""
        hook = ATRToolRoutingHook(llm=mock_llm, enabled=False)

        data = {
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": sample_tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should return original tools
        assert result["tools"] == sample_tools

    @pytest.mark.asyncio
    async def test_pre_call_hook_skips_non_completion(self, mock_llm, sample_tools):
        """Test that non-completion calls are skipped."""
        hook = ATRToolRoutingHook(llm=mock_llm)

        data = {
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": sample_tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "embedding")

        # Should return original data
        assert result["tools"] == sample_tools

    @pytest.mark.asyncio
    async def test_pre_call_hook_skips_no_tools(self, mock_llm):
        """Test that requests without tools are skipped."""
        hook = ATRToolRoutingHook(llm=mock_llm)

        data = {"messages": [{"role": "user", "content": "Hello"}]}

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should return original data
        assert "tools" not in result

    @pytest.mark.asyncio
    async def test_pre_call_hook_skips_below_threshold(self, mock_llm):
        """Test that small tool sets are skipped."""
        hook = ATRToolRoutingHook(llm=mock_llm, min_tools_threshold=5)

        tools = [
            {
                "type": "function",
                "function": {"name": "tool1", "description": "Tool 1"},
            },
            {
                "type": "function",
                "function": {"name": "tool2", "description": "Tool 2"},
            },
        ]

        data = {
            "messages": [{"role": "user", "content": "Test query"}],
            "tools": tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should return original tools (below threshold)
        assert result["tools"] == tools

    @pytest.mark.asyncio
    async def test_pre_call_hook_skips_no_query(self, mock_llm, sample_tools):
        """Test that requests without user query are skipped."""
        hook = ATRToolRoutingHook(llm=mock_llm, min_tools_threshold=3)

        data = {
            "messages": [{"role": "system", "content": "You are helpful"}],
            "tools": sample_tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should return original tools
        assert result["tools"] == sample_tools

    @pytest.mark.asyncio
    async def test_pre_call_hook_fails_open_on_error(self, sample_tools):
        """Test that routing errors return original tools (fail-open)."""
        mock_llm = MagicMock()
        mock_llm.acomplete = AsyncMock(side_effect=Exception("LLM error"))

        hook = ATRToolRoutingHook(llm=mock_llm, min_tools_threshold=3)

        data = {
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": sample_tools,
        }

        result = await hook.async_pre_call_hook({}, None, data, "completion")

        # Should return original tools on error
        assert result["tools"] == sample_tools

    @pytest.mark.asyncio
    async def test_pre_call_hook_clears_tools_between_requests(self, mock_llm):
        """Test that tools are cleared between requests."""
        hook = ATRToolRoutingHook(llm=mock_llm, min_tools_threshold=2)

        # First request
        tools1 = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            },
            {
                "type": "function",
                "function": {"name": "send_email", "description": "Send email"},
            },
        ]
        data1 = {
            "messages": [{"role": "user", "content": "Weather query"}],
            "tools": tools1,
        }
        await hook.async_pre_call_hook({}, None, data1, "completion")

        # Second request with different tools
        tools2 = [
            {
                "type": "function",
                "function": {"name": "search_web", "description": "Search web"},
            },
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read file"},
            },
        ]
        data2 = {
            "messages": [{"role": "user", "content": "Search query"}],
            "tools": tools2,
        }

        # Change mock to return search_web
        mock_llm.acomplete = AsyncMock(return_value="search_web")

        result = await hook.async_pre_call_hook({}, None, data2, "completion")

        # Should only contain tools from second request
        names = [t["function"]["name"] for t in result["tools"]]
        assert "search_web" in names
        assert "get_weather" not in names

    def test_hook_initialization_defaults(self):
        """Test hook initializes with correct defaults."""
        hook = ATRToolRoutingHook()

        assert hook.enabled is True
        assert hook.max_tools == 10
        assert hook.min_tools_threshold == 5
        assert hook.llm_provider == "openrouter"
        assert hook.llm_model is None

    def test_hook_initialization_custom(self, mock_llm):
        """Test hook initializes with custom values."""
        hook = ATRToolRoutingHook(
            enabled=False,
            max_tools=5,
            min_tools_threshold=10,
            llm_provider="anthropic",
            llm_model="claude-3-haiku",
            llm=mock_llm,
        )

        assert hook.enabled is False
        assert hook.max_tools == 5
        assert hook.min_tools_threshold == 10
        assert hook.llm_provider == "anthropic"
        assert hook.llm_model == "claude-3-haiku"


class TestCreateHook:
    """Tests for create_hook factory function."""

    def test_create_hook_defaults(self):
        """Test creating hook with defaults."""
        hook = create_hook()

        assert isinstance(hook, ATRToolRoutingHook)
        assert hook.enabled is True
        assert hook.max_tools == 10

    def test_create_hook_custom(self):
        """Test creating hook with custom values."""
        mock_llm = MagicMock()

        hook = create_hook(
            enabled=False,
            max_tools=5,
            min_tools_threshold=3,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm=mock_llm,
        )

        assert hook.enabled is False
        assert hook.max_tools == 5
        assert hook.min_tools_threshold == 3
        assert hook.llm_provider == "openai"
        assert hook.llm_model == "gpt-4o-mini"
        assert hook._llm is mock_llm


class TestLazyImports:
    """Tests for lazy import functionality."""

    def test_litellm_adapter_import(self):
        """Test LiteLLMAdapter can be imported from adapters."""
        from atr.adapters import LiteLLMAdapter

        assert LiteLLMAdapter is not None

    def test_atr_tool_routing_hook_import(self):
        """Test ATRToolRoutingHook can be imported from adapters."""
        from atr.adapters import ATRToolRoutingHook

        assert ATRToolRoutingHook is not None

    def test_create_hook_import(self):
        """Test create_hook can be imported from adapters."""
        from atr.adapters import create_hook

        assert create_hook is not None

