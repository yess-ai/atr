"""Tests for FilterStrategy protocol and LLMFilterStrategy."""

from __future__ import annotations

import pytest

from atr import ToolCollection, ToolSpec
from atr.core.strategies import (
    BaseFilterStrategy,
    FilterStrategy,
    LLMFilterStrategy,
    PassthroughStrategy,
)

from .conftest import MockRoutingLLM


class TestFilterStrategyProtocol:
    """Tests for FilterStrategy protocol definition."""

    def test_filter_strategy_protocol_exists(self):
        """FilterStrategy protocol is defined."""
        assert FilterStrategy is not None

    def test_filter_strategy_is_runtime_checkable(self):
        """FilterStrategy is runtime_checkable."""
        # Can use isinstance() with protocol
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)

        assert isinstance(strategy, FilterStrategy)

    def test_passthrough_implements_protocol(self):
        """PassthroughStrategy satisfies FilterStrategy protocol."""
        strategy = PassthroughStrategy()

        assert isinstance(strategy, FilterStrategy)

    def test_llm_strategy_implements_protocol(self):
        """LLMFilterStrategy satisfies FilterStrategy protocol."""
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)

        assert isinstance(strategy, FilterStrategy)


class TestBaseFilterStrategy:
    """Tests for BaseFilterStrategy abstract base class."""

    def test_base_strategy_is_abstract(self):
        """BaseFilterStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFilterStrategy()  # type: ignore

    def test_base_strategy_requires_filter(self):
        """Subclass must implement filter method."""

        class IncompleteStrategy(BaseFilterStrategy):
            async def afilter(self, query, tools):
                return tools

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore

    def test_base_strategy_requires_afilter(self):
        """Subclass must implement afilter method."""

        class IncompleteStrategy(BaseFilterStrategy):
            def filter(self, query, tools):
                return tools

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore


class TestLLMFilterStrategy:
    """Tests for LLMFilterStrategy implementation."""

    def test_llm_strategy_init(self):
        """LLMFilterStrategy accepts RoutingLLM."""
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)

        assert strategy is not None
        assert strategy.llm is llm

    def test_llm_strategy_init_with_max_tools(self):
        """LLMFilterStrategy accepts max_tools parameter."""
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm, max_tools=5)

        assert strategy.max_tools == 5

    def test_llm_strategy_init_with_custom_prompt(self):
        """LLMFilterStrategy accepts custom system prompt."""
        llm = MockRoutingLLM(return_tools=[])
        custom_prompt = "Custom routing instructions"
        strategy = LLMFilterStrategy(llm=llm, system_prompt=custom_prompt)

        assert strategy.system_prompt == custom_prompt

    def test_llm_strategy_default_prompt(self):
        """LLMFilterStrategy has default system prompt."""
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)

        assert strategy.system_prompt is not None
        assert len(strategy.system_prompt) > 0

    def test_llm_strategy_filter(self, sample_toolspecs):
        """LLMFilterStrategy filters tools based on LLM response."""
        llm = MockRoutingLLM(return_tools=["read_file", "write_file"])
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection(tools=sample_toolspecs)

        result = strategy.filter("Work with files", collection)

        assert len(result) == 2
        assert "read_file" in result.names
        assert "write_file" in result.names

    def test_llm_strategy_filter_calls_llm(self, sample_toolspecs):
        """Strategy calls LLM with tools and query."""
        llm = MockRoutingLLM(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection(tools=sample_toolspecs)

        strategy.filter("Test query", collection)

        assert llm.call_count == 1
        assert "Test query" in llm.last_prompt
        # Check tools are included in prompt
        assert "read_file" in llm.last_prompt

    def test_llm_strategy_filter_uses_system_prompt(self, sample_toolspecs):
        """Strategy passes system prompt to LLM."""
        llm = MockRoutingLLM(return_tools=["read_file"])
        custom_prompt = "Custom instructions"
        strategy = LLMFilterStrategy(llm=llm, system_prompt=custom_prompt)
        collection = ToolCollection(tools=sample_toolspecs)

        strategy.filter("Test", collection)

        assert llm.last_system_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_llm_strategy_afilter(self, sample_toolspecs):
        """Async filter returns filtered tools."""
        llm = MockRoutingLLM(return_tools=["read_file", "write_file"])
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection(tools=sample_toolspecs)

        result = await strategy.afilter("Work with files", collection)

        assert len(result) == 2
        assert "read_file" in result.names

    def test_llm_strategy_filter_empty_collection(self):
        """Strategy handles empty tool collection."""
        llm = MockRoutingLLM(return_tools=["read_file"])
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection()

        result = strategy.filter("Test", collection)

        assert len(result) == 0
        # LLM should not be called for empty collection
        assert llm.call_count == 0

    def test_llm_strategy_filter_max_tools_limit(self, sample_toolspecs):
        """Strategy respects max_tools limit."""
        # Return all tool names
        all_names = [t.name for t in sample_toolspecs]
        llm = MockRoutingLLM(return_tools=all_names)
        strategy = LLMFilterStrategy(llm=llm, max_tools=3)
        collection = ToolCollection(tools=sample_toolspecs)

        result = strategy.filter("Need all tools", collection)

        assert len(result) <= 3

    def test_llm_strategy_filter_invalid_names_ignored(self, sample_toolspecs):
        """Strategy ignores tool names not in collection."""
        llm = MockRoutingLLM(
            return_tools=["read_file", "nonexistent", "also_fake"]
        )
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection(tools=sample_toolspecs)

        result = strategy.filter("Test", collection)

        assert "read_file" in result.names
        assert "nonexistent" not in result.names

    def test_llm_strategy_filter_fallback_on_empty_selection(self, sample_toolspecs):
        """Strategy falls back to all tools if LLM returns none."""
        llm = MockRoutingLLM(return_tools=["nonexistent"])  # No valid tools
        strategy = LLMFilterStrategy(llm=llm)
        collection = ToolCollection(tools=sample_toolspecs)

        result = strategy.filter("Test", collection)

        # Should fall back to all tools
        assert len(result) == len(sample_toolspecs)


class TestPassthroughStrategy:
    """Tests for PassthroughStrategy implementation."""

    def test_passthrough_init(self):
        """PassthroughStrategy can be instantiated."""
        strategy = PassthroughStrategy()

        assert strategy is not None

    def test_passthrough_filter_returns_all(self, sample_toolspecs):
        """PassthroughStrategy returns all tools."""
        strategy = PassthroughStrategy()
        collection = ToolCollection(tools=sample_toolspecs)

        result = strategy.filter("Any query", collection)

        assert len(result) == len(sample_toolspecs)
        assert result.names == collection.names

    def test_passthrough_filter_ignores_query(self, sample_toolspecs):
        """PassthroughStrategy doesn't use the query."""
        strategy = PassthroughStrategy()
        collection = ToolCollection(tools=sample_toolspecs)

        result1 = strategy.filter("Query 1", collection)
        result2 = strategy.filter("Completely different query", collection)

        assert result1.names == result2.names

    @pytest.mark.asyncio
    async def test_passthrough_afilter_returns_all(self, sample_toolspecs):
        """Async passthrough returns all tools."""
        strategy = PassthroughStrategy()
        collection = ToolCollection(tools=sample_toolspecs)

        result = await strategy.afilter("Any query", collection)

        assert len(result) == len(sample_toolspecs)

    def test_passthrough_empty_collection(self):
        """PassthroughStrategy handles empty collection."""
        strategy = PassthroughStrategy()
        collection = ToolCollection()

        result = strategy.filter("Test", collection)

        assert len(result) == 0
