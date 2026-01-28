"""Tests for ToolRouter - the main routing logic."""

from __future__ import annotations

import pytest

from atr import ToolCollection, ToolRouter, ToolSpec
from atr.core.strategies import PassthroughStrategy

from .conftest import MockRoutingLLM


class TestToolRouterInit:
    """Tests for ToolRouter initialization."""

    def test_router_init_with_llm(self, mock_llm):
        """Router accepts a RoutingLLM instance."""
        llm = mock_llm(return_tools=["test"])
        router = ToolRouter(llm=llm)

        assert router is not None

    def test_router_init_without_llm_uses_passthrough(self):
        """Router without LLM uses PassthroughStrategy."""
        router = ToolRouter()

        assert isinstance(router.strategy, PassthroughStrategy)

    def test_router_init_with_custom_strategy(self, mock_llm):
        """Router accepts custom strategy."""
        from atr.core.strategies import LLMFilterStrategy

        llm = mock_llm(return_tools=[])
        strategy = LLMFilterStrategy(llm=llm)
        router = ToolRouter(strategy=strategy)

        assert router.strategy is strategy

    def test_router_init_strategy_overrides_llm(self, mock_llm):
        """Custom strategy takes precedence over llm parameter."""
        llm = mock_llm(return_tools=[])
        passthrough = PassthroughStrategy()
        router = ToolRouter(llm=llm, strategy=passthrough)

        assert isinstance(router.strategy, PassthroughStrategy)

    def test_router_init_with_max_tools(self, mock_llm):
        """Router accepts max_tools parameter."""
        llm = mock_llm(return_tools=[])
        router = ToolRouter(llm=llm, max_tools=5)

        assert router is not None


class TestToolRouterTools:
    """Tests for managing tools in router."""

    def test_router_tools_property(self, mock_llm, sample_toolspecs):
        """tools property returns ToolCollection."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))
        router.add_tools(sample_toolspecs)

        assert isinstance(router.tools, ToolCollection)
        assert len(router.tools) == len(sample_toolspecs)

    def test_router_add_tool_single(self, mock_llm):
        """Can add a single tool with add_tool()."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))
        tool = ToolSpec(name="test", description="Test tool")

        router.add_tool(tool)

        assert len(router.tools) == 1
        assert "test" in router.tools.names

    def test_router_add_tools_list(self, mock_llm, sample_toolspecs):
        """Can add list of ToolSpecs with add_tools()."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))

        router.add_tools(sample_toolspecs)

        assert len(router.tools) == len(sample_toolspecs)

    def test_router_add_tools_multiple_calls(
        self, mock_llm, file_tools, communication_tools
    ):
        """Multiple add_tools calls accumulate tools."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))

        router.add_tools(file_tools)
        router.add_tools(communication_tools)

        assert len(router.tools) == len(file_tools) + len(communication_tools)

    def test_router_add_tools_empty_list(self, mock_llm):
        """Adding empty list doesn't break router."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))

        router.add_tools([])

        assert len(router.tools) == 0

    def test_router_clear_tools(self, mock_llm, sample_toolspecs):
        """clear_tools() removes all tools."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))
        router.add_tools(sample_toolspecs)

        router.clear_tools()

        assert len(router.tools) == 0


class TestToolRouterRouteSync:
    """Tests for synchronous routing."""

    def test_router_route_returns_toolcollection(self, mock_llm, sample_toolspecs):
        """route() returns a ToolCollection."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file"]))
        router.add_tools(sample_toolspecs)

        result = router.route("Read the README file")

        assert isinstance(result, ToolCollection)

    def test_router_route_filters_by_query(self, mock_llm, sample_toolspecs):
        """Routing filters tools based on LLM response."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file", "write_file"]))
        router.add_tools(sample_toolspecs)

        result = router.route("Read and write files")

        assert "read_file" in result.names
        assert "write_file" in result.names
        assert "send_email" not in result.names

    def test_router_route_calls_llm(self, mock_llm, sample_toolspecs):
        """Routing calls the LLM."""
        llm = mock_llm(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        router.add_tools(sample_toolspecs)

        router.route("Test query")

        assert llm.call_count == 1
        assert "Test query" in llm.last_prompt

    def test_router_route_empty_query(self, mock_llm, sample_toolspecs):
        """Router handles empty query string."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))
        router.add_tools(sample_toolspecs)

        # Should not raise
        result = router.route("")
        assert isinstance(result, ToolCollection)

    def test_router_route_no_tools_added(self, mock_llm):
        """Router handles routing with no tools added."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))

        result = router.route("Any query")

        assert isinstance(result, ToolCollection)
        assert len(result) == 0

    def test_router_route_all_tools_match(self, mock_llm, sample_toolspecs):
        """Query can match all available tools."""
        all_names = [t.name for t in sample_toolspecs]
        router = ToolRouter(llm=mock_llm(return_tools=all_names))
        router.add_tools(sample_toolspecs)

        result = router.route("I need all tools")

        assert len(result) == len(sample_toolspecs)

    def test_router_route_no_tools_match_fallback(self, mock_llm, sample_toolspecs):
        """When LLM returns empty, fallback to all tools."""
        router = ToolRouter(llm=mock_llm(return_tools=[]))
        router.add_tools(sample_toolspecs)

        result = router.route("Something completely unrelated")

        # LLMFilterStrategy falls back to all tools when none selected
        assert len(result) == len(sample_toolspecs)

    def test_router_route_passthrough_returns_all(self, sample_toolspecs):
        """PassthroughStrategy returns all tools."""
        router = ToolRouter()  # No LLM = PassthroughStrategy
        router.add_tools(sample_toolspecs)

        result = router.route("Any query")

        assert len(result) == len(sample_toolspecs)


class TestToolRouterRouteAsync:
    """Tests for asynchronous routing."""

    @pytest.mark.asyncio
    async def test_router_aroute_returns_toolcollection(self, mock_llm, sample_toolspecs):
        """aroute() returns a ToolCollection."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file"]))
        router.add_tools(sample_toolspecs)

        result = await router.aroute("Read the README file")

        assert isinstance(result, ToolCollection)

    @pytest.mark.asyncio
    async def test_router_aroute_filters_by_query(self, mock_llm, sample_toolspecs):
        """Async routing filters tools based on LLM response."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file", "write_file"]))
        router.add_tools(sample_toolspecs)

        result = await router.aroute("Read and write files")

        assert "read_file" in result.names
        assert "write_file" in result.names
        assert "send_email" not in result.names

    @pytest.mark.asyncio
    async def test_router_aroute_calls_llm(self, mock_llm, sample_toolspecs):
        """Async routing calls the LLM."""
        llm = mock_llm(return_tools=["read_file"])
        router = ToolRouter(llm=llm)
        router.add_tools(sample_toolspecs)

        await router.aroute("Test query")

        assert llm.call_count == 1


class TestToolRouterEdgeCases:
    """Edge case tests for ToolRouter."""

    def test_router_route_special_characters(self, mock_llm, sample_toolspecs):
        """Router handles query with special characters."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file"]))
        router.add_tools(sample_toolspecs)

        # Should not raise
        result = router.route("Read file: /path/to/file.txt & process <data>")
        assert result is not None

    def test_router_route_unicode(self, mock_llm, sample_toolspecs):
        """Router handles non-ASCII query."""
        router = ToolRouter(llm=mock_llm(return_tools=["send_email"]))
        router.add_tools(sample_toolspecs)

        result = router.route("发送电子邮件 to 用户")
        assert result is not None

    def test_router_route_very_long_query(self, mock_llm, sample_toolspecs):
        """Router handles very long query string."""
        router = ToolRouter(llm=mock_llm(return_tools=["read_file"]))
        router.add_tools(sample_toolspecs)

        long_query = "Read the file " * 1000  # ~14KB query
        result = router.route(long_query)
        assert result is not None

    def test_router_route_invalid_tool_from_llm(self, mock_llm, sample_toolspecs):
        """Router handles LLM returning nonexistent tool names."""
        # LLM returns a mix of valid and invalid names
        router = ToolRouter(
            llm=mock_llm(return_tools=["read_file", "nonexistent_tool"])
        )
        router.add_tools(sample_toolspecs)

        result = router.route("Test query")

        # Should only contain valid tools
        assert "read_file" in result.names
        assert "nonexistent_tool" not in result.names

    def test_router_set_strategy(self, mock_llm, sample_toolspecs):
        """set_strategy() changes the routing strategy."""
        from atr.core.strategies import LLMFilterStrategy

        router = ToolRouter()  # PassthroughStrategy
        router.add_tools(sample_toolspecs)

        llm = mock_llm(return_tools=["read_file"])
        new_strategy = LLMFilterStrategy(llm=llm)
        router.set_strategy(new_strategy)

        result = router.route("Read files")

        assert len(result) == 1  # LLM filtered, not passthrough


class TestToolRouterLLMResponseParsing:
    """Tests for LLM response parsing edge cases."""

    def test_llm_response_with_dashes(self, mock_llm, sample_toolspecs):
        """Handles tool names prefixed with dashes."""
        llm = MockRoutingLLM()
        llm.return_tools = []  # We'll set response directly

        # Simulate LLM returning "- tool_name" format
        def custom_complete(prompt, system_prompt=None):
            return "- read_file\n- write_file"

        llm.complete = custom_complete
        router = ToolRouter(llm=llm)
        router.add_tools(sample_toolspecs)

        result = router.route("Test")

        assert "read_file" in result.names
        assert "write_file" in result.names

    def test_llm_response_with_numbering(self, mock_llm, sample_toolspecs):
        """Handles tool names with numbered format."""
        llm = MockRoutingLLM()

        def custom_complete(prompt, system_prompt=None):
            return "1. read_file\n2. write_file"

        llm.complete = custom_complete
        router = ToolRouter(llm=llm)
        router.add_tools(sample_toolspecs)

        result = router.route("Test")

        assert "read_file" in result.names
        assert "write_file" in result.names

    def test_llm_response_with_empty_lines(self, mock_llm, sample_toolspecs):
        """Handles response with empty lines."""
        llm = MockRoutingLLM()

        def custom_complete(prompt, system_prompt=None):
            return "read_file\n\n\nwrite_file\n"

        llm.complete = custom_complete
        router = ToolRouter(llm=llm)
        router.add_tools(sample_toolspecs)

        result = router.route("Test")

        assert "read_file" in result.names
        assert "write_file" in result.names
