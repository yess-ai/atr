"""Tests for RoutingLLM protocol."""

from __future__ import annotations

import pytest

from atr.llm import RoutingLLM

from .conftest import MockRoutingLLM


class TestRoutingLLMProtocol:
    """Tests for RoutingLLM protocol definition."""

    def test_routingllm_protocol_exists(self):
        """RoutingLLM is defined as a Protocol."""
        assert RoutingLLM is not None

    def test_routingllm_is_runtime_checkable(self):
        """RoutingLLM can be used with isinstance()."""
        mock = MockRoutingLLM(return_tools=[])

        # Protocol is runtime_checkable
        assert isinstance(mock, RoutingLLM)

    def test_routingllm_complete_method(self):
        """RoutingLLM has complete method."""
        assert hasattr(RoutingLLM, "complete")

    def test_routingllm_acomplete_method(self):
        """RoutingLLM has acomplete method."""
        assert hasattr(RoutingLLM, "acomplete")


class TestMockRoutingLLM:
    """Tests that MockRoutingLLM correctly implements RoutingLLM."""

    def test_mock_llm_complete(self):
        """MockRoutingLLM.complete returns configured tools."""
        mock = MockRoutingLLM(return_tools=["tool1", "tool2"])

        result = mock.complete("test prompt")

        assert "tool1" in result
        assert "tool2" in result

    def test_mock_llm_complete_with_system_prompt(self):
        """MockRoutingLLM.complete accepts system_prompt."""
        mock = MockRoutingLLM(return_tools=["tool1"])

        mock.complete("prompt", system_prompt="system instructions")

        assert mock.last_system_prompt == "system instructions"

    @pytest.mark.asyncio
    async def test_mock_llm_acomplete(self):
        """MockRoutingLLM.acomplete returns configured tools."""
        mock = MockRoutingLLM(return_tools=["tool1", "tool2"])

        result = await mock.acomplete("test prompt")

        assert "tool1" in result
        assert "tool2" in result

    def test_mock_llm_tracks_calls(self):
        """MockRoutingLLM tracks call count and last prompt."""
        mock = MockRoutingLLM(return_tools=[])

        mock.complete("first prompt")
        mock.complete("second prompt")

        assert mock.call_count == 2
        assert mock.last_prompt == "second prompt"

    def test_mock_llm_raises_configured_error(self):
        """MockRoutingLLM raises configured error."""
        mock = MockRoutingLLM(raise_error=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            mock.complete("prompt")

    @pytest.mark.asyncio
    async def test_mock_llm_acomplete_raises_configured_error(self):
        """MockRoutingLLM.acomplete raises configured error."""
        mock = MockRoutingLLM(raise_error=RuntimeError("Async error"))

        with pytest.raises(RuntimeError, match="Async error"):
            await mock.acomplete("prompt")
