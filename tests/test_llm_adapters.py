"""Tests for LLM adapters (OpenRouterLLM, OpenAILLM, AnthropicLLM).

These tests mock the underlying clients to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from atr.core.exceptions import ConfigurationError, LLMError


class TestOpenRouterLLM:
    """Tests for OpenRouterLLM adapter."""

    def test_init_defaults(self):
        """Default initialization."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        assert llm.model == "anthropic/claude-3-haiku"
        assert llm.api_key == "test-key"
        assert llm.base_url == "https://openrouter.ai/api/v1"

    def test_init_custom_model(self):
        """Custom model initialization."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="test-key")

        assert llm.model == "openai/gpt-4o-mini"

    def test_init_from_env(self):
        """API key from environment."""
        from atr.llm.openrouter import OpenRouterLLM

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
            llm = OpenRouterLLM()
            assert llm.api_key == "env-key"

    def test_missing_api_key_raises_on_use(self):
        """Missing API key raises ConfigurationError when client is created."""
        from atr.llm.openrouter import OpenRouterLLM

        with patch.dict("os.environ", {}, clear=True):
            # Remove the env var if it exists
            import os
            env_backup = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                llm = OpenRouterLLM(api_key=None)
                llm.api_key = None  # Force clear
                with pytest.raises(ConfigurationError):
                    llm._get_client()
            finally:
                if env_backup:
                    os.environ["OPENROUTER_API_KEY"] = env_backup

    def test_complete_calls_openai_client(self):
        """complete() calls the OpenAI client correctly."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "read_file\nwrite_file"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test prompt", system_prompt="System")

        assert result == "read_file\nwrite_file"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_complete_no_choices_raises(self):
        """complete() raises LLMError when no choices returned."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        with pytest.raises(LLMError):
            llm.complete("Test")

    def test_complete_api_error_raises_llm_error(self):
        """complete() wraps API errors in LLMError."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        llm._client = mock_client

        with pytest.raises(LLMError, match="API error"):
            llm.complete("Test")

    def test_complete_none_content_returns_empty(self):
        """complete() returns empty string when content is None."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_acomplete_calls_async_client(self):
        """acomplete() calls the async OpenAI client."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "read_file"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        llm._async_client = mock_client

        result = await llm.acomplete("Test prompt")
        assert result == "read_file"

    def test_complete_without_system_prompt(self):
        """complete() works without system prompt."""
        from atr.llm.openrouter import OpenRouterLLM

        llm = OpenRouterLLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestOpenAILLM:
    """Tests for OpenAILLM adapter."""

    def test_init_defaults(self):
        """Default initialization."""
        from atr.llm.openai import OpenAILLM

        llm = OpenAILLM(api_key="test-key")

        assert llm.model == "gpt-4o-mini"
        assert llm.api_key == "test-key"

    def test_init_custom_model(self):
        """Custom model initialization."""
        from atr.llm.openai import OpenAILLM

        llm = OpenAILLM(model="gpt-4o", api_key="test-key")

        assert llm.model == "gpt-4o"

    def test_complete_calls_client(self):
        """complete() calls the OpenAI client."""
        from atr.llm.openai import OpenAILLM

        llm = OpenAILLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "tool_name"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test")
        assert result == "tool_name"

    def test_complete_api_error_raises_llm_error(self):
        """complete() wraps errors in LLMError."""
        from atr.llm.openai import OpenAILLM

        llm = OpenAILLM(api_key="test-key")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Timeout")
        llm._client = mock_client

        with pytest.raises(LLMError, match="Timeout"):
            llm.complete("Test")

    @pytest.mark.asyncio
    async def test_acomplete_calls_async_client(self):
        """acomplete() calls the async client."""
        from atr.llm.openai import OpenAILLM

        llm = OpenAILLM(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        llm._async_client = mock_client

        result = await llm.acomplete("Test")
        assert result == "result"

    def test_missing_api_key_raises(self):
        """Missing API key raises ConfigurationError."""
        from atr.llm.openai import OpenAILLM

        import os
        env_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            llm = OpenAILLM(api_key=None)
            llm.api_key = None
            with pytest.raises(ConfigurationError):
                llm._get_client()
        finally:
            if env_backup:
                os.environ["OPENAI_API_KEY"] = env_backup


class TestAnthropicLLM:
    """Tests for AnthropicLLM adapter."""

    def test_init_defaults(self):
        """Default initialization."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        assert llm.model == "claude-3-haiku-20240307"
        assert llm.api_key == "test-key"
        assert llm.max_tokens == 1024

    def test_init_custom(self):
        """Custom initialization."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            max_tokens=2048,
        )

        assert llm.model == "claude-3-5-sonnet-20241022"
        assert llm.max_tokens == 2048

    def test_complete_calls_client(self):
        """complete() calls the Anthropic client."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        mock_block = MagicMock()
        mock_block.text = "tool_name"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test", system_prompt="System")

        assert result == "tool_name"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "System"

    def test_complete_without_system_prompt(self):
        """complete() works without system prompt."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        mock_block = MagicMock()
        mock_block.text = "result"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_complete_api_error_raises_llm_error(self):
        """complete() wraps errors in LLMError."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Rate limited")
        llm._client = mock_client

        with pytest.raises(LLMError, match="Rate limited"):
            llm.complete("Test")

    @pytest.mark.asyncio
    async def test_acomplete_calls_async_client(self):
        """acomplete() calls the async Anthropic client."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        mock_block = MagicMock()
        mock_block.text = "async_result"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        llm._async_client = mock_client

        result = await llm.acomplete("Test")
        assert result == "async_result"

    def test_missing_api_key_raises(self):
        """Missing API key raises ConfigurationError."""
        from atr.llm.anthropic import AnthropicLLM

        import os
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            llm = AnthropicLLM(api_key=None)
            llm.api_key = None
            with pytest.raises(ConfigurationError):
                llm._get_client()
        finally:
            if env_backup:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

    def test_complete_multiple_text_blocks(self):
        """complete() joins multiple text blocks."""
        from atr.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test-key")

        block1 = MagicMock()
        block1.text = "read_file\n"
        block2 = MagicMock()
        block2.text = "write_file"
        mock_response = MagicMock()
        mock_response.content = [block1, block2]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm._client = mock_client

        result = llm.complete("Test")
        assert result == "read_file\nwrite_file"
