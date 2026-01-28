"""Tests for custom ATR exceptions."""

from __future__ import annotations

import pytest

from atr.core.exceptions import (
    ATRError,
    AdapterError,
    ConfigurationError,
    LLMError,
    RoutingError,
)


class TestExceptionHierarchy:
    """Test exception classes exist and have proper hierarchy."""

    def test_atr_error_exists(self):
        """Base ATRError exception exists."""
        assert ATRError is not None
        assert issubclass(ATRError, Exception)

    def test_routing_error_inherits_from_atr_error(self):
        """RoutingError inherits from ATRError."""
        assert issubclass(RoutingError, ATRError)
        assert issubclass(RoutingError, Exception)

    def test_llm_error_inherits_from_atr_error(self):
        """LLMError inherits from ATRError."""
        assert issubclass(LLMError, ATRError)
        assert issubclass(LLMError, Exception)

    def test_adapter_error_inherits_from_atr_error(self):
        """AdapterError inherits from ATRError."""
        assert issubclass(AdapterError, ATRError)
        assert issubclass(AdapterError, Exception)

    def test_configuration_error_inherits_from_atr_error(self):
        """ConfigurationError inherits from ATRError."""
        assert issubclass(ConfigurationError, ATRError)
        assert issubclass(ConfigurationError, Exception)


class TestExceptionUsage:
    """Test exceptions can be raised and caught."""

    def test_atr_error_with_message(self):
        """ATRError can be raised with message."""
        with pytest.raises(ATRError) as exc_info:
            raise ATRError("Base error message")

        assert "Base error message" in str(exc_info.value)

    def test_routing_error_with_message(self):
        """RoutingError can be raised with message."""
        with pytest.raises(RoutingError) as exc_info:
            raise RoutingError("Failed to route query")

        assert "Failed to route query" in str(exc_info.value)

    def test_llm_error_with_message(self):
        """LLMError can be raised with message."""
        with pytest.raises(LLMError) as exc_info:
            raise LLMError("API timeout")

        assert "timeout" in str(exc_info.value).lower()

    def test_adapter_error_with_message(self):
        """AdapterError can be raised with message."""
        with pytest.raises(AdapterError) as exc_info:
            raise AdapterError("Failed to convert tool")

        assert "convert" in str(exc_info.value).lower()

    def test_configuration_error_with_message(self):
        """ConfigurationError can be raised with message."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid configuration")

        assert "Invalid configuration" in str(exc_info.value)


class TestExceptionCatching:
    """Test exception catching behavior."""

    def test_catch_routing_error_as_atr_error(self):
        """RoutingError can be caught as ATRError."""
        caught = False
        try:
            raise RoutingError("Test")
        except ATRError:
            caught = True

        assert caught

    def test_catch_llm_error_as_atr_error(self):
        """LLMError can be caught as ATRError."""
        caught = False
        try:
            raise LLMError("Test")
        except ATRError:
            caught = True

        assert caught

    def test_catch_adapter_error_as_atr_error(self):
        """AdapterError can be caught as ATRError."""
        caught = False
        try:
            raise AdapterError("Test")
        except ATRError:
            caught = True

        assert caught

    def test_catch_configuration_error_as_atr_error(self):
        """ConfigurationError can be caught as ATRError."""
        caught = False
        try:
            raise ConfigurationError("Test")
        except ATRError:
            caught = True

        assert caught

    def test_catch_specific_error_not_other(self):
        """Specific exception types are distinct."""
        with pytest.raises(RoutingError):
            try:
                raise RoutingError("Test")
            except LLMError:
                pass  # Should not catch
