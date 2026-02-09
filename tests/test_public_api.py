"""Tests for public API exports."""

from __future__ import annotations

import pytest


class TestMainImports:
    """Test that expected items are importable from main package."""

    def test_import_toolrouter(self):
        """ToolRouter is importable from atr."""
        from atr import ToolRouter

        assert ToolRouter is not None

    def test_import_toolspec(self):
        """ToolSpec is importable from atr."""
        from atr import ToolSpec

        assert ToolSpec is not None

    def test_import_toolcollection(self):
        """ToolCollection is importable from atr."""
        from atr import ToolCollection

        assert ToolCollection is not None

    def test_import_version(self):
        """__version__ is importable from atr."""
        from atr import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestStrategyImports:
    """Test imports for strategy classes."""

    def test_import_filter_strategy_from_main(self):
        """FilterStrategy is importable from atr."""
        from atr import FilterStrategy

        assert FilterStrategy is not None

    def test_import_base_filter_strategy(self):
        """BaseFilterStrategy is importable from atr."""
        from atr import BaseFilterStrategy

        assert BaseFilterStrategy is not None

    def test_import_llm_filter_strategy(self):
        """LLMFilterStrategy is importable from atr."""
        from atr import LLMFilterStrategy

        assert LLMFilterStrategy is not None

    def test_import_passthrough_strategy(self):
        """PassthroughStrategy is importable from atr."""
        from atr import PassthroughStrategy

        assert PassthroughStrategy is not None

    def test_import_strategies_from_core(self):
        """Strategies are importable from atr.core.strategies."""
        from atr.core.strategies import (
            BaseFilterStrategy,
            FilterStrategy,
            LLMFilterStrategy,
            PassthroughStrategy,
        )

        assert FilterStrategy is not None
        assert BaseFilterStrategy is not None
        assert LLMFilterStrategy is not None
        assert PassthroughStrategy is not None


class TestExceptionImports:
    """Test imports for exception classes."""

    def test_import_atr_error(self):
        """ATRError is importable from atr."""
        from atr import ATRError

        assert ATRError is not None
        assert issubclass(ATRError, Exception)

    def test_import_routing_error(self):
        """RoutingError is importable from atr."""
        from atr import RoutingError

        assert RoutingError is not None

    def test_import_llm_error(self):
        """LLMError is importable from atr."""
        from atr import LLMError

        assert LLMError is not None

    def test_import_adapter_error(self):
        """AdapterError is importable from atr."""
        from atr import AdapterError

        assert AdapterError is not None

    def test_import_configuration_error(self):
        """ConfigurationError is importable from atr."""
        from atr import ConfigurationError

        assert ConfigurationError is not None

    def test_import_exceptions_from_core(self):
        """Exceptions are importable from atr.core.exceptions."""
        from atr.core.exceptions import (
            ATRError,
            AdapterError,
            ConfigurationError,
            LLMError,
            RoutingError,
        )

        assert ATRError is not None
        assert RoutingError is not None
        assert LLMError is not None
        assert AdapterError is not None
        assert ConfigurationError is not None


class TestLLMImports:
    """Test imports from atr.llm subpackage."""

    def test_import_routing_llm_protocol(self):
        """RoutingLLM protocol is importable."""
        from atr.llm import RoutingLLM

        assert RoutingLLM is not None


class TestAllExports:
    """Test that __all__ contains expected exports."""

    def test_all_contains_core_classes(self):
        """__all__ includes core classes."""
        import atr

        assert "ToolRouter" in atr.__all__
        assert "ToolSpec" in atr.__all__
        assert "ToolCollection" in atr.__all__

    def test_all_contains_version(self):
        """__all__ includes __version__."""
        import atr

        assert "__version__" in atr.__all__

    def test_all_contains_strategies(self):
        """__all__ includes strategy classes."""
        import atr

        assert "FilterStrategy" in atr.__all__
        assert "BaseFilterStrategy" in atr.__all__
        assert "LLMFilterStrategy" in atr.__all__
        assert "PassthroughStrategy" in atr.__all__

    def test_all_contains_exceptions(self):
        """__all__ includes exception classes."""
        import atr

        assert "ATRError" in atr.__all__
        assert "RoutingError" in atr.__all__
        assert "LLMError" in atr.__all__
        assert "AdapterError" in atr.__all__
        assert "ConfigurationError" in atr.__all__
