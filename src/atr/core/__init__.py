"""Core ATR components."""

from atr.core.exceptions import (
    ATRError,
    AdapterError,
    ConfigurationError,
    LLMError,
    RoutingError,
)
from atr.core.router import ToolRouter
from atr.core.strategies import (
    BaseFilterStrategy,
    FilterStrategy,
    LLMFilterStrategy,
    PassthroughStrategy,
)
from atr.core.tool import ToolCollection, ToolSpec

__all__ = [
    # Data structures
    "ToolSpec",
    "ToolCollection",
    # Router
    "ToolRouter",
    # Strategies
    "FilterStrategy",
    "BaseFilterStrategy",
    "LLMFilterStrategy",
    "PassthroughStrategy",
    # Exceptions
    "ATRError",
    "RoutingError",
    "LLMError",
    "AdapterError",
    "ConfigurationError",
]
