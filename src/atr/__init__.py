"""
ATR - Adaptive Tool Routing

Dynamic tool selection for AI agents. ATR filters tools based on user queries
before they reach the agent's system prompt, reducing context tokens by ~90%
and improving tool selection accuracy.

Example:
    ```python
    from atr import ToolRouter, ToolSpec
    from atr.llm import OpenRouterLLM

    # Create router with LLM
    router = ToolRouter(llm=OpenRouterLLM())

    # Add tools
    router.add_tools([
        ToolSpec(name="get_stock_price", description="Get current stock price"),
        ToolSpec(name="get_company_news", description="Get company news articles"),
        ToolSpec(name="get_weather", description="Get weather for a location"),
    ])

    # Route query to filter tools
    filtered = router.route("What is AAPL's stock price?")
    print(filtered.names)  # {'get_stock_price'}
    ```

For framework-specific integrations, see:
- `atr.adapters.langchain` - LangChain/LangGraph integration
- `atr.adapters.agno` - Agno integration
- `atr.adapters.openai` - OpenAI Agents SDK integration
- `atr.adapters.litellm` - LiteLLM integration
"""

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

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ToolRouter",
    "ToolSpec",
    "ToolCollection",
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
