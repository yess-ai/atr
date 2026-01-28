"""
Basic ATR Usage Example

This example demonstrates the core ATR functionality without any framework
integrations. It shows how to:
1. Create a ToolRouter with an LLM
2. Add tools via ToolSpec
3. Route queries to filter relevant tools
"""

import asyncio
import os

from atr import ToolRouter, ToolSpec
from atr.llm import OpenRouterLLM


def create_sample_tools() -> list[ToolSpec]:
    """Create sample tool specifications for demonstration."""
    return [
        ToolSpec(
            name="get_current_stock_price",
            description="Get the current stock price for a given ticker symbol",
        ),
        ToolSpec(
            name="get_company_info",
            description="Get company information and description",
        ),
        ToolSpec(
            name="get_stock_fundamentals",
            description="Get fundamental data like market cap, P/E ratio, dividend yield",
        ),
        ToolSpec(
            name="get_income_statements",
            description="Get income statement data for a company",
        ),
        ToolSpec(
            name="get_key_financial_ratios",
            description="Get key financial ratios like ROE, ROA, debt-to-equity",
        ),
        ToolSpec(
            name="get_analyst_recommendations",
            description="Get analyst buy/sell recommendations for a stock",
        ),
        ToolSpec(
            name="get_company_news",
            description="Get recent news articles for a company",
        ),
        ToolSpec(
            name="get_technical_indicators",
            description="Get technical indicators like RSI, MACD, Bollinger Bands",
        ),
        ToolSpec(
            name="get_historical_stock_prices",
            description="Get historical price data for a stock",
        ),
    ]


def sync_example():
    """Synchronous routing example."""
    print("=== Synchronous Routing Example ===\n")

    # Create router with OpenRouter LLM (or use OpenAI/Anthropic)
    llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
    router = ToolRouter(llm=llm, max_tools=5)

    # Add tools
    tools = create_sample_tools()
    router.add_tools(tools)
    print(f"Added {len(tools)} tools to router\n")

    # Test queries
    queries = [
        "What is the current price of AAPL?",
        "What are analysts saying about NVDA?",
        "Show me Tesla's historical prices and technical indicators",
        "What's Google's P/E ratio and other fundamentals?",
    ]

    for query in queries:
        print(f"Query: {query}")
        filtered = router.route(query)
        print(f"Filtered tools: {filtered.names}\n")


async def async_example():
    """Asynchronous routing example."""
    print("=== Asynchronous Routing Example ===\n")

    llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
    router = ToolRouter(llm=llm, max_tools=5)
    router.add_tools(create_sample_tools())

    # Route multiple queries concurrently
    queries = [
        "Get stock price for MSFT",
        "Recent news about Amazon",
        "Technical analysis of Bitcoin",
    ]

    # Route all queries in parallel
    results = await asyncio.gather(*[router.aroute(q) for q in queries])

    for query, filtered in zip(queries, results):
        print(f"Query: {query}")
        print(f"Filtered tools: {filtered.names}\n")


def main():
    """Run the examples."""
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Note: Set OPENROUTER_API_KEY environment variable to run this example")
        print("You can also use OPENAI_API_KEY with OpenAILLM or ANTHROPIC_API_KEY with AnthropicLLM")
        return

    # Run sync example
    sync_example()

    # Run async example
    print("\n" + "=" * 50 + "\n")
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
