"""
OpenAI Integration Example

This example demonstrates using ATR with OpenAI-format tools.
It shows how to:
1. Convert OpenAI function definitions to ToolSpecs
2. Route queries to filter relevant tools
3. Use OpenAIRouter for high-level integration

Requirements:
    pip install atr
"""

import asyncio
import os

from atr import ToolRouter
from atr.adapters.openai import OpenAIAdapter, OpenAIRouter
from atr.llm import OpenRouterLLM


def create_sample_tools() -> list[dict]:
    """Create sample OpenAI function definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Email recipient"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": "Create a new calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                    },
                    "required": ["title", "date"],
                },
            },
        },
    ]


def manual_routing_example():
    """
    Example: Manual tool conversion and routing.
    """
    print("=== Manual Routing Example ===\n")

    tools = create_sample_tools()

    # Convert to ToolSpecs
    specs = OpenAIAdapter.to_specs(tools)
    print(f"Converted {len(specs)} tools to ToolSpecs")

    # Create router
    llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
    router = ToolRouter(llm=llm, max_tools=3)
    router.add_tools(specs)

    # Test queries
    queries = [
        "What's the weather like in New York?",
        "Send an email to john@example.com",
        "Check the stock price of AAPL",
        "Schedule a meeting for tomorrow",
    ]

    from atr.core.tool import ToolCollection

    for query in queries:
        filtered_specs = router.route(query)
        filtered_tools = OpenAIAdapter.filter_tools(tools, filtered_specs)

        print(f"Query: {query}")
        print(f"Filtered tools: {[t['function']['name'] for t in filtered_tools]}\n")


async def high_level_router_example():
    """
    Example: Using OpenAIRouter for simpler integration.
    """
    print("=== High-Level Router Example ===\n")

    tools = create_sample_tools()

    # Create high-level router
    router = OpenAIRouter(
        llm=OpenRouterLLM(model="anthropic/claude-3-haiku"),
        tools=tools,
        max_tools=3,
    )

    # Route queries
    queries = [
        "What's the weather in San Francisco?",
        "Find information about Python programming",
        "Get Tesla's stock price",
    ]

    for query in queries:
        filtered = await router.aroute(query)
        print(f"Query: {query}")
        print(f"Filtered tools: {[t['function']['name'] for t in filtered]}\n")


async def main():
    """Run all examples."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY environment variable to run this example")
        return

    manual_routing_example()
    print("\n" + "=" * 50 + "\n")
    await high_level_router_example()


if __name__ == "__main__":
    asyncio.run(main())
