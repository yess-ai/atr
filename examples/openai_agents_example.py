"""
OpenAI Integration Example

This example demonstrates using ATR with OpenAI-format tools.
It shows how to:
1. Convert OpenAI function definitions to ToolSpecs
2. Route queries to filter relevant tools

Requirements:
    pip install atr
"""

import asyncio
import os

from atr import ToolRouter
from atr.adapters.openai import OpenAIAdapter, filter_tools
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


async def main():
    """Run the example."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY environment variable to run this example")
        return

    print("=== OpenAI Tools Routing Example ===\n")

    tools = create_sample_tools()

    # Create router and add tools
    llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
    router = ToolRouter(llm=llm, max_tools=3)
    router.add_tools(OpenAIAdapter.to_specs(tools))

    print(f"Loaded {len(tools)} tools\n")

    # Test queries
    queries = [
        "What's the weather like in New York?",
        "Send an email to john@example.com",
        "Check the stock price of AAPL",
        "Schedule a meeting for tomorrow",
    ]

    for query in queries:
        filtered_specs = await router.aroute(query)
        filtered_tools = filter_tools(tools, filtered_specs)

        print(f"Query: {query}")
        print(f"Filtered tools: {[t['function']['name'] for t in filtered_tools]}\n")


if __name__ == "__main__":
    asyncio.run(main())
