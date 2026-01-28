"""
Agno + MCP + ATR Integration Example

Shows how to use ATR with Agno agents. The pattern is simple:
1. Connect to MCP / create toolkit
2. Convert tools to specs using AgnoAdapter
3. Route with ATR to get filtered tool names
4. Filter original functions and pass to Agent

Requirements:
    pip install atr[agno] agno
"""

import asyncio
import os


async def example_with_adapter():
    """
    Recommended pattern: Use AgnoAdapter directly.
    """
    print("=" * 60)
    print("Example 1: Using AgnoAdapter (Recommended)")
    print("=" * 60 + "\n")

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.tools.mcp import MCPTools
    except ImportError:
        print("This example requires agno. Install with: pip install agno")
        return

    from atr import ToolRouter
    from atr.adapters import AgnoAdapter
    from atr.llm import OpenRouterLLM

    async with MCPTools(
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    ) as mcp:
        # Get all functions
        all_funcs = mcp.functions
        print(f"Discovered {len(all_funcs)} tools: {[f.name for f in all_funcs]}")

        # Convert to specs and create router
        specs = AgnoAdapter.to_specs(all_funcs)
        router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
        router.add_tools(specs)

        # Route query
        query = "List all files in /tmp"
        filtered = await router.aroute(query)
        print(f"\nQuery: '{query}'")
        print(f"ATR selected: {filtered.names}")

        # Filter functions and create agent
        filtered_funcs = AgnoAdapter.filter_tools(all_funcs, filtered)
        print(f"Agent receives {len(filtered_funcs)} tools (not {len(all_funcs)})\n")

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=filtered_funcs,
            markdown=True,
        )

        response = await agent.arun(query)
        print(response.content)


async def example_with_helpers():
    """
    Alternative: Use convenience helpers from integrations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Integration Helpers")
    print("=" * 60 + "\n")

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.tools.mcp import MCPTools
    except ImportError:
        print("This example requires agno. Install with: pip install agno")
        return

    from atr.integrations.agno import create_router, route_and_filter
    from atr.llm import OpenRouterLLM

    async with MCPTools(
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    ) as mcp:
        # Create router directly from MCPTools
        router = create_router(mcp, llm=OpenRouterLLM(), max_tools=5)

        # Route and filter in one call
        query = "Read a file"
        filtered_funcs = await route_and_filter(router, mcp, query)

        print(f"Query: '{query}'")
        print(f"Filtered to {len(filtered_funcs)} tools: {[f.name for f in filtered_funcs]}\n")

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=filtered_funcs,
            markdown=True,
        )

        response = await agent.arun(query)
        print(response.content)


async def example_with_toolkit():
    """
    Example with Agno toolkit (no MCP).
    """
    print("\n" + "=" * 60)
    print("Example 3: YFinance Toolkit")
    print("=" * 60 + "\n")

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.tools.yfinance import YFinanceTools
    except ImportError:
        print("This example requires agno with yfinance.")
        return

    from atr import ToolRouter
    from atr.adapters import AgnoAdapter
    from atr.llm import OpenRouterLLM

    # Create toolkit
    toolkit = YFinanceTools(
        stock_price=True,
        company_info=True,
        stock_fundamentals=True,
        analyst_recommendations=True,
        company_news=True,
        technical_indicators=True,
        historical_prices=True,
    )

    all_funcs = toolkit.functions
    print(f"Toolkit has {len(all_funcs)} tools")

    # Create router
    specs = AgnoAdapter.to_specs(all_funcs)
    router = ToolRouter(llm=OpenRouterLLM(), max_tools=3)
    router.add_tools(specs)

    # Test queries
    queries = [
        "What is AAPL's stock price?",
        "What do analysts think about NVDA?",
        "Show Tesla's technical indicators",
    ]

    for query in queries:
        filtered = await router.aroute(query)
        filtered_funcs = AgnoAdapter.filter_tools(all_funcs, filtered)
        print(f"'{query}' -> {[f.name for f in filtered_funcs]}")


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY environment variable")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    try:
        await example_with_adapter()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        await example_with_helpers()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        await example_with_toolkit()
    except Exception as e:
        print(f"Example 3 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
