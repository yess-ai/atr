"""
Agno + MCP + ATR Integration Example

Shows how to use ATR with Agno agents. The pattern is simple:
1. Create toolkit (same as you would without ATR)
2. Pass toolkit directly to AgnoAdapter.to_specs()
3. Route with ATR to filter tools
4. Pass filtered tools directly to Agent

No need to call .get_functions() or manipulate tool definitions!

Requirements:
    pip install atr[agno] agno
"""

import asyncio
import os


async def example_with_mcp():
    """
    Example: Using ATR with Agno MCPTools.

    Pass MCPTools directly to the adapter - no need to extract functions.
    """
    print("=" * 60)
    print("Example 1: MCP Tools")
    print("=" * 60 + "\n")

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.tools.mcp import MCPTools
    except ImportError:
        print("This example requires agno. Install with: pip install agno")
        return

    from atr import ToolRouter
    from atr.adapters.agno import AgnoAdapter, filter_tools
    from atr.llm import OpenAILLM

    async with MCPTools(
        command="uvx mcp-server-git",
    ) as mcp:
        # Pass MCPTools directly - no need to call .get_functions()
        print(f"Discovered {len(mcp.get_functions())} tools")

        # Convert to specs and create router
        router = ToolRouter(llm=OpenAILLM(), max_tools=5)
        router.add_tools(AgnoAdapter.to_specs([mcp]))  # Pass toolkit directly!

        # Route query
        query = "What is my git status of atr repository in this path: /Users/guy.yanko/yess/dev/product-development/atr ?"
        filtered_specs = await router.aroute(query)
        print(f"\nQuery: '{query}'")
        print(f"ATR selected: {filtered_specs.names}")

        # Filter and pass directly to Agent
        filtered_tools = filter_tools([mcp], filtered_specs)
        print(f"Agent receives {len(filtered_tools)} tools\n")

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=filtered_tools,  # Direct use - no conversion needed!
            markdown=True,
        )

        response = await agent.arun(query)
        print(response.content)


async def example_with_toolkit():
    """
    Example: Using ATR with Agno toolkit (no MCP).

    Pass the toolkit directly - same format you'd use with Agent(tools=[toolkit]).
    """
    print("\n" + "=" * 60)
    print("Example 2: YFinance Toolkit")
    print("=" * 60 + "\n")

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.tools.yfinance import YFinanceTools
    except ImportError:
        print("This example requires agno with yfinance.")
        return

    from atr import ToolRouter
    from atr.adapters.agno import AgnoAdapter, filter_tools
    from atr.llm import OpenAILLM

    # Test query with real agent
    query = "What is AAPL's current stock price and what do analysts recommend?"
    # Create toolkit - same as you would without ATR
    toolkit = YFinanceTools()
    print(f"Toolkit has {len(toolkit.get_functions())} tools")

    # Pass toolkit directly to adapter - no .get_functions() needed!
    router = ToolRouter(llm=OpenAILLM(), max_tools=3)
    router.add_tools(AgnoAdapter.to_specs([toolkit]))

    # Route to get relevant tools
    filtered_specs = await router.aroute(query)

    # Filter and pass directly to Agent - no conversion needed!
    filtered_tools = filter_tools([toolkit], filtered_specs)

    print(f"Query: '{query}'")
    print(
        f"ATR selected: {[getattr(f, 'name', getattr(f, '__name__', '?')) for f in filtered_tools]}"
    )
    print(f"Agent receives {len(filtered_tools)} tools (not {len(toolkit.get_functions())})\n")

    # Create agent with filtered tools - direct use!
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=filtered_tools,
        markdown=True,
    )

    # Run the agent
    response = await agent.arun(query)
    print(response.content)


async def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY environment variable")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    try:
        await example_with_mcp()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        await example_with_toolkit()
    except Exception as e:
        print(f"Example 2 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
