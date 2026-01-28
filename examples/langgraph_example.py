"""
LangGraph Integration Example

This example demonstrates using ATR with LangGraph and LangChain MCP tools.
It shows how to:
1. Load tools from MCP servers using langchain-mcp-adapters
2. Convert them to ToolSpecs for routing
3. Filter tools based on user query
4. Use filtered tools with a LangGraph agent

Requirements:
    pip install atr[langgraph]
    # Or: pip install langgraph langchain-mcp-adapters langchain-core
"""

import asyncio
import os

# Check for required packages
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    print("This example requires langchain-mcp-adapters.")
    print("Install with: pip install atr[langgraph]")
    exit(1)

from atr import ToolRouter
from atr.adapters import LangChainAdapter
from atr.integrations.langgraph import LangGraphRouter, filter_tools
from atr.llm import OpenRouterLLM


async def basic_langgraph_example():
    """
    Basic example: Manual tool loading and filtering.
    """
    print("=== Basic LangGraph Example ===\n")

    # Configure MCP servers
    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
    }

    # Load tools from MCP servers
    async with MultiServerMCPClient(server_configs) as client:
        all_tools = await client.get_tools()
        print(f"Loaded {len(all_tools)} tools from MCP servers")

        # Create router and add tools
        llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
        router = ToolRouter(llm=llm, max_tools=5)
        router.add_tools(LangChainAdapter.to_specs(all_tools))

        # Route query
        query = "Read the contents of /tmp/test.txt"
        filtered_specs = router.route(query)
        print(f"\nQuery: {query}")
        print(f"Filtered tool names: {filtered_specs.names}")

        # Filter original LangChain tools
        filtered_lc_tools = filter_tools(all_tools, filtered_specs)
        print(f"Filtered LangChain tools: {[t.name for t in filtered_lc_tools]}")


async def high_level_router_example():
    """
    High-level example: Using LangGraphRouter for simpler integration.
    """
    print("\n=== High-Level LangGraphRouter Example ===\n")

    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
    }

    async with MultiServerMCPClient(server_configs) as client:
        all_tools = await client.get_tools()

        # Create high-level router
        router = LangGraphRouter(
            llm=OpenRouterLLM(model="anthropic/claude-3-haiku"),
            tools=all_tools,
            max_tools=5,
        )

        # Route and get filtered LangChain tools directly
        queries = [
            "List files in /tmp",
            "Write 'hello' to /tmp/greeting.txt",
            "Search for Python files",
        ]

        for query in queries:
            filtered = await router.aroute(query)
            print(f"Query: {query}")
            print(f"Filtered tools: {[t.name for t in filtered]}\n")


async def langgraph_node_example():
    """
    Example: Creating a LangGraph node for dynamic routing.
    """
    print("\n=== LangGraph Node Example ===\n")

    try:
        from langgraph.graph import StateGraph
        from typing import TypedDict, Any
    except ImportError:
        print("This example requires langgraph. Install with: pip install langgraph")
        return

    # Define state
    class AgentState(TypedDict):
        query: str
        tools: list[Any]
        response: str

    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
    }

    async with MultiServerMCPClient(server_configs) as client:
        all_tools = await client.get_tools()

        # Create router
        router = LangGraphRouter(
            llm=OpenRouterLLM(model="anthropic/claude-3-haiku"),
            tools=all_tools,
        )

        # Create routing node
        routing_node = router.create_node(
            state_key="tools",
            query_key="query",
        )

        # Build graph
        graph = StateGraph(AgentState)
        graph.add_node("route_tools", routing_node)
        graph.set_entry_point("route_tools")
        graph.set_finish_point("route_tools")

        app = graph.compile()

        # Test the graph
        result = app.invoke({"query": "Read a file from /tmp"})
        print(f"Query: Read a file from /tmp")
        print(f"Routed tools: {[t.name for t in result['tools']]}")


async def main():
    """Run all examples."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY environment variable to run this example")
        return

    await basic_langgraph_example()
    await high_level_router_example()
    await langgraph_node_example()


if __name__ == "__main__":
    asyncio.run(main())
