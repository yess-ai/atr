"""
LangGraph Integration Example

This example demonstrates using ATR with LangGraph and LangChain MCP tools.
It shows how to:
1. Load tools from MCP servers using langchain-mcp-adapters
2. Convert them to ToolSpecs for routing
3. Filter tools based on user query
4. Create a LangGraph routing node

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
from atr.adapters.langchain import create_router_node, filter_tools
from atr.llm import OpenRouterLLM


async def basic_example():
    """
    Basic example: Tool loading, routing, and filtering.
    """
    print("=== Basic LangGraph Example ===\n")

    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
        },
    }

    async with MultiServerMCPClient(server_configs) as client:
        all_tools = await client.get_tools()
        print(f"Loaded {len(all_tools)} tools from MCP servers")

        # Create router and add tools
        llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
        router = ToolRouter(llm=llm, max_tools=5)
        router.add_tools(LangChainAdapter.to_specs(all_tools))

        # Route queries
        queries = [
            "Read the contents of /tmp/test.txt",
            "List files in /tmp",
            "Write 'hello' to /tmp/greeting.txt",
        ]

        for query in queries:
            filtered_specs = await router.aroute(query)
            filtered_lc_tools = filter_tools(all_tools, filtered_specs)
            print(f"Query: {query}")
            print(f"Filtered tools: {[t.name for t in filtered_lc_tools]}\n")


async def langgraph_node_example():
    """
    Example: Creating a LangGraph node for dynamic routing.
    """
    print("=== LangGraph Node Example ===\n")

    try:
        from langgraph.graph import StateGraph
        from typing import TypedDict, Any
    except ImportError:
        print("This example requires langgraph. Install with: pip install langgraph")
        return

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
        llm = OpenRouterLLM(model="anthropic/claude-3-haiku")
        router = ToolRouter(llm=llm, max_tools=5)
        router.add_tools(LangChainAdapter.to_specs(all_tools))

        # Create routing node
        routing_node = create_router_node(router, all_tools)

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

    await basic_example()
    print("\n" + "=" * 50 + "\n")
    await langgraph_node_example()


if __name__ == "__main__":
    asyncio.run(main())
