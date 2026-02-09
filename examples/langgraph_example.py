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
from atr.adapters.langchain import create_async_router_node, filter_tools
from atr.llm import OpenAILLM


async def basic_example():
    """
    Basic example: Tool loading, routing, and filtering.
    """
    print("=== Basic LangGraph Example ===\n")

    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
        },
    }

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    print(f"Loaded {len(all_tools)} tools from MCP servers")

    # Create router and add tools
    llm = OpenAILLM(model="gpt-4o-mini")
    router = ToolRouter(llm=llm, max_tools=5)
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    # Route queries
    queries = [
        "Read the contents of /private/tmp/test.txt",
        "List files in /private/tmp",
        "Write 'hello' to /private/tmp/greeting.txt",
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
        from typing import Any, TypedDict

        from langgraph.graph import StateGraph
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
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
        },
    }

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()

    # Create router
    llm = OpenAILLM(model="gpt-4o-mini")
    router = ToolRouter(llm=llm, max_tools=5)
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    # Create async routing node
    routing_node = create_async_router_node(router, all_tools)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("route_tools", routing_node)
    graph.set_entry_point("route_tools")
    graph.set_finish_point("route_tools")

    app = graph.compile()

    # Test the graph
    result = await app.ainvoke({"query": "Read a file from /private/tmp"})
    print("Query: Read a file from /private/tmp")
    print(f"Routed tools: {[t.name for t in result['tools']]}")


async def agent_with_routing_example():
    """
    Full agent example: query -> route tools -> execute with filtered tools.

    Shows the real value of ATR — an agent receives a query, ATR dynamically
    selects only the relevant tools from a large pool, and the agent executes
    the task with a focused toolset via langchain's create_agent.

    Requirements:
        pip install atr[langgraph] langchain-openai
    """
    print("=== Agent with Dynamic Tool Routing ===\n")

    try:
        from langchain.agents import (
            create_agent,  # pyright: ignore[reportMissingImports]
        )
        from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("This example requires langchain, langgraph and langchain-openai.")
        print("Install with: pip install langchain langgraph langchain-openai")
        return

    # --- Step 1: Load tools from multiple MCP servers ---
    # Having many tools from different servers is the scenario where ATR shines.
    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
        },
        "fetch": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    }

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    print(f"Total tools loaded from MCP servers: {len(all_tools)}")
    print(f"All tools: {[t.name for t in all_tools]}\n")

    # --- Step 2: Set up the ATR router ---
    routing_llm = OpenAILLM(model="gpt-4o-mini")
    router = ToolRouter(llm=routing_llm, max_tools=5)
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    # --- Step 3: Agent LLM (the one that actually reasons and calls tools) ---
    agent_llm = ChatOpenAI(model="gpt-4o")

    # --- Step 4: Run queries through route -> filter -> agent pipeline ---
    queries = [
        "List all files in /private/tmp and read the contents of any .txt files you find",
        "Fetch the content from https://httpbin.org/json and summarize it",
        "Create a file /private/tmp/atr_demo.txt with the text 'Hello from ATR!'",
    ]

    for query in queries:
        print(f"{'=' * 60}")
        print(f"User Query: {query}\n")

        # Route: ATR selects only the relevant tools for this query
        filtered_specs = await router.aroute(query)
        filtered_tools = filter_tools(all_tools, filtered_specs)

        print(
            f"ATR selected {len(filtered_tools)}/{len(all_tools)} tools: "
            f"{[t.name for t in filtered_tools]}"
        )

        # Create an agent with ONLY the filtered tools
        agent = create_agent(agent_llm, filtered_tools)

        # Execute the task
        print("Agent executing...\n")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

        # Print the final agent response
        last_message = result["messages"][-1]
        print(f"Agent Response:\n{last_message.content}\n")


async def agent_with_routing_graph_example():
    """
    Graph-based agent: routing node feeds into a react agent node.

    This wires ATR routing directly into the LangGraph state machine so
    the route -> agent flow is a single compiled graph.

    Requirements:
        pip install atr[langgraph] langchain-openai
    """
    print("=== Graph-Based Agent with Routing Node ===\n")

    try:
        from typing import Annotated, Any

        from langchain_core.messages import AnyMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode
    except ImportError:
        print("This example requires langgraph and langchain-openai.")
        print("Install with: pip install langgraph langchain-openai")
        return

    from typing import TypedDict

    # --- State definition ---
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        tools: list[Any]
        query: str

    # --- Setup servers, router, model ---
    server_configs = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
        },
        "fetch": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    }

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()

    routing_llm = OpenAILLM(model="gpt-4o-mini")
    router = ToolRouter(llm=routing_llm, max_tools=5)
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    agent_llm = ChatOpenAI(model="gpt-4o")

    # --- Node 1: Route tools based on query ---
    async def route_tools_node(state: AgentState) -> dict:
        query = state["query"]
        filtered_specs = await router.aroute(query)
        filtered = filter_tools(all_tools, filtered_specs)
        print(f"Routed to {len(filtered)} tools: {[t.name for t in filtered]}")
        return {"tools": filtered}

    # --- Node 2: Call the LLM with filtered tools ---
    async def call_model_node(state: AgentState) -> dict:
        tools = state.get("tools", all_tools)
        llm_with_tools = agent_llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # --- Node 3: Execute tool calls ---
    async def tool_executor_node(state: AgentState) -> dict:
        tools = state.get("tools", all_tools)
        tool_node = ToolNode(tools)
        return await tool_node.ainvoke(state)

    # --- Conditional edge: should we call tools or finish? ---
    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # --- Build the graph ---
    graph = StateGraph(AgentState)
    graph.add_node("route_tools", route_tools_node)
    graph.add_node("agent", call_model_node)
    graph.add_node("tools", tool_executor_node)

    graph.set_entry_point("route_tools")
    graph.add_edge("route_tools", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    app = graph.compile()

    # --- Run it ---
    query = "List the files in /private/tmp directory"
    print(f"Query: {query}\n")

    result = await app.ainvoke(
        {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "tools": [],
        }
    )

    last_message = result["messages"][-1]
    print(f"\nAgent Response:\n{last_message.content}")


async def main():
    """Run all examples."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable to run this example")
        return

    await basic_example()
    print("\n" + "=" * 50 + "\n")
    await langgraph_node_example()
    print("\n" + "=" * 50 + "\n")
    await agent_with_routing_example()
    print("\n" + "=" * 50 + "\n")
    await agent_with_routing_graph_example()


if __name__ == "__main__":
    asyncio.run(main())
