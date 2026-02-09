"""
OpenAI + MCP Integration Example

This example demonstrates using ATR with OpenAI models and MCP servers.
It shows how to:
1. Load tools from MCP servers
2. Convert MCP tools to ToolSpecs for routing
3. Route queries to filter relevant tools
4. Convert filtered tools back to OpenAI function format

Requirements:
    pip install atr[openai,mcp]
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from atr import ToolRouter
from atr.adapters.mcp import MCPAdapter
from atr.llm import OpenAILLM


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """Convert MCP tools to OpenAI function definitions."""
    openai_tools = []
    for tool in mcp_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        description = getattr(tool, "description", "") or ""
        input_schema = getattr(tool, "inputSchema", None)
        parameters = {}
        if input_schema is not None:
            if hasattr(input_schema, "model_dump"):
                parameters = input_schema.model_dump()
            elif isinstance(input_schema, dict):
                parameters = input_schema

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return openai_tools


async def basic_mcp_routing():
    """
    Basic example: Load MCP tools, route with ATR, filter for OpenAI usage.
    """
    print("=== Basic MCP + OpenAI Routing Example ===\n")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            mcp_tools = result.tools

            print(f"Loaded {len(mcp_tools)} tools from MCP server")
            print(f"Tools: {[t.name for t in mcp_tools]}\n")

            # Create router with OpenAI LLM
            llm = OpenAILLM(model="gpt-4o-mini")
            router = ToolRouter(llm=llm, max_tools=3)
            router.add_tools(MCPAdapter.to_specs(mcp_tools))

            # Route queries
            queries = [
                "Read the contents of /private/tmp/test.txt",
                "List files in /private/tmp",
                "Write 'hello world' to /private/tmp/greeting.txt",
                "Search for files matching *.log",
            ]

            for query in queries:
                filtered_specs = await router.aroute(query)
                filtered_mcp = MCPAdapter.filter_tools(mcp_tools, filtered_specs)

                # Convert filtered tools to OpenAI format
                openai_tools = mcp_tools_to_openai(filtered_mcp)

                print(f"Query: {query}")
                print(f"Filtered tools: {[t['function']['name'] for t in openai_tools]}\n")


async def multi_server_routing():
    """
    Advanced example: Multiple MCP servers, route and filter to OpenAI format.
    """
    print("=== Multi-Server MCP + OpenAI Routing ===\n")

    servers = {
        "filesystem": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
        ),
        "fetch": StdioServerParameters(
            command="uvx",
            args=["mcp-server-fetch"],
        ),
    }

    all_mcp_tools = []
    sessions = []

    # Gather tools from all servers
    for name, params in servers.items():
        read, write = await stdio_client(params).__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        result = await session.list_tools()
        all_mcp_tools.extend(result.tools)
        sessions.append(session)
        print(f"[{name}] Loaded {len(result.tools)} tools")

    print(f"\nTotal: {len(all_mcp_tools)} tools across {len(servers)} servers\n")

    # Create router
    llm = OpenAILLM(model="gpt-4o-mini")
    router = ToolRouter(llm=llm, max_tools=3)
    router.add_tools(MCPAdapter.to_specs(all_mcp_tools))

    # Route queries
    queries = [
        "List all files in /private/tmp",
        "Fetch the content from https://httpbin.org/json",
        "Read /private/tmp/notes.txt and fetch https://example.com",
    ]

    for query in queries:
        filtered_specs = await router.aroute(query)
        filtered_mcp = MCPAdapter.filter_tools(all_mcp_tools, filtered_specs)
        openai_tools = mcp_tools_to_openai(filtered_mcp)

        print(f"Query: {query}")
        print(
            f"Selected {len(openai_tools)}/{len(all_mcp_tools)} tools: "
            f"{[t['function']['name'] for t in openai_tools]}\n"
        )


async def openai_chat_with_routing():
    """
    Full example: Route MCP tools, then use them with OpenAI chat completions.
    """
    print("=== OpenAI Chat + MCP Tool Routing ===\n")

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("This example requires openai. Install with: pip install openai")
        return

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/private/tmp"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            mcp_tools = result.tools

            # Route
            llm = OpenAILLM(model="gpt-4o-mini")
            router = ToolRouter(llm=llm, max_tools=3)
            router.add_tools(MCPAdapter.to_specs(mcp_tools))

            query = "List all files in /private/tmp"
            filtered_specs = await router.aroute(query)
            filtered_mcp = MCPAdapter.filter_tools(mcp_tools, filtered_specs)
            openai_tools = mcp_tools_to_openai(filtered_mcp)

            print(f"Query: {query}")
            print(f"Routed to: {[t['function']['name'] for t in openai_tools]}\n")

            # Use with OpenAI chat completions
            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": query}],
                tools=openai_tools,  # Only the routed tools!
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Tool call: {tc.function.name}({tc.function.arguments})")

                    # Execute via MCP session
                    import json

                    tool_result = await session.call_tool(
                        tc.function.name, json.loads(tc.function.arguments)
                    )
                    print(
                        f"Result: {tool_result.content[0].text[:200]}...\n"
                        if len(tool_result.content[0].text) > 200
                        else f"Result: {tool_result.content[0].text}\n"
                    )
            else:
                print(f"Response: {msg.content}")


async def main():
    """Run all examples."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable to run this example")
        return

    await basic_mcp_routing()
    print("\n" + "=" * 50 + "\n")
    await openai_chat_with_routing()


if __name__ == "__main__":
    asyncio.run(main())
