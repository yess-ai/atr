"""
LiteLLM Integration Example

This example demonstrates using ATR as a LiteLLM callback hook that
transparently filters tools before they reach the model.

Unlike agent frameworks, LiteLLM is an LLM gateway — ATR plugs into its
callback system to intercept and prune tools on every completion call.
This is useful when you're using LiteLLM as your LLM layer (directly or
via the proxy) and want automatic tool routing without changing call sites.

Requirements:
    pip install atr[litellm]

Environment variables:
    OPENROUTER_API_KEY - Required for the routing LLM
    OPENAI_API_KEY     - Required for the main LLM call (via LiteLLM)
"""

import asyncio
import os


def create_sample_tools() -> list[dict]:
    """Create a bunch of tools in OpenAI format — enough to trigger routing."""
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
                            "description": "City and state, e.g. San Francisco, CA",
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
                        "query": {"type": "string", "description": "Search query"},
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
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
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
                        "symbol": {"type": "string", "description": "Ticker symbol"},
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
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file from the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]


async def hook_example():
    """
    End-to-end example: ATR hook filters tools transparently on a real
    litellm.acompletion() call.

    The hook intercepts the request, routes the tools based on the user
    query, and only sends the relevant subset to the model.
    """
    print("=== LiteLLM + ATR Hook Example ===\n")

    try:
        import litellm
    except ImportError:
        print("Install litellm first: pip install litellm")
        return

    from atr.adapters.litellm import create_hook

    # 1. Create ATR hook
    hook = create_hook(
        llm_provider="openrouter",
        llm_model="anthropic/claude-3-haiku",
        max_tools=3,
        min_tools_threshold=4,  # Only route when 4+ tools present
    )

    # 2. Register hook — LiteLLM will call it on every completion
    litellm.callbacks = [hook]

    tools = create_sample_tools()
    print(f"Registered {len(tools)} tools")
    print(f"Tool names: {[t['function']['name'] for t in tools]}\n")

    # 3. Make a normal litellm call — ATR filters tools automatically
    queries = [
        "What's the weather like in Tokyo?",
        "Send an email to bob@example.com about the meeting tomorrow",
    ]

    for query in queries:
        print(f"Query: {query}")

        response = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            tools=tools,  # Pass all tools — ATR prunes before they hit the model
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  Tool call: {tc.function.name}({tc.function.arguments})")
        else:
            print(f"  Response: {msg.content[:120]}")
        print()


def show_proxy_config():
    """Print example LiteLLM proxy YAML config using ATR as a hook."""
    print("=== LiteLLM Proxy Config ===\n")
    print(
        """\
# proxy_config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks:
    - atr.adapters.litellm.ATRToolRoutingHook
  atr_config:
    enabled: true
    max_tools: 10
    min_tools_threshold: 5
    llm_provider: openrouter
    llm_model: anthropic/claude-3-haiku

# Start: litellm --config proxy_config.yaml
# Then any client sending tools will get automatic ATR filtering.
"""
    )


async def main():
    has_router_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    if has_router_key and has_openai_key:
        await hook_example()
        print("=" * 50 + "\n")
        show_proxy_config()
    elif has_router_key:
        print("Set OPENAI_API_KEY to run the end-to-end hook example.\n")
        show_proxy_config()
    else:
        print(
            "Set OPENROUTER_API_KEY (for routing) and OPENAI_API_KEY "
            "(for the LLM call) to run the end-to-end example.\n"
        )
        show_proxy_config()


if __name__ == "__main__":
    asyncio.run(main())
