"""
LiteLLM Integration Example

This example demonstrates using ATR with LiteLLM as a custom hook
for automatic tool routing.

It shows how to:
1. Create and configure the ATR hook programmatically
2. Add the hook to LiteLLM callbacks
3. Use with the LiteLLM proxy (config example)

Requirements:
    pip install atr[litellm]
    # Or: pip install litellm

Environment variables:
    OPENROUTER_API_KEY - Required for routing (uses OpenRouter by default)
    OPENAI_API_KEY - Or use OpenAI for routing (set llm_provider="openai")
"""

import asyncio
import os


def create_sample_tools() -> list[dict]:
    """Create sample tools in OpenAI format (used by LiteLLM)."""
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


async def programmatic_example():
    """
    Example: Using ATR hook programmatically with LiteLLM.

    This shows how to add the ATR hook to filter tools before
    they are sent to the model.
    """
    print("=== Programmatic LiteLLM Hook Example ===\n")

    try:
        import litellm
    except ImportError:
        print("Please install litellm: pip install litellm")
        return

    from atr.adapters.litellm import create_hook

    # Create hook with custom configuration
    hook = create_hook(
        llm_provider="openrouter",
        llm_model="anthropic/claude-3-haiku",
        max_tools=3,
        min_tools_threshold=4,  # Only route if 4+ tools
    )

    # Add to LiteLLM callbacks
    litellm.callbacks = [hook]

    print("Hook configured with:")
    print(f"  - Provider: {hook.llm_provider}")
    print(f"  - Model: {hook.llm_model}")
    print(f"  - Max tools: {hook.max_tools}")
    print(f"  - Min threshold: {hook.min_tools_threshold}")
    print()

    # Simulate what the hook does
    tools = create_sample_tools()
    print(f"Original tools ({len(tools)}):")
    for t in tools:
        print(f"  - {t['function']['name']}")
    print()

    # The hook's async_pre_call_hook method is called by LiteLLM
    # Let's simulate its behavior
    data = {
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
        "tools": tools,
    }

    print(f"Query: {data['messages'][0]['content']}")
    print("Routing...")

    # Call the hook's pre_call_hook method
    result = await hook.async_pre_call_hook(
        user_api_key_dict={},
        cache=None,
        data=data,
        call_type="completion",
    )

    print(f"\nFiltered tools ({len(result['tools'])}):")
    for t in result["tools"]:
        print(f"  - {t['function']['name']}")


def show_proxy_config():
    """
    Show example LiteLLM proxy configuration.
    """
    print("\n=== LiteLLM Proxy Configuration Example ===\n")

    config = """
# proxy_config.yaml
# Configure ATR as a custom hook for the LiteLLM proxy

model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  # Add ATR hook to callbacks
  callbacks:
    - atr.adapters.litellm.ATRToolRoutingHook

  # ATR configuration
  atr_config:
    enabled: true
    max_tools: 10
    min_tools_threshold: 5
    llm_provider: openrouter  # or: openai, anthropic
    llm_model: anthropic/claude-3-haiku

# Environment variables needed:
# OPENROUTER_API_KEY - For routing LLM calls
# OPENAI_API_KEY - For the main model (if using OpenAI)
"""
    print(config)

    print("To start the proxy with this config:")
    print("  litellm --config proxy_config.yaml")
    print()
    print("Then make requests to the proxy:")
    print('  curl http://localhost:4000/chat/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"model": "gpt-4", "messages": [...], "tools": [...]}\'')


async def main():
    """Run all examples."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Note: Set OPENROUTER_API_KEY to run the programmatic example")
        print("      (showing config example instead)\n")
        show_proxy_config()
        return

    await programmatic_example()
    print("\n" + "=" * 50)
    show_proxy_config()


if __name__ == "__main__":
    asyncio.run(main())
