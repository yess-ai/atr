# ATR - Adaptive Tool Routing

[![PyPI version](https://badge.fury.io/py/adaptive-tools.svg)](https://badge.fury.io/py/adaptive-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Dynamic tool selection for AI agents.** ATR filters tools based on user queries before they reach the agent's system prompt, reducing context tokens by ~90% and improving tool selection accuracy.

## The Problem

When agents have access to many tools (50+ MCP tools), two things happen:
1. **Context explosion** - Tool definitions consume 10,000-15,000 tokens before the conversation starts
2. **Tool selection degradation** - Research shows 7-85% accuracy drops with large tool catalogs

## The Solution

ATR intercepts the agent's tool resolution flow and filters tools *before* they reach the system prompt, using a lightweight LLM (e.g., Claude Haiku, GPT-4o-mini) to select only relevant tools.

```
Before ATR: 50 tools × 250 tokens = 12,500 tokens
After ATR:   5 tools × 250 tokens =  1,250 tokens
Savings: 90%
```

## How It Works

```
User Query ──> ATR Router ──> Lightweight LLM ──> Filtered Tool List ──> Agent
                  │                                      │
                  └── Full tool catalog (50+)             └── Only relevant tools (3-5)
```

1. Register your tools with ATR (from any framework - MCP, LangChain, Agno, OpenAI, etc.)
2. Before each agent call, pass the user query through ATR's router
3. ATR uses a cheap, fast LLM to pick only the relevant tools
4. Pass the filtered tools to your agent - smaller context, better accuracy

## Installation

```bash
# Core package (zero dependencies)
pip install adaptive-tools

# With LLM provider
pip install adaptive-tools[openrouter]  # Recommended - access to many models
pip install adaptive-tools[openai]
pip install adaptive-tools[anthropic]

# With framework integration
pip install adaptive-tools[langgraph]
pip install adaptive-tools[agno]
pip install adaptive-tools[openai-agents]
pip install adaptive-tools[litellm]

# Everything
pip install adaptive-tools[all]
```

## Quick Start

```python
from atr import ToolRouter, ToolSpec
from atr.llm import OpenRouterLLM

# Create router with LLM
router = ToolRouter(llm=OpenRouterLLM())

# Add tools
router.add_tools([
    ToolSpec(name="get_stock_price", description="Get current stock price"),
    ToolSpec(name="get_company_news", description="Get company news articles"),
    ToolSpec(name="get_weather", description="Get weather for a location"),
    ToolSpec(name="send_email", description="Send an email"),
    ToolSpec(name="create_calendar_event", description="Create a calendar event"),
])

# Route query to filter tools
filtered = router.route("What is AAPL's stock price?")
print(filtered.names)  # {'get_stock_price'}
```

## Framework Integrations

### LangGraph

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from atr import ToolRouter
from atr.adapters import LangChainAdapter
from atr.adapters.langchain import filter_tools
from atr.llm import OpenRouterLLM

async with MultiServerMCPClient(server_configs) as client:
    all_tools = await client.get_tools()

    # Create router from LangChain tools
    router = ToolRouter(llm=OpenRouterLLM())
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    # Route and filter
    filtered_specs = router.route("Read the README")
    filtered_tools = filter_tools(all_tools, filtered_specs)

    # Use filtered tools with your agent
    agent = create_react_agent(model, filtered_tools)
```

For LangGraph graphs, use the built-in node creators to add routing as a graph node:

```python
from atr.adapters.langchain import create_async_router_node

# Create a LangGraph-compatible node that routes tools
route_node = create_async_router_node(router, all_tools)

# Add to your graph
graph = StateGraph(AgentState)
graph.add_node("route_tools", route_node)
graph.add_node("agent", agent_node)
graph.add_edge("route_tools", "agent")
```

### Agno

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from atr import ToolRouter
from atr.adapters import AgnoAdapter
from atr.adapters.agno import filter_tools
from atr.llm import OpenRouterLLM

async with MCPTools(command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]) as mcp:
    # Convert toolkit to specs and create router
    router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
    router.add_tools(AgnoAdapter.to_specs([mcp]))

    # Route and filter
    filtered_specs = await router.aroute("List files")
    filtered_funcs = filter_tools([mcp], filtered_specs)

    # Create agent with filtered tools
    agent = Agent(model=OpenAIChat(id="gpt-4o"), tools=filtered_funcs)
```

### OpenAI Agents SDK

```python
from atr import ToolRouter
from atr.adapters import OpenAIAdapter
from atr.adapters.openai import filter_tools
from atr.llm import OpenRouterLLM

# Create router from OpenAI function definitions
router = ToolRouter(llm=OpenRouterLLM())
router.add_tools(OpenAIAdapter.to_specs(openai_tools))

# Route and filter
filtered_specs = router.route("What's the weather?")
filtered_tools = filter_tools(openai_tools, filtered_specs)
```

### LiteLLM

ATR integrates with LiteLLM as a custom hook for automatic tool routing - no manual filtering needed:

```python
import litellm
from atr.adapters.litellm import create_hook

# Create and register the hook
hook = create_hook(
    llm_provider="openrouter",
    llm_model="anthropic/claude-3-haiku",
    max_tools=5,
)
litellm.callbacks = [hook]

# Tools are now automatically filtered before reaching the model
response = await litellm.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=all_tools,  # ATR filters these automatically
)
```

For LiteLLM proxy, configure in `proxy_config.yaml`:

```yaml
litellm_settings:
  callbacks:
    - atr.adapters.litellm.ATRToolRoutingHook
  atr_config:
    enabled: true
    max_tools: 10
    llm_provider: openrouter
    llm_model: anthropic/claude-3-haiku
```

### MCP (Direct)

```python
from mcp import ClientSession
from atr import ToolRouter
from atr.adapters import MCPAdapter
from atr.llm import OpenRouterLLM

# Get tools from MCP session
mcp_tools = await session.list_tools()

# Convert and route
router = ToolRouter(llm=OpenRouterLLM())
router.add_tools(MCPAdapter.to_specs(mcp_tools.tools))

filtered_specs = router.route("Read the README")
filtered_tools = MCPAdapter.filter_tools(mcp_tools.tools, filtered_specs)
```

## API Reference

### Core Classes

#### `ToolSpec`

Framework-agnostic tool specification.

```python
ToolSpec(
    name="get_weather",
    description="Get weather for a location",
    parameters={"type": "object", ...},  # JSON Schema (optional)
    source="mcp:weather",                # Origin identifier (optional)
    metadata={},                         # Framework-specific data (optional)
)
```

#### `ToolCollection`

Returned from routing operations. Provides convenient access to filtered tools.

```python
collection = ToolCollection(tools=[...])
collection.names                           # Set of tool names
collection.filter_by_names(["tool1"])      # Filter by name
collection.to_summaries()                  # For routing prompts
len(collection)                            # Number of tools
"tool_name" in collection                  # Membership check
collection[0]                              # Index access
collection["tool_name"]                    # Name-based access
```

#### `ToolRouter`

Main router class. Routes queries to select relevant tools.

```python
router = ToolRouter(
    llm=OpenRouterLLM(),  # Or any RoutingLLM
    max_tools=10,         # Max tools to return (default: 10)
)
router.add_tools([...])
router.add_tool(single_spec)
router.clear_tools()

filtered = router.route("query")        # Sync
filtered = await router.aroute("query") # Async
```

### LLM Providers

All providers use lazy client initialization and support both sync and async.

```python
from atr.llm import OpenRouterLLM, OpenAILLM, AnthropicLLM

# OpenRouter (recommended - access to many models via single API key)
llm = OpenRouterLLM(model="anthropic/claude-3-haiku")  # default model

# OpenAI
llm = OpenAILLM(model="gpt-4o-mini")

# Anthropic
llm = AnthropicLLM(model="claude-3-haiku-20240307")
```

### Adapters

Every adapter follows the same pattern: `to_specs()` to convert, `filter_tools()` to filter back.

```python
from atr.adapters import MCPAdapter, LangChainAdapter, AgnoAdapter, OpenAIAdapter, LiteLLMAdapter

# Convert framework tools to ToolSpecs
specs = MCPAdapter.to_specs(mcp_tools)
specs = LangChainAdapter.to_specs(langchain_tools)
specs = AgnoAdapter.to_specs([toolkit_or_function, ...])
specs = OpenAIAdapter.to_specs(openai_tool_defs)
specs = LiteLLMAdapter.to_specs(litellm_tool_defs)

# After routing, filter original tools by the routing result
filtered = MCPAdapter.filter_tools(mcp_tools, filtered_collection)
filtered = LangChainAdapter.filter_tools(langchain_tools, filtered_collection)
filtered = AgnoAdapter.filter_tools([toolkit_or_function, ...], filtered_collection)
filtered = OpenAIAdapter.filter_tools(openai_tool_defs, filtered_collection)
filtered = LiteLLMAdapter.filter_tools(litellm_tool_defs, filtered_collection)
```

Each adapter module also exports a standalone `filter_tools()` convenience function:

```python
from atr.adapters.langchain import filter_tools
from atr.adapters.agno import filter_tools
from atr.adapters.openai import filter_tools
from atr.adapters.litellm import filter_tools
```

## Custom Filter Strategies

ATR uses a pluggable strategy pattern. The default `LLMFilterStrategy` uses an LLM, but you can implement your own:

```python
from atr import ToolRouter, ToolCollection, BaseFilterStrategy

class MyCustomStrategy(BaseFilterStrategy):
    def filter(self, query: str, tools: ToolCollection) -> ToolCollection:
        # Your custom filtering logic (embeddings, keyword matching, etc.)
        return tools.filter_by_names(["relevant_tool"])

    async def afilter(self, query: str, tools: ToolCollection) -> ToolCollection:
        return self.filter(query, tools)

router = ToolRouter(strategy=MyCustomStrategy())
```

Built-in strategies:
- `LLMFilterStrategy` - Uses an LLM to select relevant tools (default when `llm` is provided)
- `PassthroughStrategy` - Returns all tools unfiltered (default when no `llm` or `strategy` is provided)

## Design

- **Zero core dependencies** - Optional extras only for the frameworks you use
- **Fail-open** - If routing fails, all original tools are returned
- **Protocol-based** - `FilterStrategy` and `RoutingLLM` are Protocols, not base classes - duck typing works
- **Lazy imports** - Optional dependencies are loaded only when their adapter is accessed
- **Typed** - Full type hints with `py.typed` marker, strict mypy config

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |

## Examples

See the [examples/](examples/) directory for complete working examples:
- `basic_usage.py` - Core functionality without frameworks
- `langgraph_example.py` - LangGraph with MCP tools
- `agno_example.py` - Agno with MCP and toolkits
- `openai_agents_example.py` - OpenAI Agents SDK
- `litellm_example.py` - LiteLLM with automatic hook-based routing

## Development

```bash
git clone https://github.com/yess-ai/atr.git
cd atr

# Install with dev dependencies using uv
uv sync --all-extras

# Run tests
uv run pytest

# Format & lint
uv run ruff format
uv run ruff check --fix
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
