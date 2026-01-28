# ATR - Adaptive Tool Routing

[![PyPI version](https://badge.fury.io/py/atr.svg)](https://badge.fury.io/py/atr)
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

## Installation

```bash
# Core package (zero dependencies)
pip install atr

# With LLM provider
pip install atr[openrouter]  # Recommended - access to many models
pip install atr[openai]
pip install atr[anthropic]

# With framework integration
pip install atr[langgraph]
pip install atr[agno]
pip install atr[openai-agents]
pip install atr[litellm]

# Everything
pip install atr[all]
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

# Load MCP tools
async with MultiServerMCPClient(server_configs) as client:
    all_tools = await client.get_tools()

    # Create router
    router = ToolRouter(llm=OpenRouterLLM())
    router.add_tools(LangChainAdapter.to_specs(all_tools))

    # Route and filter
    filtered_specs = router.route("Read the README")
    filtered_tools = filter_tools(all_tools, filtered_specs)

    # Use filtered tools with your agent
    agent = create_react_agent(model, filtered_tools)
```

Or use the high-level `LangGraphRouter`:

```python
from atr.adapters import LangGraphRouter

router = LangGraphRouter(llm=OpenRouterLLM(), tools=all_tools)
filtered = await router.aroute("Read the README")  # Returns LangChain tools
```

### Agno

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from atr import ToolRouter
from atr.adapters import AgnoAdapter
from atr.llm import OpenRouterLLM

async with MCPTools(command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]) as mcp:
    # Convert to specs and create router
    specs = AgnoAdapter.to_specs(mcp.functions)
    router = ToolRouter(llm=OpenRouterLLM(), max_tools=5)
    router.add_tools(specs)

    # Route and filter
    filtered = await router.aroute("List files")
    filtered_funcs = AgnoAdapter.filter_tools(mcp.functions, filtered)

    # Create agent with filtered tools
    agent = Agent(model=OpenAIChat(id="gpt-4o"), tools=filtered_funcs)
```

Or use the convenience helpers:

```python
from atr.adapters.agno import create_router, route_and_filter

async with MCPTools(...) as mcp:
    router = create_router(mcp, llm=OpenRouterLLM())
    filtered_funcs = await route_and_filter(router, mcp, "List files")
```

### OpenAI Agents SDK

```python
from atr.adapters import OpenAIRouter

router = OpenAIRouter(llm=OpenRouterLLM(), tools=openai_tools)
filtered = router.route("What's the weather?")  # Returns OpenAI tool definitions
```

### LiteLLM

ATR integrates with LiteLLM as a custom hook for automatic tool routing:

```python
import litellm
from atr.adapters.litellm import create_hook

# Create and configure the hook
hook = create_hook(
    llm_provider="openrouter",
    llm_model="anthropic/claude-3-haiku",
    max_tools=5,
)

# Add to LiteLLM callbacks
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

## API Reference

### Core Classes

#### `ToolSpec`
Framework-agnostic tool specification.

```python
ToolSpec(
    name="get_weather",
    description="Get weather for a location",
    parameters={"type": "object", ...},  # JSON Schema (optional)
    source="mcp:weather",                  # Origin identifier (optional)
    metadata={},                           # Framework-specific data (optional)
)
```

#### `ToolCollection`
A collection of tools with utility methods.

```python
collection = ToolCollection(tools=[...])
collection.names        # Set of tool names
collection.filter_by_names(["tool1", "tool2"])
collection.to_summaries()  # For routing prompts
```

#### `ToolRouter`
Main router class.

```python
router = ToolRouter(
    llm=OpenRouterLLM(),  # Or any RoutingLLM
    max_tools=10,         # Max tools to return
)
router.add_tools([...])
filtered = router.route("query")        # Sync
filtered = await router.aroute("query") # Async
```

### LLM Adapters

```python
from atr.llm import OpenRouterLLM, OpenAILLM, AnthropicLLM

# OpenRouter (recommended - access to many models)
llm = OpenRouterLLM(model="anthropic/claude-3-haiku")

# OpenAI
llm = OpenAILLM(model="gpt-4o-mini")

# Anthropic
llm = AnthropicLLM(model="claude-3-haiku-20240307")
```

### Tool Adapters

```python
from atr.adapters import MCPAdapter, LangChainAdapter, AgnoAdapter, OpenAIAdapter, LiteLLMAdapter

# MCP tools
specs = MCPAdapter.to_specs(mcp_tools)
filtered_mcp = MCPAdapter.filter_tools(mcp_tools, filtered_collection)

# LangChain tools
specs = LangChainAdapter.to_specs(langchain_tools)
filtered_lc = LangChainAdapter.filter_tools(langchain_tools, filtered_collection)

# Agno functions
specs = AgnoAdapter.to_specs(agno_functions)
filtered_agno = AgnoAdapter.filter_tools(agno_functions, filtered_collection)

# OpenAI function definitions
specs = OpenAIAdapter.to_specs(openai_tools)
filtered_openai = OpenAIAdapter.filter_tools(openai_tools, filtered_collection)

# LiteLLM (same format as OpenAI)
specs = LiteLLMAdapter.to_specs(litellm_tools)
filtered_litellm = LiteLLMAdapter.filter_tools(litellm_tools, filtered_collection)
```

## Custom Filter Strategies

ATR supports pluggable filter strategies:

```python
from atr.core.strategies import FilterStrategy, BaseFilterStrategy
from atr import ToolRouter, ToolCollection

class MyCustomStrategy(BaseFilterStrategy):
    def filter(self, query: str, tools: ToolCollection) -> ToolCollection:
        # Your custom filtering logic
        return tools.filter_by_names(["relevant_tool"])

    async def afilter(self, query: str, tools: ToolCollection) -> ToolCollection:
        return self.filter(query, tools)

router = ToolRouter(strategy=MyCustomStrategy())
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |

## Examples

See the [examples/](examples/) directory for complete examples:
- `basic_usage.py` - Core functionality without frameworks
- `langgraph_example.py` - LangGraph with MCP tools
- `agno_example.py` - Agno with MCP and toolkits
- `openai_agents_example.py` - OpenAI Agents SDK
- `litellm_example.py` - LiteLLM with automatic hook-based routing

## Development

```bash
# Clone the repository
git clone https://github.com/yess-ai/atr.git
cd atr

# Install with dev dependencies using uv
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run ruff format
uv run ruff check --fix
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
