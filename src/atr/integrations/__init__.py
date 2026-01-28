"""Framework integrations for ATR."""

__all__: list[str] = []


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name in ("filter_tools", "create_router_node", "LangGraphRouter"):
        from atr.integrations import langgraph

        return getattr(langgraph, name)
    elif name in (
        "to_specs",
        "filter_tools",
        "create_router",
        "route_and_filter",
        "route_and_filter_sync",
    ):
        from atr.integrations import agno

        return getattr(agno, name)
    elif name in ("OpenAIAgentsAdapter", "FilteredRunner", "OpenAIAgentsRouter"):
        from atr.integrations import openai_agents

        return getattr(openai_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
