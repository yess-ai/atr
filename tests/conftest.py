"""Shared fixtures for ATR tests."""

from __future__ import annotations

import pytest

from atr import ToolSpec


@pytest.fixture
def sample_toolspecs() -> list[ToolSpec]:
    """Common set of ToolSpecs for testing routing scenarios."""
    return [
        ToolSpec(name="read_file", description="Read contents of a file from the filesystem"),
        ToolSpec(name="write_file", description="Write content to a file on the filesystem"),
        ToolSpec(name="send_email", description="Send an email message to a recipient"),
        ToolSpec(name="search_web", description="Search the web for information"),
        ToolSpec(name="execute_sql", description="Execute SQL query on database"),
        ToolSpec(name="list_directory", description="List files in a directory"),
        ToolSpec(name="create_issue", description="Create a GitHub issue"),
        ToolSpec(name="send_slack", description="Send a Slack message"),
    ]


@pytest.fixture
def file_tools(sample_toolspecs: list[ToolSpec]) -> list[ToolSpec]:
    """Tools related to file operations."""
    return [t for t in sample_toolspecs if "file" in t.name or "directory" in t.name]


@pytest.fixture
def communication_tools(sample_toolspecs: list[ToolSpec]) -> list[ToolSpec]:
    """Tools related to communication (email, slack)."""
    return [t for t in sample_toolspecs if any(x in t.name for x in ["email", "slack"])]


class MockRoutingLLM:
    """Configurable mock LLM for testing routing behavior."""

    def __init__(
        self,
        return_tools: list[str] | None = None,
        raise_error: Exception | None = None,
    ):
        self.return_tools = return_tools or []
        self.raise_error = raise_error
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Sync completion that returns configured tool names."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.raise_error:
            raise self.raise_error

        return "\n".join(self.return_tools)

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Async completion that returns configured tool names."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.raise_error:
            raise self.raise_error

        return "\n".join(self.return_tools)


@pytest.fixture
def mock_llm():
    """Factory for creating mock LLMs with specific behavior."""

    def _create(
        return_tools: list[str] | None = None,
        raise_error: Exception | None = None,
    ) -> MockRoutingLLM:
        return MockRoutingLLM(return_tools=return_tools, raise_error=raise_error)

    return _create
