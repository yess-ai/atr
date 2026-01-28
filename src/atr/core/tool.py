"""Core tool data structures for ATR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ToolSpec:
    """
    Framework-agnostic tool specification.

    ToolSpec represents a tool in a standardized format that can be converted
    to/from any framework's tool definition (MCP, LangChain, OpenAI, etc.).

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema for tool parameters (optional).
        source: Origin of the tool, e.g., "mcp:filesystem", "langchain" (optional).
        metadata: Framework-specific data for round-trip conversion (optional).
    """

    name: str
    description: str
    parameters: dict[str, Any] | None = None
    source: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolSpec):
            return NotImplemented
        return self.name == other.name

    def to_summary(self, max_description_length: int = 150) -> str:
        """
        Convert to a lightweight summary for routing prompts.

        Args:
            max_description_length: Maximum length of description before truncation.

        Returns:
            A string in format "- name: description"
        """
        desc = self.description
        if len(desc) > max_description_length:
            desc = desc[: max_description_length - 3] + "..."
        return f"- {self.name}: {desc}"


@dataclass
class ToolCollection:
    """
    A collection of ToolSpecs with utility methods.

    ToolCollection is returned from routing operations and provides
    convenient access to filtered tools by name or iteration.
    """

    tools: list[ToolSpec] = field(default_factory=list)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, item: str | ToolSpec) -> bool:
        if isinstance(item, str):
            return item in self.names
        return item in self.tools

    def __getitem__(self, key: str | int) -> ToolSpec:
        if isinstance(key, int):
            return self.tools[key]
        for tool in self.tools:
            if tool.name == key:
                return tool
        raise KeyError(f"Tool '{key}' not found")

    @property
    def names(self) -> set[str]:
        """Get set of all tool names in the collection."""
        return {tool.name for tool in self.tools}

    def filter_by_names(self, names: set[str] | list[str]) -> ToolCollection:
        """
        Filter collection to only include tools with given names.

        Args:
            names: Set or list of tool names to keep.

        Returns:
            New ToolCollection with only the matching tools.
        """
        name_set = set(names)
        return ToolCollection(tools=[t for t in self.tools if t.name in name_set])

    def to_summaries(self, max_description_length: int = 150) -> str:
        """
        Convert all tools to summaries for routing prompts.

        Args:
            max_description_length: Maximum length of each description.

        Returns:
            Newline-separated tool summaries.
        """
        return "\n".join(t.to_summary(max_description_length) for t in self.tools)

    def add(self, tool: ToolSpec) -> None:
        """Add a tool to the collection."""
        self.tools.append(tool)

    def extend(self, tools: list[ToolSpec]) -> None:
        """Add multiple tools to the collection."""
        self.tools.extend(tools)
