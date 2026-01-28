"""Base protocol for routing LLMs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RoutingLLM(Protocol):
    """
    Protocol for LLMs used in tool routing.

    RoutingLLM defines the interface that LLM adapters must implement
    to be used with ATR's filtering strategies.

    Implementations should be lightweight and optimized for fast responses,
    as routing is typically done with smaller models (e.g., GPT-4o-mini,
    Claude Haiku) to minimize overhead.
    """

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt for context.

        Returns:
            The model's response text.
        """
        ...

    async def acomplete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Async generate a completion for the given prompt.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt for context.

        Returns:
            The model's response text.
        """
        ...
