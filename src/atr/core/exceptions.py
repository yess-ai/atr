"""Custom exceptions for ATR."""

from __future__ import annotations


class ATRError(Exception):
    """Base exception for ATR errors."""


class RoutingError(ATRError):
    """Error during tool routing."""


class LLMError(ATRError):
    """Error communicating with LLM provider."""


class AdapterError(ATRError):
    """Error converting tools between formats."""


class ConfigurationError(ATRError):
    """Error in ATR configuration."""
