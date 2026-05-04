"""Normalized usage counters.

The harness aggregates :class:`Usage` records across turns; the
plugin emits per-request usage in :class:`ModelResponse`. This module
holds the accumulator so observability and recovery layers share one
view of consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from molexp.agent.types import Usage


@dataclass
class UsageAccumulator:
    """Mutable running totals; safe for single-session use."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.total_tokens += usage.total_tokens
        self.requests += usage.requests

    def snapshot(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            total_tokens=self.total_tokens,
            requests=self.requests,
        )
