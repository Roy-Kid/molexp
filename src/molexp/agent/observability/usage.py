"""Normalized usage counters.

The harness aggregates :class:`Usage` records across turns; the
plugin emits per-request usage in :class:`ModelResponse`. This module
holds the accumulator so observability and recovery layers share one
view of consumption.
"""

from __future__ import annotations

from molexp.agent.types import Usage


class UsageAccumulator:
    """Mutable running totals; safe for single-session use.

    A plain Python class (not BaseModel, not a stdlib dataclass) because it is a
    runtime-mutating accumulator. The :class:`Usage` snapshots it returns
    are pydantic ``BaseModel(frozen=True)`` data.
    """

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        total_tokens: int = 0,
        requests: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.total_tokens = total_tokens
        self.requests = requests

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
