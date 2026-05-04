"""Compression hooks.

The harness ships with a no-op compressor; later phases can plug in
summarizers without touching the orchestrator. The protocol is
defined now so callers commit to the right shape from day one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.agent.types import Message


@runtime_checkable
class ContextCompressor(Protocol):
    """Reduce a message history to fit within ``max_chars``."""

    async def compress(
        self,
        history: tuple[Message, ...],
        max_chars: int,
    ) -> tuple[Message, ...]: ...


class NoopCompressor:
    """Default compressor: returns the history unchanged."""

    async def compress(
        self,
        history: tuple[Message, ...],
        max_chars: int,
    ) -> tuple[Message, ...]:
        return history


class TailCompressor:
    """Tail-only compression: keep recent messages until the budget fits.

    System messages always survive (they ride free of the budget).
    Non-system messages are walked in reverse and kept while the
    running cost stays under ``max_chars``. The output preserves the
    original order.
    """

    async def compress(
        self,
        history: tuple[Message, ...],
        max_chars: int,
    ) -> tuple[Message, ...]:
        if max_chars <= 0:
            return history

        system_indices = [i for i, m in enumerate(history) if m.role == "system"]
        kept_indices: set[int] = set(system_indices)
        running = 0
        for idx in range(len(history) - 1, -1, -1):
            if idx in kept_indices:
                continue
            msg = history[idx]
            cost = len(msg.content) + len(msg.role)
            if running + cost > max_chars:
                break
            kept_indices.add(idx)
            running += cost
        return tuple(history[i] for i in sorted(kept_indices))
