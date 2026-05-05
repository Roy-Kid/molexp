"""ContextManager protocol + default implementation.

The manager is the only code path that produces a
:class:`ContextPacket`. The default implementation is intentionally
conservative — character-budget tail selection, no automatic
filesystem crawl, no tokenization — so tests can pin its behavior
exactly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from molexp.agent.context.compression import ContextCompressor, TailCompressor
from molexp.agent.context.packet import (
    ContextBudget,
    ContextBuildRequest,
    ContextPacket,
)
from molexp.agent.context.prompt import PromptComposer


@runtime_checkable
class ContextManager(Protocol):
    """Assembles a ``ContextPacket`` for one turn."""

    async def build(self, request: ContextBuildRequest) -> ContextPacket: ...


class DefaultContextManager:
    """Default implementation: layered prompt + tail-compressed history."""

    def __init__(
        self,
        composer: PromptComposer | None = None,
        compressor: ContextCompressor | None = None,
        max_chars: int = 200_000,
    ) -> None:
        self._composer = composer or PromptComposer()
        self._compressor = compressor or TailCompressor()
        self._max_chars = max_chars

    async def build(self, request: ContextBuildRequest) -> ContextPacket:
        system = self._composer.compose(
            base=request.base_system,
            workspace=request.workspace_addendum,
            skill=request.skill_addendum,
            override=request.instructions_override,
        )

        history_budget = max(0, self._max_chars - len(system))
        history = await self._compressor.compress(request.history, history_budget)

        history_chars = sum(len(m.content) + len(m.role) for m in history)
        budget = ContextBudget(
            max_chars=self._max_chars,
            used_chars=len(system) + history_chars,
            history_chars=history_chars,
            system_chars=len(system),
        )

        diagnostics: list[str] = []
        if budget.used_chars > self._max_chars:
            diagnostics.append(f"Context exceeds budget: {budget.used_chars} > {self._max_chars}")

        return ContextPacket(
            system=system,
            messages=tuple(history),
            included_refs=request.extra_refs,
            budget=budget,
            diagnostics=tuple(diagnostics),
        )
