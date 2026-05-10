"""``ReviewMode`` — phase-2 placeholder.

Public surface is declared so downstream code can import the name; the
``run`` body raises :class:`NotImplementedError` until phase 2 implements
the actual review workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


class ReviewModeConfig(BaseModel):
    """Tunables for :class:`ReviewMode` (reserved for phase 2)."""

    model_config = ConfigDict(frozen=True)


class ReviewMode(AgentMode):
    """Reserved for phase 2 — :meth:`run` raises ``NotImplementedError``."""

    name = "review"

    def __init__(self, *, config: ReviewModeConfig | None = None) -> None:
        self.config = config or ReviewModeConfig()

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        raise NotImplementedError("ReviewMode is reserved for phase 2")


__all__ = ["ReviewMode", "ReviewModeConfig"]
