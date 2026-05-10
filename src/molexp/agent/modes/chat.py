"""``ChatMode`` — single-turn LLM round-trip mode.

Drives one prompt through the runner-supplied :class:`Router`, appends
the exchange to the session, and returns the assistant text.
Multi-turn support is implicit: the session history is the
conversation log; each ``AgentRunner.run`` call is one assistant turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


_LOG = get_logger(__name__)


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


class ChatMode(AgentMode):
    """One ``user_input`` → one LLM call → one ``AgentRunResult``."""

    name = "chat"

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()

    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        router.clear_usage()
        session.append(Message(role="user", content=user_input))
        result = await router.complete_text(
            prompt=user_input,
            system=self.config.system_prompt,
            tier=ModelTier.DEFAULT,
        )
        session.append(Message(role="assistant", content=result.text))
        breakdown = router.snapshot_usage()
        _LOG.info(
            f"[chat-mode] usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} total={breakdown.total.total_tokens} "
            f"reqs={breakdown.total.requests}"
        )
        return AgentRunResult(
            text=result.text,
            messages=tuple(session.history),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )


__all__ = ["ChatMode", "ChatModeConfig"]
