"""``ChatMode`` — single-turn LLM round-trip mode.

Drives one prompt through the runner-supplied :class:`Router`, appends
the exchange to the session, and returns the assistant text.
Multi-turn support is real: the session's ``model_messages`` field
carries the pydantic-ai-native ``ModelMessage`` history back into
``Agent.run(message_history=...)`` on every turn, so the LLM sees the
full conversation context — not just the latest prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
            message_history=session.model_messages,
            tier=ModelTier.DEFAULT,
        )
        session.append(Message(role="assistant", content=result.text))
        session.model_messages = _extract_all_messages(result.raw, session.model_messages)
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


def _extract_all_messages(
    raw: Any,  # noqa: ANN401 — opaque pydantic-ai RunResult; the agent layer firewall
    fallback: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Pull the cumulative pydantic-ai message list off a ``RunResult``.

    pydantic-ai's ``AgentRunResult.all_messages()`` returns the full
    conversation including the latest turn — the canonical value to
    pass back as ``message_history`` next time. Stub routers (used by
    tests) leave ``raw`` empty / shapeless; we degrade to the existing
    history so callers can still chain turns deterministically.
    """
    if raw is None:
        return fallback
    getter = getattr(raw, "all_messages", None)
    if not callable(getter):
        return fallback
    return tuple(getter())


__all__ = ["ChatMode", "ChatModeConfig"]
