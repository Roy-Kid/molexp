"""``ChatMode`` — the minimal one-turn LLM mode.

ChatMode is one ``user_input`` → one LLM round-trip → one
:class:`~molexp.agent.mode.AgentRunResult`. Plain ``async def run`` body
(no ``Stage`` / ``ModePipeline`` / ``RepairPolicy``); events flow
through the injected :class:`~molexp.agent.events.AsyncIteratorEventSink`
in emission order.

Conversation context comes from the session entry-tree: prior turns are
rebuilt with :meth:`Session.build_context` and rendered into the prompt
when the router supports it. The terminal
:class:`~molexp.agent.events.ModeCompletedEvent` carries the JSON dump
of the run's :class:`AgentRunResult` so the runner can rebuild it.
"""

from __future__ import annotations

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.events import (
    AsyncIteratorEventSink,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.runtime import AgentRuntime
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["ChatMode", "ChatModeConfig"]


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


class ChatMode(AgentMode):
    """One ``user_input`` → one LLM round-trip → one :class:`AgentRunResult`."""

    name = "chat"

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()

    async def run(
        self,
        *,
        runtime: AgentRuntime,
        sink: AsyncIteratorEventSink,
        user_input: str,
    ) -> None:
        """Drive one chat turn; emit events through ``sink``."""
        await sink(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        runtime.router.clear_usage()

        runtime.session.append_message(Message(role="user", content=user_input))
        result = await runtime.router.complete_text(
            prompt=user_input,
            system=self.config.system_prompt,
        )
        runtime.session.append_message(Message(role="assistant", content=result.text))

        breakdown = runtime.router.snapshot_usage()
        _LOG.info(
            f"[chat-mode] usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} "
            f"total={breakdown.total.total_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=result.text,
            messages=runtime.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        await sink(
            ModeCompletedEvent(
                text=result.text,
                result=run_result.model_dump(mode="json"),
            )
        )
