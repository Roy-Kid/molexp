"""``ChatMode`` — the simple harness-based reference mode.

ChatMode is one ``user_input`` → one LLM round-trip → one
:class:`~molexp.agent.mode.AgentRunResult`. It is the minimal
:class:`~molexp.agent.mode.AgentMode` implementation and the canonical
example of the harness-based contract:

- ``run`` is an async generator that *yields*
  :data:`~molexp.agent.harness.events.AgentEvent`\\ s;
- it appends the user turn and the assistant turn to the harness's
  :class:`~molexp.agent.harness.session.Session` entry-tree;
- it brackets the LLM call in an ``AgentHarness.stage``;
- its terminal yield is a
  :class:`~molexp.agent.harness.events.ModeCompletedEvent` carrying the
  final :class:`AgentRunResult`.

Conversation context comes from the session entry-tree: prior turns are
rebuilt with :meth:`Session.build_context` and rendered into the prompt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import AgentEvent, ModeCompletedEvent, ModeStartedEvent
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["ChatMode", "ChatModeConfig"]


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


def _render_prior_context(history: tuple[Message, ...]) -> str:
    """Render prior conversation turns into a plain-text preamble.

    The harness entry-tree stores molexp-shaped :class:`Message`\\ s;
    ChatMode flattens them into a transcript the LLM reads as context.
    Returns the empty string when there is no prior history.
    """
    if not history:
        return ""
    lines = [f"{msg.role}: {msg.content}" for msg in history]
    return "Conversation so far:\n" + "\n".join(lines)


class ChatMode(AgentMode):
    """One ``user_input`` → one LLM round-trip → one :class:`AgentRunResult`."""

    name = "chat"

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive one chat turn, yielding the orchestration event stream."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        router = harness.router
        router.clear_usage()

        prior = harness.session.build_context()
        harness.session.append_message(Message(role="user", content=user_input))

        async with harness.stage("chat-turn"):
            preamble = _join_nonempty(self.config.system_prompt, _render_prior_context(prior))
            result = await router.complete_text(
                prompt=user_input,
                system=preamble,
                tier=ModelTier.DEFAULT,
            )

        harness.session.append_message(Message(role="assistant", content=result.text))
        breakdown = router.snapshot_usage()
        _LOG.info(
            f"[chat-mode] usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} "
            f"total={breakdown.total.total_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=result.text,
            messages=tuple(_messages_from_context(harness)),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        yield ModeCompletedEvent(
            text=result.text,
            result=run_result.model_dump(mode="json"),
        )


def _join_nonempty(*fragments: str) -> str:
    """Join non-empty fragments with a blank line."""
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _messages_from_context(harness: AgentHarness) -> tuple[Message, ...]:
    """Return the full conversation as molexp :class:`Message`\\ s."""
    return harness.session.build_context()
