"""ChatMode's single ``ChatTurn`` Stage.

Wraps the existing one-LLM-round-trip body so :class:`ChatMode` can
delegate ``run`` to :meth:`AgentMode.run_pipeline` (the substrate
introduced in ``agent-mode-stage-pipeline-01``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.router import ModelTier
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.chat import ChatMode

__all__ = ["ChatTurn"]


def _join_nonempty(*fragments: str) -> str:
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _render_prior_context(history: tuple[Message, ...]) -> str:
    if not history:
        return ""
    lines = [f"{msg.role}: {msg.content}" for msg in history]
    return "Conversation so far:\n" + "\n".join(lines)


class ChatTurn(Stage[str, str]):
    """One LLM round-trip; reads prior session context for the preamble.

    The stage stores the assistant text on ``chat_mode._result_text``;
    :class:`ChatMode` reads it post-pipeline to build the terminal
    :class:`ModeCompletedEvent`.
    """

    name: ClassVar[str] = "chat-turn"

    def __init__(self, *, chat_mode: ChatMode) -> None:
        self._chat_mode = chat_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: str,
    ) -> AsyncIterator[AgentEvent | str]:
        prior = self._chat_mode._captured_prior
        preamble = _join_nonempty(
            self._chat_mode.config.system_prompt,
            _render_prior_context(prior),
        )
        result = await harness.router.complete_text(
            prompt=input,
            system=preamble,
            tier=ModelTier.DEFAULT,
        )
        self._chat_mode._result_text = result.text
        yield result.text
