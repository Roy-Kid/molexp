"""``ChatMode`` — the simple harness-based reference mode.

ChatMode is one ``user_input`` → one LLM round-trip → one
:class:`~molexp.agent.mode.AgentRunResult`. It is the minimal
:class:`~molexp.agent.mode.AgentMode` implementation and the canonical
example of the harness-based contract:

- ``run`` is an async generator that *yields*
  :data:`~molexp.agent.harness.events.AgentEvent`\\ s;
- it appends the user turn and the assistant turn to the harness's
  :class:`~molexp.agent.harness.session.Session` entry-tree;
- it delegates the core LLM-call body to a single
  :class:`~molexp.agent.modes.chat_stages.ChatTurn` Stage driven by
  the substrate's :func:`execute_pipeline`;
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
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes.chat_stages import ChatTurn
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["ChatMode", "ChatModeConfig"]


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


# Class-level declarative pipeline (NameOnlyStage placeholder) — used
# by ``get_flowchart`` on bare instances + the no-drift guard. The real
# instance-level pipeline (carrying the executable ``ChatTurn`` Stage)
# is built in ``__init__`` and shadows this attribute.
_CLASS_PIPELINE = ModePipeline(
    stages=(NameOnlyStage("chat-turn"),),
    edges=(PipelineEdge(from_stage="chat-turn", to_stage="completed"),),
    terminal_states=("completed",),
    entry="chat-turn",
)


class ChatMode(AgentMode):
    """One ``user_input`` → one LLM round-trip → one :class:`AgentRunResult`."""

    name = "chat"
    pipeline = _CLASS_PIPELINE

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()
        self._captured_prior: tuple[Message, ...] = ()
        self._result_text: str = ""
        self.pipeline = ModePipeline(
            stages=(ChatTurn(chat_mode=self),),
            edges=_CLASS_PIPELINE.edges,
            terminal_states=_CLASS_PIPELINE.terminal_states,
            entry="chat-turn",
        )

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive one chat turn, yielding the orchestration event stream."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.router.clear_usage()

        self._captured_prior = harness.session.build_context()
        harness.session.append_message(Message(role="user", content=user_input))
        self._result_text = ""

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=user_input,
        ):
            yield event

        harness.session.append_message(Message(role="assistant", content=self._result_text))
        breakdown = harness.router.snapshot_usage()
        _LOG.info(
            f"[chat-mode] usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} "
            f"total={breakdown.total.total_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=self._result_text,
            messages=harness.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        yield ModeCompletedEvent(
            text=self._result_text,
            result=run_result.model_dump(mode="json"),
        )
