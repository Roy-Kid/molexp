"""``InteractiveMode`` — the emergent, tool-using agentic loop.

InteractiveMode is the CLI's default interactive mode and the one
*emergent* :class:`~molexp.agent.mode.AgentMode`: the LLM autonomously
decides whether to answer or call a tool, observes the result, and
loops until done. The declarative modes (Chat / Plan / Author / Run /
Review) are its siblings — InteractiveMode neither inherits from nor
specializes them; it *composes* PlanMode through
:func:`~molexp.agent.modes.interactive.delegation.delegate_to_plan`.

The loop itself is pydantic-ai's native ``Agent.iter()``, reached
through :meth:`~molexp.agent.router.Router.stream_agentic`. After
``agent-mode-stage-pipeline-03``, the whole loop lives inside one
:class:`~molexp.agent.modes.interactive.stages.EmergentLoop` Stage —
:class:`InteractiveMode.run` delegates to the substrate's
:meth:`AgentMode.run_pipeline`. The Stage's internals stay
sequential by construction; the substrate's :class:`Stage` ABC does
not constrain what happens *inside* a stage body.

Two delegation paths into the structured PlanMode pipeline:

- a human ``/plan <preliminary plan>`` prefix — handled
  deterministically inside the Stage, skipping the emergent LLM turn
  entirely;
- a ``run_plan_pipeline`` tool the LLM may call mid-loop on its own.

Both funnel through ``delegate_to_plan``. v1 ships **read-only**
tools: the auditable output (executable workflow / plan) is
PlanMode's job.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.harness.events import (
    AgentEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes.interactive.stages import EmergentLoop
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["InteractiveMode", "InteractiveModeConfig"]


class InteractiveModeConfig(BaseModel):
    """Tunables for :class:`InteractiveMode`.

    Attributes:
        system_prompt: Extra system-prompt text appended to the
            built-in interactive-assistant preamble.
        workspace_root: Directory the read-only tools are confined to
            and PlanMode delegation persists under. ``None`` falls
            back to the current working directory at run time.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    workspace_root: Path | None = None


_CLASS_PIPELINE = ModePipeline(
    stages=(NameOnlyStage("agentic-loop"),),
    edges=(PipelineEdge(from_stage="agentic-loop", to_stage="completed"),),
    terminal_states=("completed",),
    entry="agentic-loop",
)


class InteractiveMode(AgentMode):
    """The emergent tool-using loop — the CLI's default interactive mode."""

    name = "interactive"
    pipeline = _CLASS_PIPELINE

    def __init__(self, *, config: InteractiveModeConfig | None = None) -> None:
        self.config = config or InteractiveModeConfig()
        self._captured_prior: tuple[Message, ...] = ()
        self._final_text: str = ""
        self.pipeline = ModePipeline(
            stages=(EmergentLoop(interactive_mode=self),),
            edges=_CLASS_PIPELINE.edges,
            terminal_states=_CLASS_PIPELINE.terminal_states,
            entry="agentic-loop",
        )

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive one interactive turn, yielding the orchestration event stream."""
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        harness.router.clear_usage()
        self._captured_prior = harness.session.build_context()
        harness.session.append_message(Message(role="user", content=user_input))
        self._final_text = ""

        async for event in self.run_pipeline(
            harness=harness,
            user_input=user_input,
            initial_input=user_input,
        ):
            yield event

        harness.session.append_message(Message(role="assistant", content=self._final_text))
        breakdown = harness.router.snapshot_usage()
        _LOG.info(
            f"[interactive] turn done — usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=self._final_text,
            messages=harness.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        yield ModeCompletedEvent(text=self._final_text, result=run_result.model_dump(mode="json"))
