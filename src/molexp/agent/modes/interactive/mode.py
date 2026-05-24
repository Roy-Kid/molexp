"""``InteractiveMode`` — the emergent, tool-using agentic loop.

InteractiveMode is the CLI's default interactive mode and the one
*emergent* :class:`~molexp.agent.mode.AgentMode`: the LLM autonomously
decides whether to answer or call a tool, observes the result, and
loops until done. The declarative modes (Chat / Plan / Author / Run /
Review) are its siblings — InteractiveMode neither inherits from nor
specializes them; it *composes* PlanMode through
:func:`~molexp.agent.modes.interactive.delegation.delegate_to_plan`.

The loop itself is pydantic-ai's native ``Agent.iter()``, reached
through :meth:`~molexp.agent.router.Router.stream_agentic`. This module
only *translates* the resulting :data:`~molexp.agent.router.AgenticChunk`
flow into the harness :data:`AgentEvent` stream and persists each turn
to the :class:`~molexp.agent.harness.session.Session` entry-tree.

Two delegation paths into the structured PlanMode pipeline:

- a human ``/plan <preliminary plan>`` prefix — handled deterministically,
  skipping the emergent LLM turn entirely;
- a ``run_plan_pipeline`` tool the LLM may call mid-loop on its own.

Both funnel through ``delegate_to_plan``. v1 ships **read-only** tools:
the auditable output (executable workflow / plan) is PlanMode's job.
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
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.stage import NameOnlyStage
from molexp.agent.mode import AgentMode, AgentRunResult, ModePipeline, PipelineEdge
from molexp.agent.modes.interactive.delegation import delegate_to_plan, run_plan_pipeline_tool
from molexp.agent.modes.interactive.tools import readonly_tools
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    TextDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.types import Message

_LOG = get_logger(__name__)

__all__ = ["InteractiveMode", "InteractiveModeConfig"]

_PLAN_COMMAND = "/plan"
"""Prefix that deterministically routes a turn to the PlanMode pipeline."""

_DEFAULT_SYSTEM_PROMPT = (
    "You are an interactive research assistant for the molexp experiment "
    "platform. You help the user understand and plan computational "
    "experiments. Inspect the user's workspace with the read-only tools "
    "(read_file, list_directory, search_code) so your answers are grounded "
    "in the actual project. When the user wants a concrete, reviewable "
    "experiment plan or workflow, call the run_plan_pipeline tool instead "
    "of answering directly — it produces an auditable plan through a "
    "structured refine, decompose, and review pipeline. Keep answers concise."
)


class InteractiveModeConfig(BaseModel):
    """Tunables for :class:`InteractiveMode`.

    Attributes:
        system_prompt: Extra system-prompt text appended to the
            built-in interactive-assistant preamble.
        workspace_root: Directory the read-only tools are confined to
            and PlanMode delegation persists under. ``None`` falls back
            to the current working directory at run time.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    workspace_root: Path | None = None


def _join_nonempty(*fragments: str) -> str:
    """Join non-empty fragments with a blank line."""
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _render_prior_context(history: tuple[Message, ...]) -> str:
    """Render prior conversation turns into a plain-text preamble.

    Cross-turn context follows ChatMode's approach — prior turns are
    flattened into the system preamble. Within a single turn, the
    multi-step tool history is managed by pydantic-ai's ``Agent.iter()``.
    """
    if not history:
        return ""
    lines = [f"{msg.role}: {msg.content}" for msg in history]
    return "Conversation so far:\n" + "\n".join(lines)


def _chunk_to_event(chunk: AgenticChunk) -> AgentEvent | None:
    """Translate one :data:`AgenticChunk` into an :data:`AgentEvent`.

    Returns ``None`` for :class:`FinalChunk` — its text is the loop's
    terminal output, captured by the caller rather than streamed.
    """
    if isinstance(chunk, TextDeltaChunk):
        return TokenDeltaEvent(text=chunk.text)
    if isinstance(chunk, ToolCallChunk):
        return ToolCallStartedEvent(tool_name=chunk.tool_name, args_summary=chunk.args_summary)
    if isinstance(chunk, ToolResultChunk):
        return ToolCallCompletedEvent(
            tool_name=chunk.tool_name,
            result_summary=chunk.result_summary,
            ok=chunk.ok,
        )
    return None


class InteractiveMode(AgentMode):
    """The emergent tool-using loop — the CLI's default interactive mode."""

    name = "interactive"
    pipeline = ModePipeline(
        stages=(NameOnlyStage("agentic-loop"),),
        edges=(PipelineEdge(from_stage="agentic-loop", to_stage="completed"),),
        terminal_states=("completed",),
        entry="agentic-loop",
    )

    def __init__(self, *, config: InteractiveModeConfig | None = None) -> None:
        self.config = config or InteractiveModeConfig()

    async def run(
        self,
        *,
        harness: AgentHarness,
        user_input: str,
    ) -> AsyncIterator[AgentEvent]:
        """Drive one interactive turn, yielding the orchestration event stream.

        A ``/plan`` prefix is routed deterministically to the structured
        PlanMode pipeline; every other turn enters the emergent loop,
        where the LLM may still call ``run_plan_pipeline`` on its own.
        """
        await harness.emit(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        router = harness.router
        router.clear_usage()

        prior = harness.session.build_context()
        harness.session.append_message(Message(role="user", content=user_input))
        root = (
            Path(self.config.workspace_root)
            if self.config.workspace_root is not None
            else Path.cwd()
        )

        stripped = user_input.strip()
        if stripped == _PLAN_COMMAND or stripped.startswith(_PLAN_COMMAND + " "):
            preliminary = stripped[len(_PLAN_COMMAND) :].strip()
            _LOG.info("[interactive] /plan prefix — routing turn to PlanMode")
            final_text = await delegate_to_plan(harness, preliminary, workspace_root=root)
        else:
            final_text = ""
            preamble = _join_nonempty(
                _DEFAULT_SYSTEM_PROMPT,
                self.config.system_prompt,
                _render_prior_context(prior),
            )
            tools = (*readonly_tools(root), run_plan_pipeline_tool(harness, root))
            async with harness.stage("agentic-loop"):
                async for chunk in router.stream_agentic(
                    prompt=user_input,
                    system=preamble,
                    tools=tools,
                    tier=ModelTier.DEFAULT,
                ):
                    event = _chunk_to_event(chunk)
                    if event is not None:
                        yield event
                    elif isinstance(chunk, FinalChunk):
                        final_text = chunk.text

        harness.session.append_message(Message(role="assistant", content=final_text))
        breakdown = router.snapshot_usage()
        _LOG.info(
            f"[interactive] turn done — usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=final_text,
            messages=harness.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        yield ModeCompletedEvent(text=final_text, result=run_result.model_dump(mode="json"))
