"""InteractiveMode's single ``EmergentLoop`` Stage.

Wraps the existing ``/plan`` branch + pydantic-ai ``Agent.iter()``
loop so :class:`InteractiveMode` can delegate ``run`` to
:meth:`AgentMode.run_pipeline`. Stage internals are sequential by
construction — the substrate's :class:`Stage` ABC does not constrain
what happens *inside* a stage body.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.agent.harness.events import (
    AgentEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.harness.stage import Stage
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

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.interactive.mode import InteractiveMode

__all__ = ["PLAN_COMMAND", "EmergentLoop"]

_LOG = get_logger(__name__)

PLAN_COMMAND = "/plan"
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


def _join_nonempty(*fragments: str) -> str:
    return "\n\n".join(fragment for fragment in fragments if fragment)


def _render_prior_context(history: tuple[Message, ...]) -> str:
    if not history:
        return ""
    lines = [f"{msg.role}: {msg.content}" for msg in history]
    return "Conversation so far:\n" + "\n".join(lines)


def _chunk_to_event(chunk: AgenticChunk) -> AgentEvent | None:
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


class EmergentLoop(Stage[str, str]):
    """Drives one interactive turn — either ``/plan`` delegation or
    the pydantic-ai ``Agent.iter()`` emergent loop.

    Sets ``interactive_mode._final_text`` for the mode's post-pipeline
    :class:`ModeCompletedEvent`.
    """

    name: ClassVar[str] = "agentic-loop"

    def __init__(self, *, interactive_mode: InteractiveMode) -> None:
        self._mode = interactive_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: str,
    ) -> AsyncIterator[AgentEvent | str]:
        user_input = input
        router = harness.router
        prior = self._mode._captured_prior
        root = (
            Path(self._mode.config.workspace_root)
            if self._mode.config.workspace_root is not None
            else Path.cwd()
        )

        stripped = user_input.strip()
        if stripped == PLAN_COMMAND or stripped.startswith(PLAN_COMMAND + " "):
            preliminary = stripped[len(PLAN_COMMAND) :].strip()
            _LOG.info("[interactive] /plan prefix — routing turn to PlanMode")
            final_text = await delegate_to_plan(harness, preliminary, workspace_root=root)
            self._mode._final_text = final_text
            yield final_text
            return

        final_text = ""
        preamble = _join_nonempty(
            _DEFAULT_SYSTEM_PROMPT,
            self._mode.config.system_prompt,
            _render_prior_context(prior),
        )
        tools = (*readonly_tools(root), run_plan_pipeline_tool(harness, root))
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
        self._mode._final_text = final_text
        yield final_text
