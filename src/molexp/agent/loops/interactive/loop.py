"""``InteractiveLoop`` — the emergent, tool-using agentic loop.

The CLI's default interactive loop. Plain ``async def run`` body
drives :meth:`molexp.agent.router.Router.stream_agentic` and forwards
each :data:`AgenticChunk` to the injected sink as the corresponding
:data:`AgentEvent`:

* ``ThinkingDeltaChunk`` → ``ThinkingDeltaEvent``
* ``TextDeltaChunk`` → ``TokenDeltaEvent``
* ``ToolCallChunk``  → ``ToolCallStartedEvent``
* ``ToolResultChunk`` → ``ToolCallCompletedEvent``
* ``FinalChunk``     → the assistant's terminal text (captured + appended
  to the session entry-tree; emitted as ``ModeCompletedEvent``)

Read-only tools are pulled from
:func:`~molexp.agent.loops.interactive.tools.readonly_tools` and passed
to ``stream_agentic`` as the ``tools=`` kwarg; the loop body itself is
pydantic-ai's native ``Agent.iter()``, reached through the Router
Protocol — this module imports nothing from pydantic-ai directly.

The harness's planning pipeline lives in ``molexp.harness.PlanMode`` (a
harness ``Mode``), reached through the ``AgentGateway`` Protocol — not from
this agent loop. See ``examples/harness/plan_mode_live.py`` for the
end-to-end flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict

from molexp.agent.events import (
    AsyncIteratorEventSink,
    ModeCompletedEvent,
    ModeStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.loop import AgentLoop, AgentRunResult
from molexp.agent.loops.interactive.tools import readonly_tools
from molexp.agent.router import (
    FinalChunk,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.runtime import AgentRuntime

_LOG = get_logger(__name__)

__all__ = ["InteractiveLoop", "InteractiveLoopConfig"]


class InteractiveLoopConfig(BaseModel):
    """Tunables for :class:`InteractiveLoop`.

    Attributes:
        system_prompt: Extra system-prompt text appended to the
            built-in interactive-assistant preamble.
        workspace_root: Directory the read-only tools are confined to.
            ``None`` falls back to the current working directory at run
            time.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    workspace_root: Path | None = None


class InteractiveLoop(AgentLoop):
    """The emergent tool-using loop — the CLI's default interactive loop."""

    name = "interactive"

    def __init__(self, *, config: InteractiveLoopConfig | None = None) -> None:
        self.config = config or InteractiveLoopConfig()

    async def run(
        self,
        *,
        runtime: AgentRuntime,
        sink: AsyncIteratorEventSink,
        user_input: str,
    ) -> None:
        """Drive one interactive turn; forward router chunks to ``sink``."""
        await sink(ModeStartedEvent(mode_name=self.name, user_input=user_input))
        runtime.router.clear_usage()
        runtime.session.append_message(Message(role="user", content=user_input))

        workspace = self.config.workspace_root or Path.cwd()
        tools = tuple(readonly_tools(workspace_root=workspace))

        final_text = ""
        async for chunk in runtime.router.stream_agentic(
            prompt=user_input,
            system=self.config.system_prompt,
            tools=tools,
        ):
            if isinstance(chunk, ThinkingDeltaChunk):
                await sink(ThinkingDeltaEvent(text=chunk.text))
            elif isinstance(chunk, TextDeltaChunk):
                await sink(TokenDeltaEvent(text=chunk.text))
            elif isinstance(chunk, ToolCallChunk):
                await sink(
                    ToolCallStartedEvent(
                        tool_name=chunk.tool_name,
                        args_summary=chunk.args_summary,
                    )
                )
            elif isinstance(chunk, ToolResultChunk):
                await sink(
                    ToolCallCompletedEvent(
                        tool_name=chunk.tool_name,
                        result_summary=chunk.result_summary,
                        ok=chunk.ok,
                    )
                )
            elif isinstance(chunk, FinalChunk):
                final_text = chunk.text

        runtime.session.append_message(Message(role="assistant", content=final_text))
        breakdown = runtime.router.snapshot_usage()
        _LOG.info(
            f"[interactive] turn done — usage in={breakdown.total.input_tokens} "
            f"out={breakdown.total.output_tokens} reqs={breakdown.total.requests}"
        )
        run_result = AgentRunResult(
            text=final_text,
            messages=runtime.session.build_context(),
            usage=breakdown.total,
            usage_breakdown=breakdown,
        )
        await sink(
            ModeCompletedEvent(
                text=final_text,
                result=run_result.model_dump(mode="json"),
            )
        )
