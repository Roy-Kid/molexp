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
  to the session entry-tree; emitted as ``LoopCompletedEvent``)

Tools always include read-only file tools, knowledge tools, and the
code tools (:func:`~molexp.agent.loops.interactive.code_tools.code_tools`
— ``write_file`` / ``execute_python``). Optional lifecycle tools
(cancel/harvest) append when ``operation_mode == "lifecycle"``.
``operation_mode`` does **not** strip write/exec — it only gates the
extra lifecycle pair.

MCP toolsets from :class:`~molexp.agent.mcp.store.McpStore` are opened
best-effort via :func:`~molexp.agent.loops.interactive.mcp_toolsets.open_mcp_toolsets`
and passed as ``stream_agentic(toolsets=...)``. A single server build
failure is logged and skipped; the turn still completes. Bare tools go
to ``tools=``; the loop body itself is pydantic-ai's native
``Agent.iter()``, reached through the Router Protocol — this module
imports nothing from pydantic-ai directly.

The harness's planning pipeline lives in ``molexp.harness.PlanMode`` (a
harness ``Mode``), reached through the ``AgentGateway`` Protocol — not from
this agent loop. See ``examples/harness/experiment_pipeline.py`` for the
end-to-end flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mollog import get_logger
from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.compaction import CompactionSettings
from molexp.agent.events import (
    AsyncIteratorEventSink,
    LoopCompletedEvent,
    LoopStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.loop import AgentLoop, AgentRunResult
from molexp.agent.loops._compact import maybe_compact
from molexp.agent.loops.interactive.code_tools import code_tools
from molexp.agent.loops.interactive.knowledge_tools import knowledge_tools
from molexp.agent.loops.interactive.lifecycle_tools import lifecycle_tools
from molexp.agent.loops.interactive.mcp_toolsets import open_mcp_toolsets
from molexp.agent.loops.interactive.tools import readonly_tools
from molexp.agent.router import (
    FinalChunk,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.session_storage import JsonlSessionStorage
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.runtime import AgentRuntime
    from molexp.agent.session import Session

_LOG = get_logger(__name__)

__all__ = ["DEFAULT_CODE_LOOP_PREAMBLE", "InteractiveLoop", "InteractiveLoopConfig"]


def _session_messages_path(session: Session) -> Path | None:
    """Return ``…/messages.jsonl`` beside the session entry tree, if on-disk."""
    storage = session.storage
    if not isinstance(storage, JsonlSessionStorage):
        return None
    from molexp.agent.folders import MESSAGES_FILENAME

    return storage.directory / MESSAGES_FILENAME


def _load_model_history(session: Session) -> tuple[object, ...]:
    path = _session_messages_path(session)
    if path is None or not path.exists():
        return ()
    from molexp.agent._pydanticai.messages_codec import load_model_messages

    try:
        return load_model_messages(path.read_bytes())
    except Exception as exc:
        _LOG.warning(f"[interactive] could not load model history ({exc!r}); starting fresh")
        return ()


def _save_model_history(session: Session, messages_json: bytes) -> None:
    path = _session_messages_path(session)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(messages_json)


#: Default behavior contract for the code-loop (consult → write → exec).
#: Always composed into ``system`` before user ``system_prompt`` and
#: ``context_block``. Tests assert the marked substrings.
DEFAULT_CODE_LOOP_PREAMBLE = """\
You are molexp's interactive research agent. Work by consulting tools, \
then writing and running Python — not by inventing parallel APIs.

1. Prefer `molmcp` tools (`molmcp__*` / `molexp_*`) for discovery and \
workspace scaffold (layout, materialize, add project/experiment). \
MCP is not a batch science executor — do not use it to run large sweeps.
2. Implement experiments, parameter sweeps, recovery, and analysis by \
writing Python against molexp APIs (see examples/agent/code_loop_golden_path.py). \
Use `write_file` then `execute_python` to run scripts under the workspace.
3. Plot with `import molplot` in that Python — molexp has no built-in plot tool.
4. A short plan before multi-step work is fine; planning never locks tools.
"""


class InteractiveLoopConfig(BaseModel):
    """Tunables for :class:`InteractiveLoop`.

    Attributes:
        system_prompt: Extra system-prompt text composed **after**
            :data:`DEFAULT_CODE_LOOP_PREAMBLE` (or ``behavior_preamble``).
        workspace_root: Directory file/knowledge/code tools are confined
            to. ``None`` falls back to the current working directory at
            run time.
        context_block: Mount-point context (vision-loop-11) — a rendered
            snapshot of the entity this session is attached to, composed
            after the preamble and ``system_prompt``. The block is built
            by ``services.agent_context``.
        compaction: Context-compaction policy; pass
            ``CompactionSettings(enabled=False)`` to opt out.
        operation_mode: **Behavior label only**, not a capability mask.
            ``write_file`` / ``execute_python`` are always mounted.
            ``lifecycle`` additionally adds cancel/harvest tools (no
            harness ApprovalGate — use Plan/Curate for gated
            execute/resume/rerun). Legacy ``readonly`` does **not**
            strip code tools.
        behavior_preamble: Override the default code-loop preamble.
            Empty string keeps :data:`DEFAULT_CODE_LOOP_PREAMBLE`.
            Set a custom string to replace it entirely.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    workspace_root: Path | None = None
    context_block: str = ""
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    operation_mode: str = "readonly"
    behavior_preamble: str = ""


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
        await sink(LoopStartedEvent(loop_name=self.name, user_input=user_input))
        runtime.router.clear_usage()
        runtime.session.append_message(Message(role="user", content=user_input))
        await maybe_compact(
            runtime=runtime,
            sink=sink,
            settings=self.config.compaction,
            loop_name=self.name,
        )

        workspace = self.config.workspace_root or Path.cwd()
        # Read + knowledge + write/exec (always). Lifecycle cancel/harvest is
        # optional; operation_mode never strips code tools.
        tools = (
            tuple(readonly_tools(workspace_root=workspace))
            + tuple(knowledge_tools(workspace_root=workspace))
            + tuple(
                code_tools(
                    workspace_root=workspace,
                    execution_env=runtime.execution_env,
                )
            )
        )
        if self.config.operation_mode == "lifecycle":
            tools = tools + tuple(lifecycle_tools(workspace_root=workspace))

        # Composition order: behavior preamble → user system_prompt → context_block.
        preamble = self.config.behavior_preamble or DEFAULT_CODE_LOOP_PREAMBLE
        parts = [preamble.strip()]
        if self.config.system_prompt.strip():
            parts.append(self.config.system_prompt.strip())
        if self.config.context_block.strip():
            parts.append(self.config.context_block.strip())
        system = "\n\n".join(parts)

        history = _load_model_history(runtime.session)
        # MCP toolsets: best-effort open; Agent.iter owns enter/exit lifecycle.
        toolsets = open_mcp_toolsets(workspace)
        final_text = ""
        async for chunk in runtime.router.stream_agentic(
            prompt=user_input,
            system=system,
            tools=tools,
            toolsets=toolsets,
            message_history=history,
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
                if chunk.model_messages_json is not None:
                    _save_model_history(runtime.session, chunk.model_messages_json)

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
            LoopCompletedEvent(
                text=final_text,
                result=run_result.model_dump(mode="json"),
            )
        )
