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

Stable tools come from :mod:`molexp.agent.ops` (StructureOps / CodeEnv /
Discovery adapters) — not a grab-bag of hard-coded third-party names.
MCP toolsets open best-effort and pass as ``stream_agentic(toolsets=...)``;
their **names** are never compiled into molexp (auto-discovery law).
Optional lifecycle tools append when ``operation_mode == "lifecycle"``.
``operation_mode`` is behavior only — never a capability mask.

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
from molexp.agent.loops.interactive.lifecycle_tools import lifecycle_tools
from molexp.agent.loops.interactive.mcp_toolsets import open_mcp_toolsets
from molexp.agent.ops import (
    DEFAULT_OPS_PREAMBLE,
    build_ops_tools,
    build_session_context,
)
from molexp.agent.ops.tools import render_discovery_catalog
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

__all__ = [
    "DEFAULT_CODE_LOOP_PREAMBLE",
    "DEFAULT_OPS_PREAMBLE",
    "InteractiveLoop",
    "InteractiveLoopConfig",
]

# Back-compat alias — prefer DEFAULT_OPS_PREAMBLE (no hard-coded MCP tool names).
DEFAULT_CODE_LOOP_PREAMBLE = DEFAULT_OPS_PREAMBLE


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


class InteractiveLoopConfig(BaseModel):
    """Tunables for :class:`InteractiveLoop`.

    Attributes:
        system_prompt: Extra system-prompt text composed **after** the
            ops behavior preamble (or ``behavior_preamble`` override).
        workspace_root: Session workspace root for StructureOps / CodeEnv.
            ``None`` falls back to the current working directory at run
            time.
        context_block: Mount-point context (vision-loop-11) — composed
            after the preamble and ``system_prompt``.
        compaction: Context-compaction policy; pass
            ``CompactionSettings(enabled=False)`` to opt out.
        operation_mode: Behavior label only — **not a capability mask**.
            Ops tools (``code_write`` / ``code_run`` / …) are always
            mounted. ``lifecycle`` additionally adds cancel/harvest.
        behavior_preamble: Override :data:`DEFAULT_OPS_PREAMBLE`. Empty
            keeps the default (stable ops names only — no hard-coded
            third-party MCP tool list).
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
        # MCP toolsets first (runtime catalog); names never hard-coded in molexp.
        toolsets = open_mcp_toolsets(workspace)
        ctx = build_session_context(
            workspace_root=workspace,
            execution_env=runtime.execution_env,
            mcp_toolsets=toolsets,
        )
        tools = tuple(build_ops_tools(ctx))
        if self.config.operation_mode == "lifecycle":
            tools = tools + tuple(lifecycle_tools(workspace_root=workspace))

        # Composition: ops preamble → optional live MCP catalog → user → context.
        preamble = self.config.behavior_preamble or ctx.behavior.system_preamble()
        parts = [preamble.strip()]
        catalog = render_discovery_catalog(ctx)
        if catalog:
            parts.append(catalog)
        if self.config.system_prompt.strip():
            parts.append(self.config.system_prompt.strip())
        if self.config.context_block.strip():
            parts.append(self.config.context_block.strip())
        system = "\n\n".join(parts)

        history = _load_model_history(runtime.session)
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
