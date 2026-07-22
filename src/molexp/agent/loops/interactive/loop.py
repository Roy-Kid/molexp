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
    LoopSuspendedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.loop import AgentLoop, AgentRunResult
from molexp.agent.loops._compact import maybe_compact
from molexp.agent.loops.hooks import LoopHooks, LoopState, invoke_should_stop
from molexp.agent.loops.interactive.lifecycle_tools import lifecycle_tools
from molexp.agent.loops.interactive.mcp_toolsets import (
    list_mcp_tool_specs,
    open_mcp_toolsets,
)
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
    from collections.abc import AsyncIterator

    from molexp.agent.router import AgenticChunk
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


def _session_history_paths(session: Session) -> tuple[Path, Path] | None:
    """Return ``(messages.jsonl, messages.leaf)`` beside the entry tree, if on-disk.

    ``None`` for a non-disk (in-memory) session — it carries no lossless blob,
    so there is nothing to reconcile.
    """
    storage = session.storage
    if not isinstance(storage, JsonlSessionStorage):
        return None
    from molexp.agent.folders import MESSAGES_FILENAME, MESSAGES_LEAF_FILENAME

    directory = storage.directory
    return directory / MESSAGES_FILENAME, directory / MESSAGES_LEAF_FILENAME


def _stamp_on_active_branch(session: Session, stamp_path: Path) -> bool:
    """Whether the blob's leaf stamp names an entry on the session's active branch.

    A present, non-empty stamp whose id is on ``path_to_root`` is a linear
    continuation of the branch the blob was produced against; a branch or
    resume moves the tip off that lineage, so the stamp no longer matches.
    """
    if not stamp_path.exists():
        return False
    try:
        stamp = stamp_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not stamp:
        return False
    return any(entry.id == stamp for entry in session.path_to_root())


def _reseed_from_entry_tree(session: Session) -> tuple[object, ...]:
    """Rebuild LLM history semantically from the canonical entry tree."""
    from molexp.agent._pydanticai.messages_codec import model_messages_from_messages

    return model_messages_from_messages(session.build_context())


def _load_model_history(session: Session) -> tuple[object, ...]:
    """Reconcile the lossless blob against the active leaf; entry tree is canonical.

    The lossless ``messages.jsonl`` blob is fed back *only* when its leaf stamp
    names an entry on the current branch (a linear continuation) **and** the
    blob file is present. Any branch/resume (stamp off the active lineage), a
    missing stamp, a deleted blob, or an unreadable blob discards it and
    reseeds from ``build_context()``. A non-disk session keeps today's empty
    history (it has no blob to reconcile).
    """
    paths = _session_history_paths(session)
    if paths is None:
        return ()
    msgs_path, stamp_path = paths
    if _stamp_on_active_branch(session, stamp_path) and msgs_path.exists():
        from molexp.agent._pydanticai.messages_codec import load_model_messages

        try:
            return load_model_messages(msgs_path.read_bytes())
        except Exception as exc:
            _LOG.warning(
                f"[interactive] lossless history unreadable ({exc!r}); reseeding from entry tree"
            )
    return _reseed_from_entry_tree(session)


def _save_model_history(session: Session, messages_json: bytes | None, *, leaf_id: str) -> None:
    """Persist the lossless blob plus a ``leaf_id`` stamp.

    ``messages_json is None`` (a router that emitted no lossless history) is a
    no-op — the prior blob + stamp survive untouched.
    """
    if messages_json is None:
        return
    paths = _session_history_paths(session)
    if paths is None:
        return
    msgs_path, stamp_path = paths
    msgs_path.parent.mkdir(parents=True, exist_ok=True)
    msgs_path.write_bytes(messages_json)
    stamp_path.write_text(leaf_id, encoding="utf-8")


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
        max_passes: Hard safety bound on the outer steering loop — the
            maximum number of inner ReAct passes a single turn may run
            before it stops regardless of the ``should_stop`` guard (an
            always-``deny`` guard can never spin forever). ``>= 1``;
            defaults to 8. Only the outer loop (``hooks`` present) can run
            more than one pass — with ``hooks=None`` a turn is always a
            single pass.
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    workspace_root: Path | None = None
    context_block: str = ""
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    operation_mode: str = "readonly"
    behavior_preamble: str = ""
    max_passes: int = Field(default=8, ge=1)


class InteractiveLoop(AgentLoop):
    """The emergent tool-using loop — the CLI's default interactive loop.

    Structurally a Pi-style **outer** loop wrapping pydantic-ai's inner
    model↔tool ReAct (which stays 100% behind
    :meth:`~molexp.agent.router.Router.stream_agentic`). Each turn runs one or
    more inner passes; the outer loop only adds steering, follow-up,
    ``should_stop``-deny feedback, and durable suspend — driven entirely by an
    injected :class:`~molexp.agent.loops.hooks.LoopHooks` bundle.

    Hooks seam (constructor injection):

    * ``hooks=None`` → a single pass that always proceeds, terminating in a
      :class:`~molexp.agent.events.LoopCompletedEvent` — byte-identical to the
      pre-refactor loop.
    * a ``should_stop`` returning
      :meth:`~molexp.agent.loops.hooks.HookOutcome.deny` **steers**: the deny
      message is appended as the next user turn and another pass runs.
    * a ``should_stop`` returning
      :meth:`~molexp.agent.loops.hooks.HookOutcome.suspend` **suspends**: the
      loop stops with a :class:`~molexp.agent.events.LoopSuspendedEvent`
      carrying the suspend token — no pending record, since the entry tree +
      leaf pointer are already durable.

    History reconciliation (the session **entry tree is canonical**): the
    lossless pydantic-ai ``model_messages`` blob beside it (``messages.jsonl``)
    is a *cache*. It is reused only when its leaf stamp (``messages.leaf``)
    names an entry on the active branch — a linear continuation. A branch,
    resume, or a deleted ``messages.jsonl`` discards the blob and reseeds
    history from ``session.build_context()`` via
    :func:`~molexp.agent._pydanticai.messages_codec.model_messages_from_messages`;
    a sibling branch's model history is never inherited, and deleting
    ``messages.jsonl`` loses no session state.

    See spec ``plan-emergent-02-loop-refactor``.
    """

    # On-disk Agent folder name under the workspace root (also loop_name on events).
    name = "agent"

    def __init__(
        self,
        *,
        config: InteractiveLoopConfig | None = None,
        hooks: LoopHooks | None = None,
    ) -> None:
        self.config = config or InteractiveLoopConfig()
        self.hooks = hooks

    async def run(
        self,
        *,
        runtime: AgentRuntime,
        sink: AsyncIteratorEventSink,
        user_input: str,
    ) -> None:
        """Drive one interactive turn as a Pi-style outer loop over inner ReAct passes.

        Per-turn setup (loop-started event, usage reset, the first user turn,
        compaction, tool/system assembly) runs once; the outer ``while`` then
        drives one or more inner :meth:`~molexp.agent.router.Router.stream_agentic`
        passes, consulting the injected ``should_stop`` guard after each. With
        ``hooks=None`` the loop runs exactly one pass and completes.
        """
        # ── per-turn setup (once) ────────────────────────────────────────────
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
        # MCP toolsets + runtime list_tools catalog (auto-discovery law).
        toolsets = open_mcp_toolsets(workspace)
        try:
            mcp_specs = await list_mcp_tool_specs(toolsets)
        except Exception as exc:
            # Catalog is best-effort; missing/broken MCP must not abort the turn.
            _LOG.warning(f"[interactive] MCP catalog list failed: {exc!r}")
            mcp_specs = ()
        ctx = build_session_context(
            workspace_root=workspace,
            execution_env=runtime.execution_env,
            mcp_toolsets=toolsets,
            mcp_tool_specs=mcp_specs,
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

        # ── outer steering loop (per-pass) ───────────────────────────────────
        passes = 0
        current = user_input
        final_text = ""
        last_tool_name = ""
        suspended = False
        suspend_reason = ""
        while True:
            passes += 1
            history = _load_model_history(runtime.session)
            final_text = ""
            pending_blob: bytes | None = None
            async for chunk in self._stream(
                runtime=runtime,
                prompt=current,
                system=system,
                tools=tools,
                toolsets=toolsets,
                history=history,
            ):
                if isinstance(chunk, ThinkingDeltaChunk):
                    await sink(ThinkingDeltaEvent(text=chunk.text))
                elif isinstance(chunk, TextDeltaChunk):
                    await sink(TokenDeltaEvent(text=chunk.text))
                elif isinstance(chunk, ToolCallChunk):
                    last_tool_name = chunk.tool_name
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
                        pending_blob = chunk.model_messages_json

            runtime.session.append_message(Message(role="assistant", content=final_text))
            # Stamp the lossless blob against the assistant tip it was produced at.
            _save_model_history(
                runtime.session, pending_blob, leaf_id=runtime.session.leaf_id or ""
            )

            # Outer-loop decision: proceed / steer / suspend, bounded by max_passes.
            if (
                self.hooks is None
                or self.hooks.should_stop is None
                or passes >= self.config.max_passes
            ):
                break
            outcome = await invoke_should_stop(
                self.hooks.should_stop,
                state=LoopState(step=passes, last_tool=last_tool_name, draft_final=final_text),
            )
            if outcome.is_proceed:
                break
            if outcome.is_suspend:
                suspended = True
                suspend_reason = outcome.token
                break
            # deny == steer: inject the deny message as the next user turn, re-run.
            runtime.session.append_message(Message(role="user", content=outcome.message))
            current = outcome.message

        # ── terminal event ───────────────────────────────────────────────────
        if suspended:
            await sink(
                LoopSuspendedEvent(
                    reason=suspend_reason,
                    leaf_id=runtime.session.leaf_id or "",
                )
            )
            return

        breakdown = runtime.router.snapshot_usage()
        _LOG.info(
            f"[interactive] turn done — passes={passes} "
            f"usage in={breakdown.total.input_tokens} "
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

    def _stream(
        self,
        *,
        runtime: AgentRuntime,
        prompt: str,
        system: str,
        tools: tuple[object, ...],
        toolsets: tuple[object, ...],
        history: tuple[object, ...],
    ) -> AsyncIterator[AgenticChunk]:
        """Open one inner ReAct pass, wiring the hook bundle only when present.

        With ``hooks=None`` the hook kwargs are **omitted entirely** (not passed
        as ``None``), so the call is byte-identical to the pre-refactor loop and
        a router that predates the hook kwargs still accepts it.
        """
        if self.hooks is None:
            return runtime.router.stream_agentic(
                prompt=prompt,
                system=system,
                tools=tools,
                toolsets=toolsets,
                message_history=history,
            )
        return runtime.router.stream_agentic(
            prompt=prompt,
            system=system,
            tools=tools,
            toolsets=toolsets,
            message_history=history,
            before_tool=self.hooks.before_tool,
            after_tool=self.hooks.after_tool,
            should_stop=self.hooks.should_stop,
        )
