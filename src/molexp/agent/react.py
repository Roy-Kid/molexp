"""One ReAct turn — pydantic-ai ``stream_agentic``, no outer steering loop.

ReAct already stops when the model stops calling tools. Session history,
ops/MCP tool assembly, and AgentEvent projection live here so
:class:`~molexp.agent.runner.AgentRunner` does not grow a second loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mollog import get_logger

from molexp.agent.events import (
    EventSink,
    LoopCompletedEvent,
    LoopStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.loop import AgentRunResult
from molexp.agent.loops._compact import maybe_compact
from molexp.agent.mcp.catalog import McpCatalog, filter_toolsets
from molexp.agent.ops import build_ops_tools, build_session_context
from molexp.agent.ops.builtins import declared_requirements
from molexp.agent.ops.lifecycle import lifecycle_tools
from molexp.agent.ops.preamble import DefaultOpsBehavior, FullOpsBehavior
from molexp.agent.ops.surface import SurfaceKey, required_keys, surface_for_mode
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
    from molexp.agent.compaction import CompactionSettings
    from molexp.agent.execution_env import ExecutionEnv
    from molexp.agent.loops.hooks import LoopHooks
    from molexp.agent.runtime import AgentRuntime
    from molexp.agent.session import Session

_LOG = get_logger(__name__)

__all__ = ["assemble_react", "drive_react", "run_react_turn"]


def _session_history_paths(session: Session) -> tuple[Path, Path] | None:
    """Return ``(messages.jsonl, messages.leaf)`` beside the entry tree, if on-disk."""
    storage = session.storage
    if not isinstance(storage, JsonlSessionStorage):
        return None
    from molexp.agent.folders import MESSAGES_FILENAME, MESSAGES_LEAF_FILENAME

    directory = storage.directory
    return directory / MESSAGES_FILENAME, directory / MESSAGES_LEAF_FILENAME


def _stamp_on_active_branch(session: Session, stamp_path: Path) -> bool:
    """Whether the blob's leaf stamp names an entry on the session's active branch."""
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


def load_model_history(session: Session) -> tuple[object, ...]:
    """Reuse the lossless blob only when its leaf stamp is on the active branch."""
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
                f"[react] lossless history unreadable ({exc!r}); reseeding from entry tree"
            )
    return _reseed_from_entry_tree(session)


def save_model_history(session: Session, messages_json: bytes | None, *, leaf_id: str) -> None:
    """Persist the lossless blob plus a ``leaf_id`` stamp."""
    if messages_json is None:
        return
    paths = _session_history_paths(session)
    if paths is None:
        return
    msgs_path, stamp_path = paths
    msgs_path.parent.mkdir(parents=True, exist_ok=True)
    msgs_path.write_bytes(messages_json)
    stamp_path.write_text(leaf_id, encoding="utf-8")


class ReactAssembly:
    """Ops tools + MCP toolsets + composed system prompt for one ReAct."""

    __slots__ = ("catalog", "system", "tools", "toolsets")

    def __init__(
        self,
        *,
        tools: tuple[object, ...],
        toolsets: tuple[object, ...],
        system: str,
        catalog: McpCatalog | None,
    ) -> None:
        self.tools = tools
        self.toolsets = toolsets
        self.system = system
        self.catalog = catalog


async def assemble_react(
    *,
    workspace: Path,
    execution_env: ExecutionEnv,
    operation_mode: str = "chat",
    system_prompt: str = "",
    context_block: str = "",
    extra_tools: tuple[object, ...] = (),
    behavior_preamble: str = "",
) -> ReactAssembly:
    """Open MCP (best-effort) and build the ops + injected tool surface."""
    mode = (operation_mode or "chat").strip().lower()
    spec = surface_for_mode(mode)
    confine = "agent/.scratch" if spec.name == "chat" else None
    behavior = DefaultOpsBehavior() if spec.name == "chat" else FullOpsBehavior()

    catalog = McpCatalog(workspace)
    await catalog.open()
    try:
        mcp_specs = await catalog.list_specs()
    except Exception as exc:
        _LOG.warning(f"[react] MCP catalog list failed: {exc!r}")
        mcp_specs = ()
    declared = declared_requirements()

    def _allow_name(name: str) -> bool:
        return spec.allows(required_keys(name, declared=declared))

    toolsets = filter_toolsets(catalog.toolsets, allow=_allow_name)
    mcp_specs = tuple(s for s in mcp_specs if _allow_name(s.name))
    ctx = build_session_context(
        workspace_root=workspace,
        execution_env=execution_env,
        mcp_toolsets=toolsets,
        mcp_tool_specs=mcp_specs,
        behavior=behavior,
        confine_code_to=confine,
        surface=spec.name,
    )
    tools = tuple(build_ops_tools(ctx, surface=spec.name))
    if SurfaceKey.LIFECYCLE in spec.keys:
        tools = tools + tuple(lifecycle_tools(workspace_root=workspace))
    tools = tools + extra_tools

    preamble = behavior_preamble or ctx.behavior.system_preamble()
    parts = [preamble.strip()]
    catalog_text = render_discovery_catalog(ctx, surface=spec.name)
    if catalog_text:
        parts.append(catalog_text)
    if system_prompt.strip():
        parts.append(system_prompt.strip())
    if context_block.strip():
        parts.append(context_block.strip())
    return ReactAssembly(
        tools=tools,
        toolsets=toolsets,
        system="\n\n".join(p for p in parts if p),
        catalog=catalog,
    )


def _stream_agentic(
    runtime: AgentRuntime,
    *,
    prompt: str,
    system: str,
    tools: tuple[object, ...],
    toolsets: tuple[object, ...],
    history: tuple[object, ...],
    hooks: LoopHooks | None,
) -> AsyncIterator[Any]:
    """Open one ReAct; omit hook kwargs when unused so older fakes still bind."""
    base: dict[str, Any] = {
        "prompt": prompt,
        "system": system,
        "tools": tools,
        "toolsets": toolsets,
        "message_history": history,
    }
    if hooks is None:
        return runtime.router.stream_agentic(**base)
    import inspect

    try:
        params = inspect.signature(runtime.router.stream_agentic).parameters
    except (TypeError, ValueError):
        params = {}
    if "before_tool" in params:
        return runtime.router.stream_agentic(
            **base,
            before_tool=hooks.before_tool,
            after_tool=hooks.after_tool,
            should_stop=hooks.should_stop,
        )
    return runtime.router.stream_agentic(**base)


async def drive_react(
    *,
    runtime: AgentRuntime,
    sink: EventSink,
    prompt: str,
    system: str = "",
    tools: tuple[object, ...] = (),
    toolsets: tuple[object, ...] = (),
    hooks: LoopHooks | None = None,
) -> str:
    """Consume one ``stream_agentic`` and project chunks onto ``sink``.

    Returns the final assistant text. No outer should-stop loop.
    """
    history = load_model_history(runtime.session)
    final_text = ""
    pending_blob: bytes | None = None
    async for chunk in _stream_agentic(
        runtime,
        prompt=prompt,
        system=system,
        tools=tools,
        toolsets=toolsets,
        history=history,
        hooks=hooks,
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
                    artifacts=getattr(chunk, "artifacts", ()) or (),
                )
            )
        elif isinstance(chunk, FinalChunk):
            final_text = chunk.text
            if chunk.model_messages_json is not None:
                pending_blob = chunk.model_messages_json
    runtime.session.append_message(Message(role="assistant", content=final_text))
    save_model_history(runtime.session, pending_blob, leaf_id=runtime.session.leaf_id or "")
    return final_text


async def run_react_turn(
    *,
    runtime: AgentRuntime,
    sink: EventSink,
    user_input: str,
    workspace: Path,
    operation_mode: str = "chat",
    system_prompt: str = "",
    context_block: str = "",
    extra_tools: tuple[object, ...] = (),
    compaction: CompactionSettings | None = None,
    name: str = "agent",
) -> None:
    """One user message → one ReAct → terminal :class:`LoopCompletedEvent`."""
    from molexp.agent.compaction import CompactionSettings as _Settings

    await sink(LoopStartedEvent(loop_name=name, user_input=user_input))
    runtime.router.clear_usage()
    runtime.session.append_message(Message(role="user", content=user_input))
    await maybe_compact(
        runtime=runtime,
        sink=sink,
        settings=compaction or _Settings(),
        loop_name=name,
    )
    assembly = await assemble_react(
        workspace=workspace,
        execution_env=runtime.execution_env,
        operation_mode=operation_mode,
        system_prompt=system_prompt,
        context_block=context_block,
        extra_tools=extra_tools,
    )
    try:
        final_text = await drive_react(
            runtime=runtime,
            sink=sink,
            prompt=user_input,
            system=assembly.system,
            tools=assembly.tools,
            toolsets=assembly.toolsets,
        )
    finally:
        if assembly.catalog is not None:
            await assembly.catalog.aclose()

    breakdown = runtime.router.snapshot_usage()
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
