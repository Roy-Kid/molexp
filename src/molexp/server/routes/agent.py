"""Agent routes — session lifecycle over the rebuilt ``AgentRunner`` surface.

Spec ``agent-live-event-streaming-ui-00a`` relit the read/create session
calls (``create_session`` / ``get_session`` / ``list_sessions``) on top of the
:mod:`molexp.server.agent_runtime` registry; the approval/message/stream calls
stay 503-stubbed until specs 00b / 00c land.

These are plain ``async``/``def`` functions (not FastAPI endpoints): the real
HTTP surface is ``routes/agent_tasks.py``, which calls them directly with an
explicit ``workspace``. They reach the process-singleton
:class:`~molexp.server.agent_runtime.AgentSessionRegistry` via
:func:`~molexp.server.dependencies.get_agent_runtime`.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..dependencies import get_agent_runtime
from ..schemas import (
    AgentSessionListResponse,
    AgentSessionResponse,
    MessageResponse,
    SessionEventResponse,
    SessionStatsResponse,
)

if TYPE_CHECKING:
    from molexp.agent.runner import AgentRunner
    from molexp.server.agent_runtime import AgentSessionRuntime
    from molexp.workspace.workspace import Workspace

    from ..schemas import (
        GoalCreateRequest,
        UserMessageCreateRequest,
    )

# NOTE: no ``/api`` here — the router is mounted under the global ``/api``
# prefix by ``create_app`` (a hardcoded ``/api/agent`` used to produce live
# ``/api/api/agent/*`` paths).
router = APIRouter(prefix="/agent", tags=["agent"])

# Copied from routes/molq.py:27 (the canonical SSE idiom) — not imported, so the
# agent stream stays decoupled from the molq job dashboard.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


_GONE_DETAIL = (
    "This legacy agent HTTP route is retired. Live agent surfaces: "
    "/api/agent-tasks (sessions), /api/agent/provider and /api/agent/mcp/servers "
    "(settings), /api/approvals (approvals inbox)."
)


def _gone() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=_GONE_DETAIL,
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    name="agent_disabled",
)
async def agent_disabled(path: str) -> None:  # noqa: ARG001
    raise _gone()


# ── Runner construction (a test seam) ───────────────────────────────────────
#
# ``create_session`` builds one :class:`AgentRunner` per session. Production
# resolves the model from in-code ``molexp.config`` (``agent.model``, bridged
# from ``~/.molexp/config.json`` at server startup); tests install a factory
# that injects a fake/scripted ``Router`` so no real LLM is constructed.
# Server must not import ``molexp.cli`` (layer inversion), so the model is
# read from ``molexp.config`` / the shared operator-config loader rather than
# the CLI module.

RunnerFactory = Callable[["Workspace"], "AgentRunner"]

_runner_factory: RunnerFactory | None = None


def set_runner_factory(factory: RunnerFactory | None) -> None:
    """Install (or clear) the runner factory used by :func:`create_session`."""
    global _runner_factory
    _runner_factory = factory


def reset_runner_factory() -> None:
    """Drop the runner factory so production model resolution applies again."""
    global _runner_factory
    _runner_factory = None


def _configured_model() -> str | None:
    """Return the agent model registered in in-code ``molexp.config``.

    The canonical key is ``"agent.model"`` (same spelling as the CLI's
    ``molexp config set agent.model <id>``; the server startup bridge —
    :func:`molexp.services.operator_config.bridge_operator_config` — populates
    it from ``~/.molexp/config.json``). The legacy flat ``"agent_model"``
    key is still honoured for in-code users.
    """
    import molexp
    from molexp.services.operator_config import AGENT_MODEL_KEY, LEGACY_AGENT_MODEL_KEY

    model = molexp.config.get(AGENT_MODEL_KEY) or molexp.config.get(LEGACY_AGENT_MODEL_KEY)
    return model if isinstance(model, str) and model else None


def _configured_models() -> dict[str, str] | None:
    """Return a complete cheap/default/heavy map when one is configured."""
    import molexp
    from molexp.services.operator_config import AGENT_MODELS_KEY

    value = molexp.config.get(AGENT_MODELS_KEY)
    if value is None or not callable(getattr(value, "get", None)):
        return None
    models = {tier: value.get(tier) for tier in ("cheap", "default", "heavy")}
    if not all(isinstance(model, str) and model for model in models.values()):
        return None
    return {tier: str(model) for tier, model in models.items()}


def _workspace_root(workspace: Workspace) -> str:
    root = getattr(workspace, "root", None)
    return str(root) if root is not None else ""


def _mount_context(
    workspace: Workspace,
    *,
    project_id: str | None,
    experiment_id: str | None,
    run_id: str | None,
) -> tuple[str, Path | None]:
    """Build the mount-context block + session anchor dir; bad scope → 404.

    Returns ``("", None)`` for an unscoped session. The anchor is the mounted
    entity's directory — the session's on-disk folder mounts there so storage
    location and context agree (vision-loop-11 Design §3).
    """
    from molexp.services.agent_context import mount_session_scope

    try:
        # ValueError: deeper id without its parents; LookupError: the whole
        # workspace *NotFoundError hierarchy (they subclass LookupError).
        # No silent downgrade: an experiment/run id without its parents is
        # rejected, never treated as "unscoped".
        return mount_session_scope(
            workspace,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _build_runner(
    workspace: Workspace, context_block: str = "", session_anchor: Path | None = None
) -> AgentRunner:
    """Construct the :class:`AgentRunner` for a new session.

    Uses the installed test factory when present; otherwise resolves the model
    from ``molexp.config`` and raises a 503 pre-flight when none is configured
    (rather than constructing an empty, never-answering session).
    ``context_block`` is the vision-loop-11 mount snapshot the loop composes
    after its base system prompt.
    """
    if _runner_factory is not None:
        return _runner_factory(workspace)

    models = _configured_models()
    model = _configured_model()
    if models is None and not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No agent model is configured. Run `molexp config set "
                "agent.model <id>` or register molexp.config['agent.model'] "
                "in code before creating an agent session."
            ),
        )

    from molexp.agent import AgentRunner
    from molexp.agent.loops import InteractiveLoop, InteractiveLoopConfig

    root = getattr(workspace, "root", None)
    workspace_root = Path(str(root)) if root is not None else None
    loop = InteractiveLoop(
        config=InteractiveLoopConfig(
            workspace_root=workspace_root,
            context_block=context_block,
            operation_mode="lifecycle",
        )
    )
    runner_config = {"models": models} if models is not None else {"model": model}
    return AgentRunner(
        loop=loop, workspace=workspace_root, session_anchor=session_anchor, **runner_config
    )


# ── Wire translation (runtime objects never cross response_model) ────────────


def _event_to_wire(event: Any) -> SessionEventResponse:  # noqa: ANN401 — opaque AgentEvent
    dumped = event.model_dump(mode="json")
    ts = str(dumped.pop("timestamp", ""))
    return SessionEventResponse(type=str(dumped.get("kind", "")), ts=ts, payload=dumped)


def _to_session_response(
    runtime: AgentSessionRuntime,
    *,
    plan_mode: bool = False,
    skill_id: str | None = None,
) -> AgentSessionResponse:
    """Translate a live runtime into its frozen wire ``AgentSessionResponse``."""
    events = [_event_to_wire(event) for event in runtime.events()]
    return AgentSessionResponse(
        sessionId=runtime.session_id,
        status=runtime.status(),
        goalDescription=runtime.goal,
        createdAt=runtime.created_at,
        events=events,
        stats=SessionStatsResponse(events=len(events), startedAt=runtime.created_at),
        planMode=plan_mode,
        skillId=skill_id,
    )


# ── Relit session lifecycle (00a) ───────────────────────────────────────────


async def create_session(
    request: GoalCreateRequest,
    *,
    workspace: Workspace,
) -> AgentSessionResponse:
    """Create a session, kick its first background turn, return the wire shape."""
    context_block, session_anchor = _mount_context(
        workspace,
        project_id=request.project_id,
        experiment_id=request.experiment_id,
        run_id=request.run_id,
    )
    runner = _build_runner(workspace, context_block, session_anchor)
    session_id = secrets.token_hex(6)
    session = runner.session(session_id)
    runtime = get_agent_runtime().create(
        workspace_root=_workspace_root(workspace),
        runner=runner,
        session=session,
        goal=request.description,
        user_input=request.description,
    )
    return _to_session_response(runtime, plan_mode=request.plan_mode, skill_id=request.skill_id)


def list_sessions(*, workspace: Workspace) -> AgentSessionListResponse:
    """List the live sessions registered under ``workspace``."""
    runtimes = get_agent_runtime().list_runtimes(_workspace_root(workspace))
    sessions = [_to_session_response(runtime) for runtime in runtimes]
    return AgentSessionListResponse(sessions=sessions, total=len(sessions))


def get_session(session_id: str, *, workspace: Workspace) -> AgentSessionResponse:
    """Return one live session by id, or 404 when it is not registered."""
    runtime = get_agent_runtime().get(_workspace_root(workspace), session_id)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"agent session {session_id!r} not found",
        )
    return _to_session_response(runtime)


# ── Still-stubbed calls (owned by 00b / 00c) ────────────────────────────────


async def stream_events(session_id: str, *, workspace: Workspace) -> StreamingResponse:
    """Stream a session's live ``AgentEvent`` flow as Server-Sent Events.

    Fails fast with 404 (before any stream byte) when the session is not
    registered. Otherwise frames each event as ``data: {json}\\n\\n`` in
    replay-then-tail order, closes with a terminal ``done`` frame after the
    turn's ``loop_completed``, and emits exactly one ``error`` frame (then a
    clean close) when the turn ended in failure.
    """
    runtime = get_agent_runtime().get(_workspace_root(workspace), session_id)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"agent session {session_id!r} not found",
        )

    async def _generate() -> AsyncGenerator[str, None]:
        from molexp.server.agent_runtime.serialize import (
            done_frame,
            error_frame,
            event_to_sse_frame,
        )

        try:
            async for event in runtime.subscribe_events():
                yield event_to_sse_frame(event)
        except Exception as exc:  # streaming failure → one error frame, clean close
            yield error_frame(str(exc))
            return
        if runtime.status() == "failed":
            yield error_frame(str(runtime.error) if runtime.error else "turn failed")
        else:
            yield done_frame()

    return StreamingResponse(_generate(), media_type="text/event-stream", headers=_SSE_HEADERS)


async def post_user_message(
    session_id: str,
    request: UserMessageCreateRequest,
    *,
    workspace: Workspace,
) -> MessageResponse:
    """Start a follow-up turn on an existing session from a user message.

    Resolves the live runtime (404 when absent). A message arriving while the
    current turn is still ``running`` is rejected with 409 (no interleaving);
    otherwise a fresh background turn is spawned on the same ``Session`` so the
    conversation continues, and a wire ``MessageResponse`` is returned.
    """
    runtime = get_agent_runtime().get(_workspace_root(workspace), session_id)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"agent session {session_id!r} not found",
        )
    if runtime.status() == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a turn is already in flight for this session",
        )
    runtime.start_turn(request.content)
    return MessageResponse(message="accepted")


async def cancel_session(session_id: str, *, workspace: Workspace) -> MessageResponse:
    """Cancel the in-flight turn for ``session_id`` (idempotent).

    404 when the session is not registered. When no turn is running the call
    is a no-op success so the UI can always offer Stop without racing the
    terminal frame.
    """
    runtime = get_agent_runtime().get(_workspace_root(workspace), session_id)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"agent session {session_id!r} not found",
        )
    runtime.cancel()
    await runtime.await_finished()
    return MessageResponse(message="cancelled")


async def delete_session(session_id: str, *, workspace: Workspace) -> MessageResponse:
    """Cancel any in-flight turn and drop the runtime from the registry."""
    registry = get_agent_runtime()
    root = _workspace_root(workspace)
    runtime = registry.get(root, session_id)
    if runtime is not None:
        runtime.cancel()
        await runtime.await_finished()
        registry.remove(root, session_id)
    return MessageResponse(message="deleted")


__all__ = [
    "cancel_session",
    "create_session",
    "delete_session",
    "get_session",
    "list_sessions",
    "post_user_message",
    "reset_runner_factory",
    "router",
    "set_runner_factory",
    "stream_events",
]
