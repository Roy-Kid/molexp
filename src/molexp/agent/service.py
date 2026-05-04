"""AgentService — public entry point.

The workspace-scoped facade. It owns the live session registry, the
tool registry, and the asyncio task that drives each session. Server
routes import only this class.
"""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import AsyncIterator, Callable

from molexp.agent.model import ModelClient
from molexp.agent.orchestration.events import (
    SessionEvent,
    SessionStarted,
)
from molexp.agent.orchestration.runner import AgentRunner
from molexp.agent.orchestration.session import AgentSession
from molexp.agent.state.config import AgentSettings
from molexp.agent.state.memory import NoopMemoryStore
from molexp.agent.state.sessions import SessionMetadata, SessionStore
from molexp.agent.state.skills import SkillStore
from molexp.agent.state.store import AgentStateStore
from molexp.agent.tools.dispatcher import ToolDispatcher
from molexp.agent.tools.policy import PERMISSIVE_POLICY, ToolPolicy
from molexp.agent.tools.registry import ToolRegistry
from molexp.agent.tools.source import ToolSource
from molexp.agent.types import Goal, SessionStatus, utc_now


RunnerFactory = Callable[[AgentSession], AgentRunner]


class AgentService:
    """Workspace-scoped facade over the agent harness.

    Each service instance owns its own :class:`ToolRegistry`. Native
    tools are registered at construction by walking
    ``molexp.agent.tools.native``.
    """

    AGENT_DIRNAME = ".molexp-agent"

    def __init__(
        self,
        workspace_path: Path,
        settings: AgentSettings | None = None,
        model: ModelClient | None = None,
        registry: ToolRegistry | None = None,
        state: AgentStateStore | None = None,
        runner_factory: RunnerFactory | None = None,
        policy: ToolPolicy = PERMISSIVE_POLICY,
        workspace: object | None = None,
        register_native_tools: bool = True,
        tool_sources: tuple[ToolSource, ...] = (),
    ) -> None:
        self.workspace_path = Path(workspace_path)
        self.settings = settings or AgentSettings()
        self.model = model
        self.registry = registry or ToolRegistry()
        self.state = state or self._default_state()
        self.policy = policy
        self.workspace = workspace
        self._sessions: dict[str, AgentSession] = {}
        self._runner_factory = runner_factory
        self.tool_sources = tuple(tool_sources)
        if register_native_tools:
            self._register_native_tools()
        self._mark_orphaned_sessions_interrupted()

    # Public API ----------------------------------------------------------

    @classmethod
    def from_workspace(
        cls,
        workspace_path: str | Path,
        settings: AgentSettings | None = None,
        model: ModelClient | None = None,
        workspace: object | None = None,
        tool_sources: tuple[ToolSource, ...] = (),
    ) -> "AgentService":
        """Construct a service rooted at ``workspace_path``.

        ``model`` is supplied by the caller — typically the server,
        which resolves a provider config through
        :func:`molexp.agent.model_registry.create_model_client`. The
        harness itself never imports a model plugin.

        ``tool_sources`` lets the caller plug in :class:`ToolSource`
        instances (MCP, etc.); the dispatcher merges their tools into
        the model-facing schema list and routes matching calls to
        ``source.call``.
        """

        return cls(
            workspace_path=Path(workspace_path),
            settings=settings,
            model=model,
            workspace=workspace,
            tool_sources=tool_sources,
        )

    def start_session(self, goal: Goal) -> AgentSession:
        """Register a new session and start its background runner.

        If no :class:`ModelClient` is configured the session is
        registered and metadata persisted, but no task is spawned —
        callers must attach a model before any turn can run.
        """

        session_id = self._new_session_id()
        session = AgentSession(session_id=session_id, goal=goal)
        self._sessions[session_id] = session

        meta = SessionMetadata(
            session_id=session_id,
            goal=goal,
            status=SessionStatus.PENDING,
        )
        self.state.sessions.write_metadata(meta)
        self._spawn_runner(session)
        return session

    def resume_session(self, session_id: str) -> AgentSession:
        """Re-spawn a runner for a ``RESUMABLE`` / ``INTERRUPTED`` session.

        Loads the persisted message history from ``messages.jsonl`` and
        hands it to the runner so it picks up where the previous
        process stopped — the model gets the same context, the turn
        loop continues on the next inbound user message.
        """

        meta = self.state.sessions.read_metadata(session_id)
        if meta is None:
            raise KeyError(f"session '{session_id}' not found")
        if meta.status not in (SessionStatus.RESUMABLE, SessionStatus.INTERRUPTED):
            raise ValueError(
                f"session '{session_id}' is in status {meta.status.value!r}; "
                "only RESUMABLE or INTERRUPTED sessions can be resumed"
            )
        history = list(self.state.sessions.read_messages(session_id))
        session = AgentSession(session_id=session_id, goal=meta.goal)
        self._sessions[session_id] = session
        self.state.sessions.write_metadata(
            SessionMetadata(
                session_id=session_id,
                goal=meta.goal,
                status=SessionStatus.PENDING,
                created_at=meta.created_at,
                updated_at=utc_now(),
                summary=meta.summary,
            )
        )
        self._spawn_runner(session, initial_history=history)
        return session

    def _spawn_runner(
        self,
        session: AgentSession,
        initial_history: list | None = None,
    ) -> None:
        if self.model is None:
            return
        runner = self._build_runner(session)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            runner.drive_session(session, initial_history=initial_history),
            name=f"agent-session-{session.session_id}",
        )
        session.attach_task(task)

    async def emit_session_started(self, session: AgentSession) -> None:
        """Publish the initial :class:`SessionStarted` event.

        Used by the no-model fixture path: when no model is configured
        the runner never spawns, so tests that assert on
        :class:`SessionStarted` arrival call this directly.
        """

        await session.bus.publish(
            SessionStarted(
                session_id=session.session_id,
                goal_description=session.goal.description,
                ts=utc_now(),
            )
        )

    def list_sessions(self) -> tuple[SessionMetadata, ...]:
        """Return persisted session metadata, oldest first."""

        return self.state.sessions.list_sessions()

    def get_session(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def stream_events(self, session_id: str) -> AsyncIterator[SessionEvent] | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.stream_events()

    async def cancel(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.cancel()
        return True

    async def shutdown(self) -> None:
        """Mark every live session interrupted."""

        for session in list(self._sessions.values()):
            session.status = SessionStatus.INTERRUPTED
            await session.bus.close()
            if session.task is not None and not session.task.done():
                session.task.cancel()
        self._sessions.clear()

    # Internals -----------------------------------------------------------

    def _agent_root(self) -> Path:
        return self.workspace_path / self.AGENT_DIRNAME

    def _default_state(self) -> AgentStateStore:
        sessions_root = self._agent_root() / "sessions"
        return AgentStateStore(
            sessions=SessionStore(sessions_root),
            skills=SkillStore(self.workspace_path),
            memory=NoopMemoryStore(),
        )

    def _build_runner(self, session: AgentSession) -> AgentRunner:
        if self._runner_factory is not None:
            return self._runner_factory(session)
        assert self.model is not None
        dispatcher = ToolDispatcher(self.registry, sources=self.tool_sources)
        return AgentRunner(
            model=self.model,
            registry=self.registry,
            store=self.state.sessions,
            workspace=self.workspace,
            dispatcher=dispatcher,
            policy=self.policy,
        )

    def _register_native_tools(self) -> None:
        """Walk ``molexp.agent.tools.native`` and register every tagged tool."""

        from molexp.agent.tools import native as native_pkg
        from molexp.agent.tools.registry import get_native_spec, is_native_tool
        import importlib
        import pkgutil

        for module_info in pkgutil.iter_modules(
            native_pkg.__path__, prefix=f"{native_pkg.__name__}."
        ):
            module = importlib.import_module(module_info.name)
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if not is_native_tool(obj):
                    continue
                spec = get_native_spec(obj)
                self.registry.register(spec, obj)

    def _mark_orphaned_sessions_interrupted(self) -> None:
        """Reconcile orphaned persisted sessions on construction.

        Sessions whose JSON still claims a live status
        (``PENDING`` / ``RUNNING`` / ``AWAITING_*``) belonged to a
        previous process. If their persisted ``messages.jsonl`` has
        replayable history they flip to ``RESUMABLE`` so
        :meth:`resume_session` can pick up where they stopped;
        otherwise they go to ``INTERRUPTED`` and the UI surfaces
        "ended unexpectedly".
        """

        live_states = {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.AWAITING_APPROVAL,
            SessionStatus.AWAITING_PLAN_DECISION,
            SessionStatus.AWAITING_USER,
        }
        for meta in self.state.sessions.list_sessions():
            if meta.status not in live_states:
                continue
            history = self.state.sessions.read_messages(meta.session_id)
            replayable = bool(history) and any(m.role == "user" for m in history)
            new_status = (
                SessionStatus.RESUMABLE if replayable else SessionStatus.INTERRUPTED
            )
            summary = meta.summary or (
                "resumable: server restart"
                if replayable
                else "interrupted: server restart"
            )
            self.state.sessions.write_metadata(
                SessionMetadata(
                    session_id=meta.session_id,
                    goal=meta.goal,
                    status=new_status,
                    created_at=meta.created_at,
                    updated_at=utc_now(),
                    summary=summary,
                )
            )

    @staticmethod
    def _new_session_id() -> str:
        return secrets.token_hex(6)
