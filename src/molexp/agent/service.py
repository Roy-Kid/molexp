"""AgentService — public entry point (spec §6.3, §8).

Per Decision O1 the service is the workspace-scoped facade: it owns
the live session registry, the tool registry, and the asyncio task
that drives each session. Server routes import only this class.

Phase 0/1a ships the surface; Phase 1b lights up the runner.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import AsyncIterator

from molexp.agent.model import ModelClient
from molexp.agent.orchestration.events import (
    SessionEvent,
    SessionStarted,
)
from molexp.agent.orchestration.session import AgentSession
from molexp.agent.state.config import AgentSettings
from molexp.agent.state.memory import NoopMemoryStore
from molexp.agent.state.sessions import SessionMetadata, SessionStore
from molexp.agent.state.skills import SkillStore
from molexp.agent.state.store import AgentStateStore
from molexp.agent.tools.registry import ToolRegistry
from molexp.agent.types import Goal, SessionStatus, utc_now


class AgentService:
    """Workspace-scoped facade over the agent harness.

    Per Decision T1, every service instance owns its own
    :class:`ToolRegistry`. Native tools are registered at construction
    time by the workspace bootstrap code (Phase 2+).

    Phase 1a only exposes ``start_session`` / ``list_sessions`` /
    ``get_session`` / ``stream_events`` with stubbed behavior so the
    server route layer can compile against the final shape.
    """

    AGENT_DIRNAME = ".molexp-agent"

    def __init__(
        self,
        workspace_path: Path,
        settings: AgentSettings | None = None,
        model: ModelClient | None = None,
        registry: ToolRegistry | None = None,
        state: AgentStateStore | None = None,
    ) -> None:
        self.workspace_path = Path(workspace_path)
        self.settings = settings or AgentSettings()
        self.model = model
        self.registry = registry or ToolRegistry()
        self.state = state or self._default_state()
        self._sessions: dict[str, AgentSession] = {}

    # Public API ----------------------------------------------------------

    @classmethod
    def from_workspace(
        cls,
        workspace_path: str | Path,
        settings: AgentSettings | None = None,
        model: ModelClient | None = None,
    ) -> "AgentService":
        """Construct a service rooted at ``workspace_path``."""

        return cls(
            workspace_path=Path(workspace_path),
            settings=settings,
            model=model,
        )

    def start_session(self, goal: Goal) -> AgentSession:
        """Register a new session and return its handle.

        Phase 1a: emits :class:`SessionStarted`, persists metadata,
        and returns the handle. The runner is wired in Phase 1b.
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
        # Fire-and-forget the synchronous SessionStarted event so any
        # subscriber attached before the first turn observes it.
        return session

    async def emit_session_started(self, session: AgentSession) -> None:
        """Publish the initial :class:`SessionStarted` event.

        Phase 1a helper — Phase 1b folds this into ``run_turn``.
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
        """Mark every live session interrupted (Decision O3, Phases 1-4)."""

        for session in list(self._sessions.values()):
            session.status = SessionStatus.INTERRUPTED
            await session.bus.close()
        self._sessions.clear()

    # Internals -----------------------------------------------------------

    def _agent_root(self) -> Path:
        return self.workspace_path / self.AGENT_DIRNAME

    def _default_state(self) -> AgentStateStore:
        sessions_root = self._agent_root() / "sessions"
        return AgentStateStore(
            sessions=SessionStore(sessions_root),
            skills=SkillStore(),
            memory=NoopMemoryStore(),
        )

    @staticmethod
    def _new_session_id() -> str:
        return secrets.token_hex(6)
