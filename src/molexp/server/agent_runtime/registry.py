"""``AgentSessionRegistry`` — per-workspace live agent sessions (spec 00a).

A plain runtime container mapping ``workspace_root -> session_id ->``
:class:`AgentSessionRuntime`. Seated on ``app.state.agent_runtime`` in the
app lifespan (NOT a module-global cache, so app instances and tests stay
isolated and shutdown can cancel every in-flight turn).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from molexp.server.agent_runtime.runtime import AgentSessionRuntime

if TYPE_CHECKING:
    from molexp.agent.runner import AgentRunner
    from molexp.agent.session import Session


class AgentSessionRegistry:
    """In-process registry of live :class:`AgentSessionRuntime`s per workspace."""

    def __init__(self) -> None:
        self._by_workspace: dict[str, dict[str, AgentSessionRuntime]] = {}

    def create(
        self,
        *,
        workspace_root: str,
        runner: AgentRunner,
        session: Session,
        goal: str,
        user_input: str,
    ) -> AgentSessionRuntime:
        """Build a runtime, kick its first background turn, and register it.

        Args:
            workspace_root: Resolved workspace root str (the outer key).
            runner: The configured :class:`AgentRunner` for this session.
            session: The persistent :class:`Session` (its id is the inner key).
            goal: The session's goal description (for wire translation).
            user_input: The first turn's user prompt.

        Returns:
            The registered, already-running :class:`AgentSessionRuntime`.
        """
        runtime = AgentSessionRuntime(
            runner=runner,
            session=session,
            goal=goal,
            created_at=datetime.now(UTC).isoformat(),
        )
        runtime.start_turn(user_input)
        self._by_workspace.setdefault(workspace_root, {})[runtime.session_id] = runtime
        return runtime

    def get(self, workspace_root: str, session_id: str) -> AgentSessionRuntime | None:
        """Return the live runtime for ``(workspace_root, session_id)`` or ``None``."""
        return self._by_workspace.get(workspace_root, {}).get(session_id)

    def list_runtimes(self, workspace_root: str) -> list[AgentSessionRuntime]:
        """Return every live runtime under ``workspace_root`` (empty if none)."""
        return list(self._by_workspace.get(workspace_root, {}).values())

    def cancel_all(self) -> None:
        """Request cancellation of every in-flight turn across all workspaces."""
        for sessions in self._by_workspace.values():
            for runtime in sessions.values():
                runtime.cancel()

    async def aclose(self) -> None:
        """Cancel and await every in-flight turn — the lifespan shutdown hook."""
        self.cancel_all()
        for sessions in self._by_workspace.values():
            for runtime in sessions.values():
                await runtime.await_finished()
