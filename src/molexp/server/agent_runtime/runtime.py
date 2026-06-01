"""``AgentSessionRuntime`` — one live server-side agent session (spec 00a).

A plain runtime container owning exactly one :class:`AgentRunner`, one
:class:`Session`, and the current :class:`AgentTurn`. It is the unit the
:class:`~molexp.server.agent_runtime.registry.AgentSessionRegistry` stores
and that the route layer translates into wire ``*Response`` shapes — runtime
objects themselves never cross into ``server.schemas``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from molexp.server.agent_runtime.turn import AgentTurn

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent
    from molexp.agent.runner import AgentRunner
    from molexp.agent.session import Session


class AgentSessionRuntime:
    """One server-side agent session: its runner, session, and current turn."""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        session: Session,
        goal: str,
        created_at: str,
    ) -> None:
        self.runner = runner
        self.session = session
        self.goal = goal
        self.created_at = created_at
        self._turn: AgentTurn | None = None

    @property
    def session_id(self) -> str:
        """The persistent session id (the registry's inner key)."""
        return self.session.session_id

    def start_turn(self, user_input: str) -> AgentTurn:
        """Spawn a new background turn on this session and make it current."""
        self._turn = AgentTurn.start(
            runner=self.runner, session=self.session, user_input=user_input
        )
        return self._turn

    def events(self) -> tuple[AgentEvent, ...]:
        """Snapshot of the current turn's collected events (empty if none)."""
        return tuple(self._turn.events) if self._turn is not None else ()

    def status(self) -> str:
        """The current turn's status, or ``"idle"`` before any turn started."""
        return self._turn.status if self._turn is not None else "idle"

    @property
    def error(self) -> BaseException | None:
        """The current turn's failure, if it ended in ``failed`` status."""
        return self._turn.error if self._turn is not None else None

    async def subscribe_events(self) -> AsyncIterator[AgentEvent]:
        """Live replay-then-tail of the current turn's events (empty if none)."""
        if self._turn is None:
            return
        async for event in self._turn.subscribe():
            yield event

    async def await_finished(self) -> None:
        """Await the current turn (no-op when no turn has started)."""
        if self._turn is not None:
            await self._turn.await_finished()

    def cancel(self) -> None:
        """Cancel the current turn (no-op when no turn has started)."""
        if self._turn is not None:
            self._turn.cancel()
