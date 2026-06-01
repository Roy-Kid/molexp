"""``AgentTurn`` — one background agent turn (spec 00a).

A plain runtime container (it owns a live ``asyncio.Task`` and a mutable
event list, so per CLAUDE.md "runtime container = plain class with explicit
``__init__``" — never pydantic). One turn drives a single
:meth:`~molexp.agent.runner.AgentRunner.run_events` stream to completion,
appending each :data:`~molexp.agent.events.AgentEvent` it yields.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from molexp.agent.events import AsyncIteratorEventSink

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent
    from molexp.agent.runner import AgentRunner
    from molexp.agent.session import Session

TurnStatus = Literal["running", "completed", "failed", "cancelled"]
"""Lifecycle state of an :class:`AgentTurn`."""


class AgentTurn:
    """A single background turn: the task, its collected events, its status."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.status: TurnStatus = "running"
        self.error: BaseException | None = None
        self._task: asyncio.Task[None] | None = None
        # Replay-then-tail fan-out: each live subscriber registers a sink the
        # drive loop pushes into; ``_lock`` makes (append + fan-out) atomic with
        # (snapshot + register) so no event is lost or duplicated at the seam.
        self._sinks: list[AsyncIteratorEventSink] = []
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def start(cls, *, runner: AgentRunner, session: Session, user_input: str) -> AgentTurn:
        """Spawn the background turn task and return the live :class:`AgentTurn`.

        Args:
            runner: The configured :class:`AgentRunner` to drive.
            session: The persistent :class:`Session` the turn runs against.
            user_input: The user prompt for this turn.

        Returns:
            A running :class:`AgentTurn` whose ``_task`` drains
            ``runner.run_events(session, user_input)`` into ``events``.
        """
        turn = cls()
        turn._task = asyncio.create_task(turn._drive(runner, session, user_input))
        return turn

    async def _drive(self, runner: AgentRunner, session: Session, user_input: str) -> None:
        """Drain the event stream into ``events``; record terminal status.

        On normal completion → ``completed``. On cancellation → ``cancelled``
        (re-raised, as asyncio requires). On any other exception → ``failed``
        with the exception stored on ``error`` and swallowed, so a shutdown
        ``await_finished`` never re-raises a turn's domain failure.
        """
        try:
            async for event in runner.run_events(session, user_input):
                async with self._lock:
                    self.events.append(event)
                    for sink in self._sinks:
                        await sink(event)
            self.status = "completed"
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:
            self.status = "failed"
            self.error = exc
        finally:
            async with self._lock:
                self._closed = True
                for sink in self._sinks:
                    await sink.close()

    async def subscribe(self) -> AsyncIterator[AgentEvent]:
        """Yield this turn's events replay-then-tail.

        Under ``_lock`` (so atomic with the drive loop's append + fan-out), the
        already-collected events are snapshotted and — when the turn is still
        live — a fresh sink is registered. The snapshot is yielded first
        (replay), then everything pushed after registration (tail), in emission
        order with no event lost or duplicated at the seam. A subscriber that
        connects after the turn has ended gets the full replay and stops.
        """
        async with self._lock:
            replay = list(self.events)
            sink: AsyncIteratorEventSink | None = None
            if not self._closed:
                sink = AsyncIteratorEventSink()
                self._sinks.append(sink)
        for event in replay:
            yield event
        if sink is not None:
            async for event in sink:
                yield event

    def cancel(self) -> None:
        """Request cancellation of the background turn task (idempotent)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def await_finished(self) -> None:
        """Await the turn task, suppressing the ``CancelledError`` from cancel."""
        if self._task is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
