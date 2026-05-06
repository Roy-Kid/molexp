"""AgentSession — per-session handle.

A thin facade that streams events, forwards inbound messages to the
runner, and routes plan/approval decisions. The live registry sits
on :class:`AgentService`.

The session owns:

- ``bus`` — :class:`EventBus` fan-out for SSE / JSONL trace / tests.
- ``inbox`` — async queue of inbound user messages the runner pulls.
- ``approvals`` — :class:`ApprovalRegistry` for HITL tool gates.
- ``user_inbox`` — :class:`UserMessageRegistry` for ``ask_user`` tools.
- ``plan`` — :class:`PlanStateMachine` for plan-mode transitions.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from pydantic import BaseModel, ConfigDict

from molexp.agent.orchestration.approvals import ApprovalRegistry
from molexp.agent.orchestration.chat import UserMessageRegistry
from molexp.agent.orchestration.events import EventBus, SessionEvent
from molexp.agent.orchestration.plan import PlanStateMachine
from molexp.agent.tools.policy import ApprovalDecision
from molexp.agent.types import Goal, SessionStatus
from molexp.workflow import PlanProposal


class _InboundMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    request_id: str | None = None


class AgentSession:
    """Per-session handle returned by :class:`AgentService`.

    The session does not run anything on its own; the
    :class:`AgentService` owns the asyncio task that drives the runner
    against this session's inbox.
    """

    def __init__(
        self,
        session_id: str,
        goal: Goal,
        bus: EventBus | None = None,
    ) -> None:
        self.session_id = session_id
        self.goal = goal
        self.status: SessionStatus = SessionStatus.PENDING
        self.plan = PlanStateMachine()
        self.approvals = ApprovalRegistry()
        self.user_inbox = UserMessageRegistry()
        self._bus = bus or EventBus()
        self._inbox: "asyncio.Queue[_InboundMessage | None]" = asyncio.Queue()
        self._task: asyncio.Task | None = None
        # Set by respond_plan() to wake the runner parked in plan mode.
        self._plan_resolved: asyncio.Event = asyncio.Event()

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def attach_task(self, task: asyncio.Task) -> None:
        """Bind the background runner task that drives this session."""

        self._task = task

    def stream_events(self) -> AsyncIterator[SessionEvent]:
        return self._bus.subscribe()

    async def next_inbound(self) -> _InboundMessage | None:
        """Await the next inbound user message, or ``None`` on close."""

        return await self._inbox.get()

    async def send_user_message(self, content: str, request_id: str | None = None) -> None:
        """Deliver a user message to the runner.

        If a ``request_id`` is provided and matches an outstanding
        ``ask_user`` pending entry, the gate's future resolves and the
        runner picks up where it left off. Otherwise the message is
        queued onto the inbox for the next free turn (or, if the runner
        is parked on an unsolicited ``ask_user``, is delivered to the
        oldest pending request).
        """

        if request_id is not None and self.user_inbox.has(request_id):
            self.user_inbox.resolve(request_id, content)
            return
        # If a tool is parked on ask_user but the user replied without
        # echoing the request id, deliver to the oldest pending prompt
        # (mirrors the legacy plugin behavior; a malformed UI should
        # not strand the session).
        oldest = self.user_inbox.pop_oldest()
        if oldest is not None and not oldest.future.done():
            oldest.future.set_result(content)
            return
        await self._inbox.put(_InboundMessage(content=content, request_id=request_id))

    async def respond_approval(self, decision: ApprovalDecision) -> bool:
        return self.approvals.resolve(decision)

    async def respond_plan(
        self,
        request_id: str,
        approved: bool,
        edited_plan: str | None = None,
        edited_proposal: PlanProposal | None = None,
        feedback: str = "",
    ) -> bool:
        if not self.plan.is_parked() or self.plan.last_request_id != request_id:
            return False
        self.plan = (
            self.plan.approve(edited_plan, edited_proposal)
            if approved
            else self.plan.reject(feedback)
        )
        self._plan_resolved.set()
        return True

    async def wait_plan_decision(self) -> None:
        """Block until :meth:`respond_plan` resolves the parked plan."""

        await self._plan_resolved.wait()
        self._plan_resolved.clear()

    async def cancel(self) -> None:
        self.status = SessionStatus.CANCELLED
        self.user_inbox.cancel_all()
        await self._inbox.put(None)
        await self._bus.close()
        if self._task is not None and not self._task.done():
            self._task.cancel()


__all__ = ["AgentSession"]
