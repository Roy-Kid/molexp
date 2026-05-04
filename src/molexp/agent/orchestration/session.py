"""AgentSession — per-session handle (spec §6.3).

Per Decision O1 the session is a thin facade: it streams events,
forwards inbound messages to the runner, and routes plan/approval
decisions. The live registry sits on :class:`AgentService`.

Phase 1a ships the surface only; Phase 1b/1c attach the runner and
the persistent stores.
"""

from __future__ import annotations

from typing import AsyncIterator

from molexp.agent.orchestration.approvals import ApprovalRegistry
from molexp.agent.orchestration.events import EventBus, SessionEvent
from molexp.agent.orchestration.plan import PlanStateMachine
from molexp.agent.tools.policy import ApprovalDecision
from molexp.agent.types import Goal, SessionStatus


class AgentSession:
    """Per-session handle returned by :class:`AgentService`."""

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
        self._bus = bus or EventBus()

    @property
    def bus(self) -> EventBus:
        return self._bus

    def stream_events(self) -> AsyncIterator[SessionEvent]:
        return self._bus.subscribe()

    async def respond_approval(self, decision: ApprovalDecision) -> bool:
        return self.approvals.resolve(decision)

    async def respond_plan(
        self,
        request_id: str,
        approved: bool,
        edited_plan: str | None = None,
        edited_workflow_ir: dict | None = None,
        feedback: str = "",
    ) -> None:
        if not self.plan.is_parked() or self.plan.last_request_id != request_id:
            return
        self.plan = (
            self.plan.approve(edited_plan, edited_workflow_ir)
            if approved
            else self.plan.reject(feedback)
        )

    async def send_user_message(self, content: str, request_id: str | None = None) -> None:
        # Phase 1b wires this into the runner. Phase 1a is no-op.
        return None

    async def cancel(self) -> None:
        self.status = SessionStatus.CANCELLED
        await self._bus.close()
