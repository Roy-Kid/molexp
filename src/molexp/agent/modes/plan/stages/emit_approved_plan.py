"""Stage 5 — the ``approve_direction`` gate + handoff emission."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.events import AgentEvent, RepairProposedEvent
from molexp.agent.modes._planning import ApprovalGate, PlanState
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.stage import Stage
from molexp.agent.types import utc_now

if TYPE_CHECKING:
    from molexp.agent.modes.plan._mode import PlanMode
    from molexp.agent.runtime import AgentHarness

__all__ = ["EmitApprovedPlan"]


class _DirectionView:
    """Minimal approval view satisfying ``AgentHarness.approve``'s contract."""

    def __init__(self, *, summary: str) -> None:
        self.summary = summary


class EmitApprovedPlan(Stage[PlanThreadState, PlanThreadState]):
    """Open the ``approve_direction`` gate; on approve emit the handoff.

    On approve: transition ``plan_folder`` to ``approved``, persist the
    approved plan, build the :class:`ApprovedPlanHandoff`, store it on
    ``plan_mode._handoff`` so the mode's terminal
    :class:`ModeCompletedEvent` carries it.

    On reject: yield :class:`RepairProposedEvent` — the registered
    repair policy decides whether to rewind to ``ResearchAndPlan`` or
    route to the ``rejected`` terminal. The threaded state's
    ``plan_graph`` + ``preflight`` survive the rewind so the rewound
    research stage can prepend them to its prompt as repair context.
    """

    name: ClassVar[str] = "EmitApprovedPlan"
    pre_state: ClassVar[str | None] = "awaiting_approval"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.plan_graph is not None
        assert state.intent is not None
        view = _DirectionView(
            summary=(
                f"Plan {state.plan_graph.plan_id}: "
                f"{len(state.plan_graph.steps)} step(s) for "
                f"{state.intent.objective!r}"
            )
        )
        decision = await harness.approve(ApprovalGate.approve_direction, view)
        if decision.approved:
            self._plan_mode._transition(PlanState.approved)
            approved_plan = state.plan_graph.model_copy(update={"state": PlanState.approved})
            self._plan_mode.plan_folder.write_plan_graph(approved_plan)
            handoff = ApprovedPlanHandoff(
                plan_id=self._plan_mode.plan_folder.plan_id,
                intent=state.intent,
                plan_graph=approved_plan,
                plan_folder_path=Path(str(self._plan_mode.plan_folder.path())),
                direction_approved_at=utc_now(),
            )
            self._plan_mode._handoff = handoff
            yield state.model_copy(update={"plan_graph": approved_plan})
            return
        yield RepairProposedEvent(
            failed_invariant="direction_rejected",
            rationale="the reviewer rejected the direction; regenerate the plan",
        )
        yield state
