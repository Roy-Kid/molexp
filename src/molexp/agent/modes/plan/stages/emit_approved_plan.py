"""Stage 7 — the ``approve_direction`` gate + handoff emission."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, RepairProposedEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes._planning import ApprovalGate, PlanState
from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.state import RepairSignal
from molexp.agent.modes.plan.tasks_planning import build_repair_diff
from molexp.agent.types import utc_now

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["EmitApprovedPlan"]


class _DirectionView:
    """Minimal approval view satisfying ``AgentHarness.approve``'s contract."""

    def __init__(self, *, summary: str) -> None:
        self.summary = summary


class EmitApprovedPlan(Stage[PlanThreadState, PlanThreadState]):
    """Open the ``approve_direction`` gate; on approve build the handoff.

    On approve: transition ``plan_folder`` to ``approved``, write the
    approved plan to disk, build the :class:`ApprovedPlanHandoff`, and
    store it on ``plan_mode._handoff`` so the mode's terminal
    :class:`ModeCompletedEvent` carries it.

    On reject: plant a :class:`RepairSignal` and yield
    :class:`RepairProposedEvent` — a registered repair policy decides
    whether to rewind to ``SynthesizeCandidates`` or route to the
    ``rejected`` terminal. The stage does **not** transition
    ``plan_folder`` to ``draft_plan`` itself; on rewind the lifecycle
    validator transitions ``awaiting_approval → draft_plan``, and on
    exhaustion PlanMode finalizes the ``awaiting_approval → rejected``
    transition post-pipeline.
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
        assert state.selected is not None
        assert state.intent is not None
        assert state.capabilities is not None
        view = _DirectionView(
            summary=(
                f"Plan {state.selected.plan_id}: "
                f"{len(state.selected.steps)} step(s) for "
                f"{state.intent.objective!r}"
            )
        )
        decision = await harness.approve(ApprovalGate.approve_direction, view)
        if decision.approved:
            self._plan_mode._transition(PlanState.approved)
            approved_plan = state.selected.model_copy(update={"state": PlanState.approved})
            self._plan_mode.plan_folder.write_selected_plan(approved_plan)
            handoff = ApprovedPlanHandoff(
                plan_id=self._plan_mode.plan_folder.plan_id,
                intent=state.intent,
                plan_graph=approved_plan,
                capability_graph=state.capabilities,
                plan_folder_path=Path(str(self._plan_mode.plan_folder.path())),
                direction_approved_at=utc_now(),
            )
            self._plan_mode._handoff = handoff
            yield state.model_copy(update={"selected": approved_plan})
            return
        diff = build_repair_diff(
            failed_invariant="direction_rejected",
            plan_graph=state.selected,
            rationale="the reviewer rejected the direction; re-synthesize candidates",
        )
        self._plan_mode._runtime.plant(RepairSignal(plan_diff=diff))
        yield RepairProposedEvent(
            failed_invariant=diff.failed_invariant,
            rationale=diff.rationale,
        )
        yield state
