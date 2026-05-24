"""Stage 6 — pure structural preflight; failure plants a repair."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, PreflightFailedEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes._planning import PlanState
from molexp.agent.modes.plan.plan_graph_preflight import run_plan_graph_preflight
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.state import RepairSignal
from molexp.agent.modes.plan.tasks_planning import build_repair_diff

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["PreflightPlanGraph"]


class PreflightPlanGraph(Stage[PlanThreadState, PlanThreadState]):
    """Run the pure structural preflight on the selected plan.

    On failure: transitions ``plan_folder`` to ``preflight_failed``,
    plants a :class:`RepairSignal` for the next iteration's
    ``SelectPlan`` to consume, and yields
    :class:`PreflightFailedEvent` — a registered repair policy decides
    whether the executor rewinds to ``SynthesizeCandidates`` or routes
    to the ``preflight_failed`` terminal.

    On pass: just emits the new state for the approval stage.
    """

    name: ClassVar[str] = "PreflightPlanGraph"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — kept for substrate's Stage.run contract
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.selected is not None
        assert state.intent is not None
        assert state.capabilities is not None
        preflight = run_plan_graph_preflight(
            plan_graph=state.selected,
            intent=state.intent,
            capabilities=state.capabilities,
        )
        self._plan_mode.plan_folder.write_preflight_report(preflight)
        new_state = state.model_copy(update={"preflight": preflight})

        if not preflight.passed:
            failed = preflight.failed_check_names()
            self._plan_mode._transition(PlanState.preflight_failed)
            diff = build_repair_diff(
                failed_invariant=failed[0] if failed else "preflight",
                plan_graph=state.selected,
                rationale=("preflight failed: " + ", ".join(failed) + "; re-synthesize candidates"),
            )
            self._plan_mode._runtime.plant(RepairSignal(plan_diff=diff))
            yield PreflightFailedEvent(failed_checks=failed)
        yield new_state
