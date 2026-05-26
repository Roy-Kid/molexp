"""Stage 4 — pure structural preflight; failure rewinds to ResearchAndPlan.

On failure: transitions ``plan_folder`` to ``preflight_failed``, persists
the report under the plan folder, and yields a
:class:`PreflightFailedEvent` — the registered :class:`RepairPolicy`
decides whether to rewind to ``ResearchAndPlan`` or route to the
``preflight_failed`` terminal. The threaded :class:`PlanThreadState`
keeps the failing plan + report so the rewound stage can prepend them
to its prompt as failure context.

On pass: emits the new state with the report attached for the next
stage (``EmitApprovedPlan``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, PreflightFailedEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes._planning import PlanState
from molexp.agent.modes.plan.plan_graph_preflight import preflight_plan_graph
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["PreflightPlanGraph"]


class PreflightPlanGraph(Stage[PlanThreadState, PlanThreadState]):
    """Run the pure structural preflight on the freshly-emitted plan."""

    name: ClassVar[str] = "PreflightPlanGraph"
    pre_state: ClassVar[str | None] = "draft_plan"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — kept for substrate's Stage.run contract
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.plan_graph is not None, "PreflightPlanGraph ran without a plan_graph"
        assert state.intent is not None
        report = preflight_plan_graph(
            plan_graph=state.plan_graph,
            required_outputs=state.intent.required_outputs,
        )
        self._plan_mode.plan_folder.write_preflight_report(report)
        new_state = state.model_copy(update={"preflight": report})

        if not report.passed:
            failed = report.failed_check_names()
            self._plan_mode._transition(PlanState.preflight_failed)
            yield PreflightFailedEvent(failed_checks=failed)
        yield new_state
