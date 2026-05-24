"""The typed carrier threaded across PlanMode's seven stages.

:class:`PlanThreadState` accumulates each stage's structured output
(IntentSpec → CapabilityGraph → CandidateSet → selected PlanGraph →
PlanGraphPreflightReport) so the next stage reads the prior stage's
result from a typed field instead of a mutated shared dict. The
substrate's executor threads this object as each stage's ``input``;
on a repair-rewind it preserves the latest snapshot so the rewound
stage sees the accumulated state, not the original user input.

Frozen pydantic — pure data, no callables. Each stage returns a new
instance via :meth:`PlanThreadState.model_copy` rather than mutating.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import CapabilityGraph, IntentSpec, PlanGraph
from molexp.agent.modes.plan.plan_graph_preflight import PlanGraphPreflightReport
from molexp.agent.modes.plan.tasks_planning import CandidateSet

__all__ = ["PlanThreadState"]


class PlanThreadState(BaseModel):
    """Typed carrier accumulating PlanMode's seven-stage artefacts.

    Attributes:
        user_input: The end-user prompt passed into the run.
        intent: The typed :class:`IntentSpec` produced by
            ``SynthesizeIntent``; ``None`` before that stage runs.
        capabilities: The typed :class:`CapabilityGraph` produced by
            ``ExploreCapabilities``; ``None`` before that stage runs.
        candidates: The :class:`CandidateSet` produced by
            ``SynthesizeCandidates``.
        selected: The selected :class:`PlanGraph` (post-apply-pending-repair)
            produced by ``SelectPlan``.
        preflight: The :class:`PlanGraphPreflightReport` produced by
            ``PreflightPlanGraph``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")

    user_input: str
    intent: IntentSpec | None = None
    capabilities: CapabilityGraph | None = None
    candidates: CandidateSet | None = None
    selected: PlanGraph | None = None
    preflight: PlanGraphPreflightReport | None = None
