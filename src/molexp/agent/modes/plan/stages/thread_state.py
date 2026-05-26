"""The typed carrier threaded across PlanMode's five stages.

:class:`PlanThreadState` accumulates each stage's structured output
(``IntentSpec`` → ``PlanGraph`` → ``PlanGraphPreflightReport``) so the
next stage reads its predecessor's result from a typed field instead of
a mutated shared dict. The substrate's executor threads this object as
each stage's ``input``; on a repair rewind it preserves the latest
snapshot so the rewound stage sees the accumulated state (including the
prior plan + preflight report, which the rewind prompt builder consumes
to tell the LLM what failed).

Frozen pydantic — pure data, no callables. Each stage returns a new
instance via :meth:`PlanThreadState.model_copy` rather than mutating.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import IntentSpec, PlanGraph
from molexp.agent.modes.plan.plan_graph_preflight import PlanGraphPreflightReport

__all__ = ["PlanThreadState"]


class PlanThreadState(BaseModel):
    """Typed carrier accumulating PlanMode's five-stage artefacts.

    Attributes:
        user_input: The end-user prompt passed into the run.
        intent: The typed :class:`IntentSpec` produced by
            ``SynthesizeIntent``; ``None`` before that stage runs.
        plan_graph: The typed :class:`PlanGraph` produced by
            ``ResearchAndPlan`` (with ``api_refs`` + ``composition_notes``
            inline on each step); ``None`` before that stage runs.
        preflight: The :class:`PlanGraphPreflightReport` produced by
            ``PreflightPlanGraph``; ``None`` before that stage runs.
            On a repair rewind from ``preflight_failed``, this carries
            the *prior* report so ``ResearchAndPlan`` can prepend it to
            its prompt as failure context.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False, extra="forbid")

    user_input: str
    intent: IntentSpec | None = None
    plan_graph: PlanGraph | None = None
    preflight: PlanGraphPreflightReport | None = None
