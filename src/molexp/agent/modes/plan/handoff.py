"""``ApprovedPlanHandoff`` — PlanMode's sole terminal output object.

PlanMode is a read-only typed planner: it ends at the
``approve_direction`` gate and emits exactly one frozen
:class:`ApprovedPlanHandoff` carrying the approved typed
:class:`~molexp.agent.modes._planning.PlanGraph`. AuthorMode (sub-spec
04) imports this contract and lowers the ``PlanGraph`` into a
``WorkflowContract``.

Pure frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import CapabilityGraph, IntentSpec, PlanGraph

__all__ = ["ApprovedPlanHandoff"]


class ApprovedPlanHandoff(BaseModel):
    """The frozen handoff PlanMode emits once the direction is approved.

    This is the published seam AuthorMode consumes — exactly six fields,
    no more.

    Attributes:
        plan_id: Identifier of the approved plan.
        intent: The typed user-intent contract the plan satisfies.
        plan_graph: The approved typed plan DAG.
        capability_graph: The typed capability graph the plan binds to.
        plan_folder_path: On-disk path of the plan's ``PlanFolder``.
        direction_approved_at: When the ``approve_direction`` gate
            resolved in the affirmative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    intent: IntentSpec
    plan_graph: PlanGraph
    capability_graph: CapabilityGraph
    plan_folder_path: Path
    direction_approved_at: datetime
