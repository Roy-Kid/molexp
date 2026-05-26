"""``ApprovedPlanHandoff`` — PlanMode's sole terminal output object.

PlanMode is a read-only typed planner: it ends at the
``approve_direction`` gate and emits exactly one frozen
:class:`ApprovedPlanHandoff` carrying the approved typed
:class:`~molexp.agent.modes._planning.PlanGraph`. AuthorMode imports this
contract and lowers the ``PlanGraph`` into a ``WorkflowContract``.

Pure frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import IntentSpec, PlanGraph

__all__ = ["ApprovedPlanHandoff"]


class ApprovedPlanHandoff(BaseModel):
    """The frozen handoff PlanMode emits once the direction is approved.

    Attributes:
        plan_id: Identifier of the approved plan.
        intent: The typed user-intent contract the plan satisfies.
        plan_graph: The approved typed plan DAG. Each
            :class:`~molexp.agent.modes._planning.PlanStep` carries
            ``api_refs`` + ``composition_notes`` inline — no separate
            capability-graph artefact.
        plan_folder_path: On-disk path of the plan's ``PlanFolder``.
        direction_approved_at: When the ``approve_direction`` gate
            resolved in the affirmative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    intent: IntentSpec
    plan_graph: PlanGraph
    plan_folder_path: Path
    direction_approved_at: datetime
