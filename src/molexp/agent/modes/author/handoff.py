"""``MaterializedWorkspaceHandoff`` — AuthorMode's sole terminal output.

AuthorMode consumes the :class:`~molexp.agent.modes.plan.handoff.ApprovedPlanHandoff`
PlanMode emits and materializes a validated experiment workspace. Its
terminal product is one frozen :class:`MaterializedWorkspaceHandoff`
carrying the ``ready_for_run`` :class:`~molexp.agent.modes._planning.PlanGraph`,
the on-disk layout, the entrypoint, and a snapshot of the workflow-layer
:class:`~molexp.workflow.ValidationReport`. RunMode (sub-spec 05) imports
this contract.

Pure frozen-pydantic data; no LLM, no I/O.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import PlanGraph
from molexp.workflow import ValidationReport

__all__ = ["MaterializedWorkspaceHandoff"]


class MaterializedWorkspaceHandoff(BaseModel):
    """The frozen handoff AuthorMode emits once materialization succeeds.

    This is the published seam RunMode consumes — exactly nine fields,
    no more.

    Attributes:
        plan_id: Identifier of the materialized plan.
        plan_graph: The plan DAG, in state
            :data:`~molexp.agent.modes._planning.PlanState.ready_for_run`.
        experiment_workspace_path: On-disk root of the materialized
            experiment workspace (the bound ``PlanFolder`` directory).
        workflow_yaml_path: Path of the generated workflow IR
            (``ir/workflow.yaml``).
        entrypoint_module: Dotted module path of the generated workflow
            entrypoint (e.g. ``experiment.workflow``).
        entrypoint_symbol: Name of the entrypoint callable in
            ``entrypoint_module`` (e.g. ``create_workflow``).
        source_root: On-disk root of the generated ``src/`` tree.
        validation_report_snapshot: The
            :class:`~molexp.workflow.ValidationReport` the final
            ``ValidateWorkspace`` stage produced.
        materialization_approved_at: When the ``approve_materialization``
            gate resolved in the affirmative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    plan_graph: PlanGraph
    experiment_workspace_path: Path
    workflow_yaml_path: Path
    entrypoint_module: str
    entrypoint_symbol: str
    source_root: Path
    validation_report_snapshot: ValidationReport
    materialization_approved_at: datetime
