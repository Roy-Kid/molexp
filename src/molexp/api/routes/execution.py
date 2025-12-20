"""Execution routes for MolExp API."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from molexp.ir.compiler import plan_execution
from molexp.ir.loader import load_workflow_from_json

from ..dependencies import get_workspace
from ..schemas import ExecutionPlanRequest, ExecutionPlanResponse, RunResponse

router = APIRouter(prefix="/api", tags=["execution"])


@router.post("/executions", response_model=RunResponse)
def create_generic_execution(
    run_data: dict,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Create a new execution in the default playground project."""
    project_id = "playground"
    experiment_id = "default"

    if not workspace.get_project(project_id):
        workspace.create_project(
            project_id, "Playground", "Default project for ad-hoc runs"
        )

    if not workspace.get_experiment(project_id, experiment_id):
        workspace.create_experiment(
            project_id, experiment_id, "Default Experiment", "adhoc"
        )

    workflow_snapshot = run_data.get("workflowSnapshot")
    workflow_file = "workflow.json"

    new_run = workspace.create_run(
        project_id=project_id,
        experiment_id=experiment_id,
        parameters={},
        workflow_file=workflow_file,
    )

    if workflow_snapshot:
        new_run.workflow_snapshot.serialized_graph = json.dumps(workflow_snapshot)
        workspace.update_run(new_run)

    return RunResponse.from_model(new_run)


@router.post("/plan", response_model=ExecutionPlanResponse)
def get_execution_plan(request: ExecutionPlanRequest) -> ExecutionPlanResponse:
    """Get execution plan for a workflow definition."""
    workflow_ir = load_workflow_from_json(request.workflow_json)
    plan = plan_execution(workflow_ir, targets=request.targets)

    return ExecutionPlanResponse(
        plan=plan,
        nodeCount=len(plan),
    )
