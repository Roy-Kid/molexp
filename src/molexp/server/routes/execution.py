"""Execution routes for MolExp API."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from molexp.workflow import Workflow
from molexp.workflow.compiler import WorkflowCompiler

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError, ProjectNotFoundError
from ..schemas import (ExecutionCreateRequest, ExecutionPlanRequest,
                       ExecutionPlanResponse, RunResponse)

router = APIRouter(prefix="", tags=["execution"])


@router.post("/executions", response_model=RunResponse)
def create_execution(
    request: ExecutionCreateRequest,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Create a new execution in a specific project/experiment."""
    project = workspace.get_project(request.project_id)
    if not project:
        raise ProjectNotFoundError(request.project_id)

    experiment = project.get_experiment(request.experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(request.project_id, request.experiment_id)

    # Create run
    new_run = experiment.create_run(parameters=request.parameters)

    # TODO: Persist workflow_json if supported by Run model
    # workflow_snapshot = request.workflow_json

    return RunResponse.from_model(new_run)


@router.post("/plan", response_model=ExecutionPlanResponse)
def get_execution_plan(request: ExecutionPlanRequest) -> ExecutionPlanResponse:
    """Get execution plan for a workflow definition."""
    workflow = Workflow.model_validate_json(request.workflow_json)
    compiler = WorkflowCompiler()
    compiled_workflow = compiler.compile(workflow)
    plan = compiled_workflow.get_execution_plan(targets=request.targets)

    return ExecutionPlanResponse(
        plan=plan,
        nodeCount=len(plan),
    )
