"""Run routes for MolExp API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from molexp.models import RunStatus

from ..dependencies import get_workspace
from ..exceptions import RunNotFoundError, InvalidStatusError
from ..schemas import (
    RunCreateRequest,
    RunResponse,
    RunStatusResponse,
    MessageResponse,
)

router = APIRouter(
    prefix="/api/projects/{project_id}/experiments/{experiment_id}/runs",
    tags=["runs"],
)


@router.get("", response_model=list[RunResponse])
def list_runs(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> list[RunResponse]:
    """List runs in an experiment."""
    runs = workspace.list_runs(project_id, experiment_id)
    return [RunResponse.from_model(r) for r in runs]


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Get run details."""
    run = workspace.get_run(project_id, experiment_id, run_id)
    if not run:
        raise RunNotFoundError(run_id, experiment_id, project_id)
    
    asset_refs = workspace.get_asset_refs(project_id, experiment_id, run_id)
    context = workspace.get_run_context(project_id, experiment_id, run_id)
    
    return RunResponse.from_model(run, asset_refs=asset_refs, context=context)


@router.post("", response_model=RunResponse, status_code=201)
def create_run(
    project_id: str,
    experiment_id: str,
    run: RunCreateRequest,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Create a new run."""
    new_run = workspace.create_run(
        project_id=project_id,
        experiment_id=experiment_id,
        parameters=run.parameters,
        workflow_file=run.workflow_file,
        git_commit=run.git_commit,
    )
    return RunResponse.from_model(new_run)


@router.patch("/{run_id}/status", response_model=RunStatusResponse)
def update_run_status(
    project_id: str,
    experiment_id: str,
    run_id: str,
    status: dict[str, str],
    workspace=Depends(get_workspace),
) -> RunStatusResponse:
    """Update run status."""
    run = workspace.get_run(project_id, experiment_id, run_id)
    if not run:
        raise RunNotFoundError(run_id, experiment_id, project_id)
    
    new_status_str = status.get("status", run.status.value)
    try:
        new_status = RunStatus(new_status_str)
    except ValueError:
        raise InvalidStatusError(run.status.value, new_status_str)
    
    run.status = new_status
    if new_status_str in ["succeeded", "failed", "cancelled"]:
        run.finished_at = datetime.now()
    
    workspace.update_run(run)
    
    return RunStatusResponse(
        id=run.run_id,
        status=run.status.value,
        finished=run.finished_at.isoformat() if run.finished_at else None,
    )


@router.post("/{run_id}/start", response_model=RunStatusResponse)
def start_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunStatusResponse:
    """Start run execution."""
    run = workspace.get_run(project_id, experiment_id, run_id)
    if not run:
        raise RunNotFoundError(run_id, experiment_id, project_id)
    
    # Check if run is already finished or running
    if run.status in [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.RUNNING]:
        raise InvalidStatusError("pending", run.status.value)
    
    # Update status to running
    run.status = RunStatus.RUNNING
    workspace.update_run(run)
    
    # Execute workflow if we have a serialized graph
    if run.workflow_snapshot.serialized_graph:
        import json
        from molexp.ir.loader import load_workflow_from_json
        from molexp.ir.engine import WorkflowEngine
        from molexp.ir.compiler import compile_workflow, plan_execution
        
        workflow_ir = load_workflow_from_json(run.workflow_snapshot.serialized_graph)
        compile_workflow(workflow_ir)
        execution_plan = plan_execution(workflow_ir)
        
        engine = WorkflowEngine()
        status_map = engine.execute(workflow_ir, run_id=run.run_id, node_ids=execution_plan)
        
        all_succeeded = all(status_map.get(nid) == "SUCCEEDED" for nid in execution_plan)
        run.status = RunStatus.SUCCEEDED if all_succeeded else RunStatus.FAILED
    else:
        # Simulation for simple cases
        import time
        time.sleep(1)
        run.status = RunStatus.SUCCEEDED
    
    run.finished_at = datetime.now()
    workspace.update_run(run)
    
    return RunStatusResponse(
        id=run.run_id,
        status=run.status.value,
        finished=run.finished_at.isoformat() if run.finished_at else None,
    )
