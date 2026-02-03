"""Run routes for MolExp API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from molexp.workspace import RunStatus

from ..dependencies import get_workspace
from ..exceptions import InvalidStatusError, RunNotFoundError
from ..schemas import (RunCreateRequest, RunResponse,
                       RunStatusResponse)

router = APIRouter(
    prefix="/projects/{project_id}/experiments/{experiment_id}/runs",
    tags=["runs"],
)


def _get_experiment(workspace, project_id: str, experiment_id: str):
    project = workspace.get_project(project_id)
    if not project:
        return None
    return project.get_experiment(experiment_id)


@router.get("", response_model=list[RunResponse])
def list_runs(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> list[RunResponse]:
    """List runs in an experiment."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, "")
        
    runs = experiment.list_runs()
    return [RunResponse.from_model(r) for r in runs]


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Get run details."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
        
    run = experiment.get_run(run_id)
    if not run:
         raise RunNotFoundError(project_id, experiment_id, run_id)

    return RunResponse.from_model(run)


@router.post("", response_model=RunResponse, status_code=201)
def create_run(
    project_id: str,
    experiment_id: str,
    run_req: RunCreateRequest,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Create a new run."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, "")

    new_run = experiment.create_run(parameters=run_req.parameters)
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
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
        
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    new_status_str = status.get("status", run.status)
    try:
        new_status = RunStatus(new_status_str)
    except ValueError:
        raise InvalidStatusError(run.status, new_status_str)

    run.status = new_status
    if new_status_str in ["succeeded", "failed", "cancelled"]:
        run.finished_at = datetime.now()

    run.save()

    return RunStatusResponse(
        id=run.id,
        status=run.status,
        finished=run.metadata.updated_at.isoformat(),
    )


@router.post("/{run_id}/start", response_model=RunStatusResponse)
def start_run(
    project_id: str,
    experiment_id: str,
    run_id: str,
    workspace=Depends(get_workspace),
) -> RunStatusResponse:
    """Start run execution."""
    experiment = _get_experiment(workspace, project_id, experiment_id)
    if not experiment:
        raise RunNotFoundError(project_id, experiment_id, run_id)
        
    run = experiment.get_run(run_id)
    if not run:
        raise RunNotFoundError(project_id, experiment_id, run_id)

    # Minimal start logic for now
    with run.context() as ctx:
        # Mock execution or async dispatch
        pass
        
    return RunStatusResponse(
        id=run.id,
        status=run.status,
        finished=None,
    )
