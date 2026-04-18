"""Experiment routes for MolExp API."""

from __future__ import annotations

from shutil import rmtree

from fastapi import APIRouter, Depends

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError, ProjectNotFoundError
from ..schemas import ExperimentCreateRequest, ExperimentResponse, MessageResponse

router = APIRouter(
    prefix="/projects/{project_id}/experiments", tags=["experiments"]
)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    project_id: str,
    workspace=Depends(get_workspace),
) -> list[ExperimentResponse]:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    return [ExperimentResponse.from_model(e) for e in project.list_experiments()]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)
    return ExperimentResponse.from_model(experiment, runs=experiment.list_runs())


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    project_id: str,
    req: ExperimentCreateRequest,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    exp = project.experiment(
        name=req.name,
        workflow_source=req.workflow_source,
        params=req.parameter_space,
    )
    return ExperimentResponse.from_model(exp)


@router.delete("/{experiment_id}", response_model=MessageResponse)
def delete_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)
    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)
    rmtree(experiment.experiment_dir)
    return MessageResponse(message="Experiment deleted")
