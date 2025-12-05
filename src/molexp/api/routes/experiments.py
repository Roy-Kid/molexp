"""Experiment routes for MolExp API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError
from ..schemas import (
    ExperimentCreateRequest,
    ExperimentResponse,
    MessageResponse,
)

router = APIRouter(prefix="/api/projects/{project_id}/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    project_id: str,
    workspace=Depends(get_workspace),
) -> list[ExperimentResponse]:
    """List experiments in a project."""
    experiments = workspace.list_experiments(project_id)
    return [ExperimentResponse.from_model(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    """Get experiment details."""
    experiment = workspace.get_experiment(project_id, experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(experiment_id, project_id)
    
    runs = workspace.list_runs(project_id, experiment_id)
    return ExperimentResponse.from_model(experiment, runs=runs)


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    project_id: str,
    experiment: ExperimentCreateRequest,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    """Create a new experiment."""
    new_exp = workspace.create_experiment(
        project_id=project_id,
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        workflow_source=experiment.workflow_source,
        description=experiment.description,
        parameter_space=experiment.parameter_space,
    )
    return ExperimentResponse.from_model(new_exp)


@router.delete("/{experiment_id}", response_model=MessageResponse)
def delete_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Delete an experiment."""
    workspace.delete_experiment(project_id, experiment_id)
    return MessageResponse(message="Experiment deleted")
