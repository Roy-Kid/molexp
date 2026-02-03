"""Experiment routes for MolExp API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError
from ..schemas import (ExperimentCreateRequest, ExperimentResponse,
                       MessageResponse)

router = APIRouter(
    prefix="/projects/{project_id}/experiments", tags=["experiments"]
)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    project_id: str,
    workspace=Depends(get_workspace),
) -> list[ExperimentResponse]:
    """List experiments in a project."""
    project = workspace.get_project(project_id)
    if not project:
        raise ExperimentNotFoundError(project_id, "")
        
    experiments = project.list_experiments()
    return [ExperimentResponse.from_model(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    """Get experiment details."""
    project = workspace.get_project(project_id)
    if not project:
        raise ExperimentNotFoundError(project_id, experiment_id)

    experiment = project.get_experiment(experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(project_id, experiment_id)

    runs = experiment.list_runs()
    return ExperimentResponse.from_model(experiment, runs=runs)


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    project_id: str,
    experiment: ExperimentCreateRequest,
    workspace=Depends(get_workspace),
) -> ExperimentResponse:
    """Create a new experiment."""
    project = workspace.get_project(project_id)
    if not project:
        raise ExperimentNotFoundError(project_id, "")

    # Project.create_experiment only takes name
    new_exp = project.create_experiment(name=experiment.name)
    
    # Metadata updates for other fields could happen here if needed
    # new_exp.metadata.description = experiment.description
    # new_exp.save()
    
    return ExperimentResponse.from_model(new_exp)


@router.delete("/{experiment_id}", response_model=MessageResponse)
def delete_experiment(
    project_id: str,
    experiment_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Delete an experiment."""
    from shutil import rmtree
    
    project = workspace.get_project(project_id)
    if not project:
        raise ExperimentNotFoundError(project_id, experiment_id)
        
    experiment_dir = project.workspace.root / "projects" / project.id / "experiments" / experiment_id
    if not experiment_dir.exists():
         raise ExperimentNotFoundError(project_id, experiment_id)
         
    rmtree(experiment_dir)
    return MessageResponse(message="Experiment deleted")
