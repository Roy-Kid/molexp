"""Project routes for MolExp API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_workspace
from ..exceptions import ProjectNotFoundError
from ..schemas import MessageResponse, ProjectCreateRequest, ProjectResponse

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(workspace=Depends(get_workspace)) -> list[ProjectResponse]:
    """List all projects."""
    projects = workspace.list_projects()
    return [ProjectResponse.from_model(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, workspace=Depends(get_workspace)) -> ProjectResponse:
    """Get project details."""
    project = workspace.get_project(project_id)
    if not project:
        raise ProjectNotFoundError(project_id)

    experiments = workspace.list_experiments(project_id)
    return ProjectResponse.from_model(project, experiment_count=len(experiments))


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(
    project: ProjectCreateRequest,
    workspace=Depends(get_workspace),
) -> ProjectResponse:
    """Create a new project."""
    new_project = workspace.create_project(
        project_id=project.project_id,
        name=project.name,
        description=project.description,
        owner=project.owner,
        tags=project.tags,
    )
    return ProjectResponse.from_model(new_project)


@router.delete("/{project_id}", response_model=MessageResponse)
def delete_project(
    project_id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Delete a project."""
    workspace.delete_project(project_id)
    return MessageResponse(message="Project deleted")
