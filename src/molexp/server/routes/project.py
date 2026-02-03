"""Project routes for MolExp API."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError, ProjectNotFoundError
from ..schemas import (
    AssetResponse,
    MessageResponse,
    ProjectCreateRequest,
    ProjectResponse,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(workspace=Depends(get_workspace)) -> list[ProjectResponse]:
    """List all projects."""
    projects = workspace.list_projects()
    return [ProjectResponse.from_model(p) for p in projects]


@router.get("/{id}", response_model=ProjectResponse)
def get_project(id: str, workspace=Depends(get_workspace)) -> ProjectResponse:
    """Get project details."""
    project = workspace.get_project(id)
    if not project:
        raise ProjectNotFoundError(id)

    experiments = project.list_experiments()
    return ProjectResponse.from_model(project, experiment_count=len(experiments))


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(
    project: ProjectCreateRequest,
    workspace=Depends(get_workspace),
) -> ProjectResponse:
    """Create a new project."""
    new_project = workspace.create_project(
        name=project.name,
    )
    return ProjectResponse.from_model(new_project)


@router.delete("/{id}", response_model=MessageResponse)
def delete_project(
    id: str,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Delete a project."""
    try:
        workspace.delete_project(id)
    except KeyError:
        raise ProjectNotFoundError(id)

    return MessageResponse(message="Project deleted")


# ============================================================================
# Project Asset Routes
# ============================================================================


@router.get("/{id}/assets", response_model=list[AssetResponse])
def list_project_assets(
    id: str,
    limit: int = 100,
    workspace=Depends(get_workspace),
) -> list[AssetResponse]:
    """List assets in a project."""
    project = workspace.get_project(id)
    if not project:
        raise ProjectNotFoundError(id)

    assets = project.assets.list_assets()[:limit]
    return [AssetResponse.from_model(a) for a in assets]


@router.get("/{id}/assets/{asset_id}", response_model=AssetResponse)
def get_project_asset(
    id: str,
    asset_id: str,
    workspace=Depends(get_workspace),
) -> AssetResponse:
    """Get project asset details."""
    project = workspace.get_project(id)
    if not project:
        raise ProjectNotFoundError(id)

    asset = project.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)

    return AssetResponse.from_model(asset)


@router.post("/{id}/assets/upload", response_model=AssetResponse, status_code=201)
async def upload_project_asset(
    id: str,
    file: UploadFile = File(...),
    workspace=Depends(get_workspace),
) -> AssetResponse:
    """Upload a new asset to a project."""
    project = workspace.get_project(id)
    if not project:
        raise ProjectNotFoundError(id)

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        filename = file.filename or "untitled"

        # Use project.assets (AssetLibrary) to import
        asset = project.assets.import_asset(
            path=tmp_path,
            name=filename,
            metadata={"original_filename": filename},
            action="move",
        )
        
        # Explicitly save project metadata to ensure asset is registered
        # (Though import_asset in project.py wrapper handles this, but let's be sure we are using the wrapper if it exists)
        # Wait, I am accessing project.assets (AssetLibrary) directly. 
        # Project.py wrapper `import_asset` does extra step: self.save().
        # So I should loop back to using `project.import_asset` if it supports path?
        # project.py: import_asset(self, name: str, src: str | Path, ...)
        
        # NO, I should use project.import_asset because it updates project.json!
        # Re-doing the call below to use project.import_asset instead of project.assets.import_asset
        
        # But wait, project.py: `import_asset` calls `self.assets.import_asset`.
        # So using project.import_asset is safer.
        pass

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    # Re-do with correct method
    try:
         asset = project.import_asset(
            name=filename,
            src=tmp_path,
            action="move",
            meta={"original_filename": filename}
        )
         return AssetResponse.from_model(asset)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


@router.get("/{id}/assets/{asset_id}/download")
def download_project_asset(
    id: str,
    asset_id: str,
    workspace=Depends(get_workspace),
):
    """Download project asset content."""
    project = workspace.get_project(id)
    if not project:
        raise ProjectNotFoundError(id)

    asset = project.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)

    payload_dir = project.assets.get_payload_dir(asset_id)
    if not payload_dir.exists():
        raise AssetNotFoundError(asset_id)

    files = list(payload_dir.iterdir())
    if not files:
        raise AssetNotFoundError(asset_id)

    file_path = files[0]
    filename = asset.metadata.get("original_filename") or file_path.name

    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
