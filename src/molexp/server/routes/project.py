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
def list_projects(workspace=Depends(get_workspace)) -> list[ProjectResponse]:  # noqa: ANN001
    return [ProjectResponse.from_model(p) for p in workspace.list_projects()]


@router.get("/{id}", response_model=ProjectResponse)
def get_project(id: str, workspace=Depends(get_workspace)) -> ProjectResponse:  # noqa: ANN001
    project = workspace.get_project(id)
    return ProjectResponse.from_model(project, experiment_count=len(project.list_experiments()))


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(
    project: ProjectCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> ProjectResponse:
    new_project = workspace.add_project(project.name)
    return ProjectResponse.from_model(new_project)


@router.delete("/{id}", response_model=MessageResponse)
def delete_project(id: str, workspace=Depends(get_workspace)) -> MessageResponse:  # noqa: ANN001
    try:
        workspace.remove_project(id)
    except KeyError:
        raise ProjectNotFoundError(id)  # noqa: B904
    return MessageResponse(message="Project deleted")


# ── Project Assets ──────────────────────────────────────────────────────────


@router.get("/{id}/assets", response_model=list[AssetResponse])
def list_project_assets(
    id: str,
    limit: int = 100,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> list[AssetResponse]:
    """List every asset (any kind) in the project scope via the catalog."""
    project = workspace.get_project(id)
    return [AssetResponse.from_model(a) for a in project.assets.list()[:limit]]


@router.get("/{id}/assets/{asset_id}", response_model=AssetResponse)
def get_project_asset(id: str, asset_id: str, workspace=Depends(get_workspace)) -> AssetResponse:  # noqa: ANN001
    project = workspace.get_project(id)
    asset = project.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)
    return AssetResponse.from_model(asset)


@router.post("/{id}/assets/upload", response_model=AssetResponse, status_code=201)
async def upload_project_asset(
    id: str,
    file: UploadFile = File(...),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AssetResponse:
    """Upload a file into the project's ``DataAssetLibrary``."""
    project = workspace.get_project(id)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        filename = file.filename or "untitled"
        asset = project.data_assets.import_asset(
            name=filename,
            src=tmp_path,
            action="move",
            meta={"original_filename": filename},
        )
        return AssetResponse.from_model(asset)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


@router.get("/{id}/assets/{asset_id}/download")
def download_project_asset(id: str, asset_id: str, workspace=Depends(get_workspace)):  # noqa: ANN001, ANN201
    project = workspace.get_project(id)
    asset = project.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)

    payload_dir = asset.absolute_path(project.project_dir)
    if not payload_dir.exists():
        raise AssetNotFoundError(asset_id)

    if payload_dir.is_dir():
        files = list(payload_dir.iterdir())
        if not files:
            raise AssetNotFoundError(asset_id)
        file_path = files[0]
    else:
        file_path = payload_dir

    filename = asset.tags.get("original_filename") or file_path.name
    return StreamingResponse(
        open(file_path, "rb"),  # noqa: PTH123
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
