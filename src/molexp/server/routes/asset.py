"""Asset routes for MolExp API."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError
from ..schemas import AssetResponse

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("", response_model=list[AssetResponse])
def list_assets(
    limit: int = 100,
    workspace=Depends(get_workspace),
) -> list[AssetResponse]:
    """List all assets."""
    # Using workspace.assets property which returns AssetLibrary
    assets = workspace.assets.list_assets()[:limit]
    return [AssetResponse.from_model(a) for a in assets]


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: str, workspace=Depends(get_workspace)) -> AssetResponse:
    """Get asset details."""
    asset = workspace.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)
    return AssetResponse.from_model(asset)


@router.post("/upload", response_model=AssetResponse, status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    workspace=Depends(get_workspace),
) -> AssetResponse:
    """Upload a new asset."""
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        filename = file.filename or "untitled"
        
        # Use workspace.assets (AssetLibrary) to import
        asset = workspace.assets.import_asset(
            path=tmp_path,
            name=filename,
            metadata={"original_filename": filename},
            action="move" # or copy, depending on tempfile behaviour
        )
        return AssetResponse.from_model(asset)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/{asset_id}/download")
def download_asset(asset_id: str, workspace=Depends(get_workspace)):
    """Download asset content."""
    asset = workspace.assets.get(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)
        
    payload_dir = workspace.assets.get_payload_dir(asset_id)
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
