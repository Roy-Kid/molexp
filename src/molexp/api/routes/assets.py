"""Asset routes for MolExp API."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse

from molexp.models import Asset, AssetType, AssetFile
from molexp.utils.id import generate_asset_id, compute_content_hash

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError
from ..schemas import AssetResponse

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("", response_model=list[AssetResponse])
def list_assets(
    limit: int = 100,
    workspace=Depends(get_workspace),
) -> list[AssetResponse]:
    """List all assets."""
    assets = workspace.list_assets()[:limit]
    return [AssetResponse.from_model(a) for a in assets]


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: str, workspace=Depends(get_workspace)) -> AssetResponse:
    """Get asset details."""
    asset = workspace.get_asset(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)
    return AssetResponse.from_model(asset)


@router.post("/upload", response_model=AssetResponse, status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    workspace=Depends(get_workspace),
) -> AssetResponse:
    """Upload a new asset."""
    # Create temp file to store upload
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        # Compute hash
        content_hash = compute_content_hash(tmp_path)
        
        # Check if already exists
        existing_id = workspace.assets.exists(content_hash)
        if existing_id:
            asset = workspace.get_asset(existing_id)
            if asset:
                return AssetResponse.from_model(asset)
        
        # Create new asset
        asset_id = generate_asset_id()
        filename = file.filename or "untitled"
        
        asset = Asset(
            asset_id=asset_id,
            type=AssetType.OTHER,
            format=Path(filename).suffix.lstrip(".") or "dat",
            content_hash=content_hash,
            size_bytes=tmp_path.stat().st_size,
            created_at=datetime.now(),
            files=[
                AssetFile(
                    path=filename,
                    size=tmp_path.stat().st_size,
                    hash=content_hash,
                )
            ],
            metadata={"original_filename": filename},
            tags=[],
            producer_run_id=None,
        )
        
        workspace.store_asset(asset, tmp_path)
        return AssetResponse.from_model(asset)
        
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/{asset_id}/download")
def download_asset(asset_id: str, workspace=Depends(get_workspace)):
    """Download asset content."""
    asset = workspace.get_asset(asset_id)
    if not asset:
        raise AssetNotFoundError(asset_id)
    
    asset_dir = workspace.assets.root / asset_id / "data"
    if not asset_dir.exists():
        raise AssetNotFoundError(asset_id)
    
    files = list(asset_dir.iterdir())
    if not files:
        raise AssetNotFoundError(asset_id)
    
    file_path = files[0]
    filename = asset.metadata.get("original_filename") or file_path.name
    
    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
