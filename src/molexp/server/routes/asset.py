"""Asset routes — unified typed Asset API under ``/api/assets``.

The old routes only served ``DataAsset`` uploads.  The new surface is
backed by the workspace ``AssetCatalog`` and supports every kind
(``data``, ``artifact``, ``log``, ``checkpoint``, …).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from molexp.workspace.assets import AssetScope, LogAsset

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError
from ..schemas import AssetResponse

router = APIRouter(prefix="/assets", tags=["assets"])


def _require_asset(workspace, asset_id: str):
    asset = workspace.catalog.get(asset_id)
    if asset is None:
        raise AssetNotFoundError(asset_id)
    return asset


def _resolve_scope_dir(workspace, scope: AssetScope) -> Path:
    """Return the on-disk directory for the given scope."""
    if scope.kind == "workspace":
        return workspace.root
    if scope.kind == "project":
        return workspace.root / "projects" / scope.ids[0]
    if scope.kind == "experiment":
        return workspace.root / "projects" / scope.ids[0] / "experiments" / scope.ids[1]
    if scope.kind == "run":
        return (
            workspace.root
            / "projects"
            / scope.ids[0]
            / "experiments"
            / scope.ids[1]
            / "runs"
            / scope.ids[2]
        )
    raise HTTPException(400, f"Unknown scope kind: {scope.kind}")


# ── Query ────────────────────────────────────────────────────────────────


@router.get("", response_model=list[AssetResponse])
def list_assets(
    kind: str | None = None,
    scope_kind: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    limit: int = 100,
    workspace=Depends(get_workspace),
) -> list[AssetResponse]:
    """Query assets from the workspace catalog with optional filters."""
    scope = None
    if scope_kind == "workspace":
        scope = AssetScope(kind="workspace", ids=())
    # Note: project/experiment/run scoping is better served by the
    # per-scope routes below (they carry the full ids tuple).

    assets = workspace.catalog.query_assets(
        kind=kind,
        scope=scope,
        producer_run=run_id,
        producer_task=task_id,
        limit=limit,
    )
    return [AssetResponse.from_model(a) for a in assets]


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: str, workspace=Depends(get_workspace)) -> AssetResponse:
    asset = _require_asset(workspace, asset_id)
    return AssetResponse.from_model(asset)


# ── Download / tail / stream ──────────────────────────────────────────────


@router.get("/{asset_id}/content")
def asset_content(asset_id: str, workspace=Depends(get_workspace)):
    """Download the asset's file content."""
    asset = _require_asset(workspace, asset_id)
    scope_dir = _resolve_scope_dir(workspace, asset.scope)
    path = asset.absolute_path(scope_dir)

    # DataAsset has a payload directory; serve the first file inside it
    if path.is_dir():
        files = list(path.iterdir())
        if not files:
            raise AssetNotFoundError(asset_id)
        path = files[0]

    if not path.exists():
        raise AssetNotFoundError(asset_id)

    return StreamingResponse(
        open(path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@router.get("/{asset_id}/tail", response_class=PlainTextResponse)
def asset_tail(asset_id: str, n: int = 100, workspace=Depends(get_workspace)) -> str:
    """Return the last N lines (``LogAsset`` only)."""
    asset = _require_asset(workspace, asset_id)
    if not isinstance(asset, LogAsset):
        raise HTTPException(400, f"tail only supported for log assets (got {asset.kind})")
    scope_dir = _resolve_scope_dir(workspace, asset.scope)
    return "\n".join(asset.tail(scope_dir, n))


# ── DataAsset import ──────────────────────────────────────────────────────


@router.post("/data/import", response_model=AssetResponse, status_code=201)
async def import_data_asset(
    file: UploadFile = File(...),
    workspace=Depends(get_workspace),
) -> AssetResponse:
    """Upload a file and register it as a workspace-scoped ``DataAsset``."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        filename = file.filename or "untitled"
        asset = workspace.data_assets.import_asset(
            name=filename,
            src=tmp_path,
            action="move",
            meta={"original_filename": filename},
        )
        return AssetResponse.from_model(asset)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
