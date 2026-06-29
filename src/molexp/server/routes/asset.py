"""Asset routes — unified typed Asset API under ``/api/assets``.

The old routes only served ``DataAsset`` uploads.  The new surface is
backed by the workspace manifest scanner (``assets.scan``, reading the
authoritative per-scope ``assets.json``) and supports every kind
(``data``, ``artifact``, ``log``, ``checkpoint``, …).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from molexp.workspace.assets import AssetScope, LogAsset, lineage, scan

from ..dependencies import get_workspace
from ..exceptions import AssetNotFoundError, InvalidPathError
from ..preview import asset_has_sidecar
from ..schemas import (
    AssetLineageNode,
    AssetLineageResponse,
    AssetResponse,
    DataAssetRegisterRequest,
)
from ._scope import resolve_scope_dir, split_workspace_relpath

router = APIRouter(prefix="/assets", tags=["assets"])


def _require_asset(workspace, asset_id: str):  # noqa: ANN001, ANN202
    asset = scan.get_asset(workspace.root, asset_id)
    if asset is None:
        raise AssetNotFoundError(asset_id)
    return asset


def _resolve_scope_dir(workspace, scope: AssetScope) -> Path:  # noqa: ANN001
    path = resolve_scope_dir(workspace, scope)
    if path is None:
        raise HTTPException(400, f"Could not resolve scope: {scope!r}")
    return path


# ── Query ────────────────────────────────────────────────────────────────


@router.get("", response_model=list[AssetResponse])
def list_assets(
    kind: str | None = None,
    scope_kind: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    limit: int = 100,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> list[AssetResponse]:
    """Query assets from the workspace catalog with optional filters."""
    scope = None
    if scope_kind == "workspace":
        scope = AssetScope(kind="workspace", ids=())
    # Note: project/experiment/run scoping is better served by the
    # per-scope routes below (they carry the full ids tuple).

    assets = scan.scan_assets(
        workspace.root,
        kind=kind,
        scope=scope,
        producer_run=run_id,
        producer_task=task_id,
        limit=limit,
    )
    return [
        AssetResponse.from_model(a, has_preview_sidecar=asset_has_sidecar(workspace, a))
        for a in assets
    ]


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(asset_id: str, workspace=Depends(get_workspace)) -> AssetResponse:  # noqa: ANN001
    asset = _require_asset(workspace, asset_id)
    return AssetResponse.from_model(asset, has_preview_sidecar=asset_has_sidecar(workspace, asset))


@router.get("/{asset_id}/lineage", response_model=AssetLineageResponse)
def get_asset_lineage(asset_id: str, workspace=Depends(get_workspace)) -> AssetLineageResponse:  # noqa: ANN001
    """Return the asset's transitive ancestors and descendants.

    Walks the ``Producer.inputs`` DAG built by run-time tasks that
    declare ``consumed=[...]`` on artifact / data registration. The
    starting asset is excluded from both lists.
    """
    _require_asset(workspace, asset_id)

    def _node(aid: str) -> AssetLineageNode | None:
        a = scan.get_asset(workspace.root, aid)
        if a is None:
            return None
        return AssetLineageNode(id=a.asset_id, name=a.name, kind=a.kind, scope_kind=a.scope.kind)

    ancestor_ids = sorted(lineage.ancestors(workspace, asset_id))
    descendant_ids = sorted(lineage.descendants(workspace, asset_id))
    return AssetLineageResponse(
        asset_id=asset_id,
        ancestors=[n for n in (_node(i) for i in ancestor_ids) if n is not None],
        descendants=[n for n in (_node(i) for i in descendant_ids) if n is not None],
    )


# ── Download / tail / stream ──────────────────────────────────────────────


@router.get("/{asset_id}/content")
def asset_content(asset_id: str, workspace=Depends(get_workspace)):  # noqa: ANN001, ANN201
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
        open(path, "rb"),  # noqa: PTH123
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


@router.get("/{asset_id}/tail", response_class=PlainTextResponse)
def asset_tail(asset_id: str, n: int = 100, workspace=Depends(get_workspace)) -> str:  # noqa: ANN001
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
    workspace=Depends(get_workspace),  # noqa: ANN001
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


@router.post("/data/register", response_model=AssetResponse, status_code=201)
def register_data_asset(
    body: DataAssetRegisterRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AssetResponse:
    """Register an existing workspace file in place as a ``DataAsset``.

    The file stays where it is — only an index entry is created — so a
    same-stem preview sidecar remains a real sibling of the resolved path.
    """
    try:
        target = split_workspace_relpath(workspace, body.path)
    except ValueError as exc:
        raise InvalidPathError(body.path, "path is outside the workspace") from exc
    if not target.exists():
        raise InvalidPathError(body.path, "file does not exist")

    asset = workspace.data_assets.register_in_place(
        name=body.name or target.name,
        src=target,
        meta=body.metadata,
    )
    return AssetResponse.from_model(asset, has_preview_sidecar=asset_has_sidecar(workspace, asset))
