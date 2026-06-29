"""Catalog routes — reverse lookup from a workspace-relative path to producer.

The asset catalog (``workspace.catalog``) records every artifact, log, and
checkpoint together with its ``producer`` and ``scope``.  The UI needs a
reverse direction: given a path, which run produced it, which experiment
groups it, which project owns it, and what other outputs share its task.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from molexp.workspace.assets import Asset, scan

from ..dependencies import get_workspace
from ..schemas import (
    CatalogByPathResponse,
    CatalogProducerInfo,
    CatalogScopeInfo,
    CatalogSibling,
)
from ._scope import resolve_scope_dir

router = APIRouter(prefix="/catalog", tags=["catalog"])


def _scope_info(asset: Asset) -> CatalogScopeInfo:
    ids = asset.scope.ids
    return CatalogScopeInfo(
        kind=asset.scope.kind,
        projectId=ids[0] if len(ids) >= 1 else None,
        experimentId=ids[1] if len(ids) >= 2 else None,
        runId=ids[2] if len(ids) >= 3 else None,
    )


def _producer_info(asset: Asset) -> CatalogProducerInfo | None:
    if asset.producer is None:
        return None
    return CatalogProducerInfo(
        runId=asset.producer.run_id,
        taskId=asset.producer.task_id,
        executionId=asset.producer.execution_id,
    )


def _siblings(workspace, asset: Asset) -> list[CatalogSibling]:  # noqa: ANN001
    """Return other assets produced by the same task in the same scope."""
    if asset.producer is None or asset.producer.task_id is None:
        return []
    peers = scan.scan_assets(
        workspace.root,
        scope=asset.scope,
        producer_task=asset.producer.task_id,
    )
    out: list[CatalogSibling] = []
    for peer in peers:
        if peer.asset_id == asset.asset_id:
            continue
        out.append(
            CatalogSibling(
                assetId=peer.asset_id,
                name=peer.name,
                kind=peer.kind,  # type: ignore[attr-defined]
                relPath=str(peer.path),
            )
        )
    return out


@router.get("/by-path", response_model=CatalogByPathResponse)
def catalog_by_path(
    path: str = Query(..., description="Workspace-relative or absolute path"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> CatalogByPathResponse:
    """Reverse lookup: find the producer for a given workspace path.

    Accepts either an absolute path (must be inside the workspace) or a
    workspace-relative path. Rejects absolute paths outside the workspace
    root with HTTP 400.
    """
    root = Path(workspace.root).resolve()
    raw = Path(path).expanduser()
    if raw.is_absolute():
        target = raw.resolve()
    else:
        target = (root / path.lstrip("/")).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path is outside workspace root") from exc

    workspace_rel = str(target.relative_to(root))

    # Scan all assets, prefer exact path match (to its scope dir).
    assets = scan.scan_assets(workspace.root)
    for asset in assets:
        scope_dir = resolve_scope_dir(workspace, asset.scope)
        if scope_dir is None:
            continue
        try:
            asset_abs = (scope_dir / asset.path).resolve()
        except OSError:
            continue
        if asset_abs == target:
            return CatalogByPathResponse(
                matched=True,
                workspaceRelPath=workspace_rel,
                assetId=asset.asset_id,
                assetKind=asset.kind,  # type: ignore[attr-defined]
                producer=_producer_info(asset),
                scope=_scope_info(asset),
                siblings=_siblings(workspace, asset),
            )

    # Not a registered asset; still attempt to derive scope from path shape.
    derived = _derive_scope_from_path(workspace_rel)
    return CatalogByPathResponse(
        matched=False,
        workspaceRelPath=workspace_rel,
        scope=derived,
    )


def _derive_scope_from_path(rel_path: str) -> CatalogScopeInfo | None:
    """Extract project/experiment/run ids from a path shape under ``projects/``.

    Recognises::

        projects/<p>
        projects/<p>/experiments/<e>
        projects/<p>/experiments/<e>/runs/run-<r>/...
    """
    parts = Path(rel_path).parts
    if not parts or parts[0] != "projects":
        return None
    project_id: str | None = parts[1] if len(parts) >= 2 else None
    experiment_id: str | None = None
    run_id: str | None = None
    if len(parts) >= 4 and parts[2] == "experiments":
        experiment_id = parts[3]
    if len(parts) >= 6 and parts[4] == "runs":
        run_dir_name = parts[5]
        run_id = run_dir_name[4:] if run_dir_name.startswith("run-") else run_dir_name

    kind = "workspace"
    if run_id:
        kind = "run"
    elif experiment_id:
        kind = "experiment"
    elif project_id:
        kind = "project"

    return CatalogScopeInfo(
        kind=kind,
        projectId=project_id,
        experimentId=experiment_id,
        runId=run_id,
    )
