"""Workspace routes for MolExp API."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from molexp.workspace import Workspace

from ..dependencies import get_workspace, set_workspace_path_override
from ..schemas import (
    FileContentResponse,
    WorkspaceInfoResponse,
    WorkspaceOpenRequest,
    WorkspaceRunRow,
    WorkspaceRunsResponse,
    compute_workspace_runs_stats,
)


class DirectoryCreateRequest(BaseModel):
    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path for new directory")


class FileContentUpdateRequest(BaseModel):
    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path within the folder")
    content: str = Field(..., description="New file content")


router = APIRouter(prefix="/workspace", tags=["workspace"])

MAX_TEXT_BYTES = 2_000_000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_workspace_path(root: Path, path_str: str) -> Path:
    """Resolve a workspace-relative or absolute path within the workspace root."""
    raw_path = Path(path_str).expanduser()
    target = raw_path.resolve() if raw_path.is_absolute() else (root / path_str).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Path is outside workspace root")
    return target


@router.get("/info", response_model=WorkspaceInfoResponse)
def get_workspace_info(workspace=Depends(get_workspace)) -> WorkspaceInfoResponse:
    """Get workspace information."""
    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(workspace.list_projects()),
        assetCount=len(workspace.assets.list()),
    )


@router.get("/runs", response_model=WorkspaceRunsResponse)
def list_workspace_runs(
    project_id: str | None = Query(default=None, alias="projectId"),
    experiment_id: str | None = Query(default=None, alias="experimentId"),
    backend: str | None = Query(default=None, description="Filter by executor backend"),
    status: str | None = Query(default=None, description="Filter by run status"),
    limit: int = Query(default=500, ge=1, le=2000),
    workspace=Depends(get_workspace),
) -> WorkspaceRunsResponse:
    """Cross-experiment list of runs, each with embedded execution attempts.

    Returns rows ordered by ``created_at`` desc.  Plugins surface
    backend-specific columns (cluster, scheduler job id, etc.) via the
    ``backend`` / ``backendMetadata`` fields on each execution row.
    """

    rows: list[WorkspaceRunRow] = []
    for project in workspace.list_projects():
        if project_id and project.id != project_id:
            continue
        project_name = project.name
        for experiment in project.list_experiments():
            if experiment_id and experiment.id != experiment_id:
                continue
            experiment_name = experiment.name
            for run in experiment.list_runs():
                row = WorkspaceRunRow.from_run(
                    run,
                    project_name=project_name,
                    experiment_name=experiment_name,
                )
                if backend and (row.backend or "").lower() != backend.lower():
                    continue
                if status and row.status.lower() != status.lower():
                    continue
                rows.append(row)

    rows.sort(key=lambda r: r.createdAt, reverse=True)
    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]

    return WorkspaceRunsResponse(
        runs=rows,
        stats=compute_workspace_runs_stats(rows),
        total=len(rows),
        truncated=truncated,
    )


@router.get("/files")
def list_workspace_files(
    path: str = Query("", description="Workspace-relative path to list"),
    max_depth: int = Query(4, ge=0, le=8, description="Maximum recursion depth"),
    include: str | None = Query(
        None,
        description="Comma-separated optional enrichments (e.g. 'catalog')",
    ),
    workspace=Depends(get_workspace),
) -> dict:
    """Return a nested file tree rooted at the requested path.

    With ``include=catalog``, file nodes that match a registered asset
    are enriched with ``assetId``, ``assetKind``, ``producerRunId`` and
    ``producerTaskId`` so the UI can render lineage chips inline.
    """
    root = Path(workspace.root).resolve()
    requested = resolve_workspace_path(root, path.lstrip("/"))
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    include_set = {part.strip() for part in (include or "").split(",") if part.strip()}
    asset_index_by_abs: dict[Path, dict] = {}
    if "catalog" in include_set:
        from ._scope import resolve_scope_dir

        for asset in workspace.catalog.query_assets():
            scope_dir = resolve_scope_dir(workspace, asset.scope)
            if scope_dir is None:
                continue
            try:
                abs_path = (scope_dir / asset.path).resolve()
            except OSError:
                continue
            asset_index_by_abs[abs_path] = {
                "assetId": asset.asset_id,
                "assetKind": asset.kind,  # type: ignore[attr-defined]
                "producerRunId": asset.producer.run_id if asset.producer else None,
                "producerTaskId": asset.producer.task_id if asset.producer else None,
            }

    def build_node(node_path: Path, depth: int) -> dict[str, Any]:
        is_file = node_path.is_file()
        node: dict[str, Any] = {
            "id": str(node_path),
            "name": node_path.name or str(node_path),
            "path": str(node_path),
            "type": "file" if is_file else "folder",
            "size": node_path.stat().st_size if is_file else None,
            "modified": node_path.stat().st_mtime,
        }
        if asset_index_by_abs:
            try:
                resolved = node_path.resolve()
            except OSError:
                resolved = node_path
            enrich = asset_index_by_abs.get(resolved)
            if enrich is not None:
                node.update(enrich)
        if not is_file and depth < max_depth:
            children = []
            for child in sorted(node_path.iterdir(), key=lambda p: (p.is_file(), p.name)):
                children.append(build_node(child, depth + 1))
            node["children"] = children
        else:
            node["children"] = []
        return node

    root_node = build_node(requested, 0)
    return {"path": str(requested), "children": root_node.get("children", [])}


@router.get("/file", response_model=FileContentResponse)
def read_workspace_file(
    path: str = Query("", description="Workspace-relative path to read"),
    workspace=Depends(get_workspace),
) -> FileContentResponse:
    """Read a text file from the workspace."""
    root = Path(workspace.root).resolve()
    target = resolve_workspace_path(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    size = target.stat().st_size
    if size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="File too large for text preview")

    content = target.read_text(encoding="utf-8", errors="replace")
    return FileContentResponse(content=content)


@router.get("/file/blob")
def read_workspace_file_blob(
    path: str = Query("", description="Workspace-relative path to read"),
    workspace=Depends(get_workspace),
) -> StreamingResponse:
    """Read a binary file from the workspace."""
    root = Path(workspace.root).resolve()
    target = resolve_workspace_path(root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported binary preview type")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return StreamingResponse(open(target, "rb"), media_type=media_type)


@router.post("/open", response_model=WorkspaceInfoResponse)
def open_workspace(
    request: WorkspaceOpenRequest,
) -> WorkspaceInfoResponse:
    """Set the active workspace path."""
    path = Path(request.path).expanduser().resolve()
    if not path.exists():
        if not request.create_if_missing:
            raise HTTPException(status_code=404, detail="Workspace path not found")
        path.mkdir(parents=True, exist_ok=True)

    set_workspace_path_override(path)
    workspace = Workspace(path)
    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(workspace.list_projects()),
        assetCount=len(workspace.assets.list()),
    )


@router.post("/directories")
def create_directory(
    request: DirectoryCreateRequest,
    workspace=Depends(get_workspace),
) -> dict:
    """Create a directory in the workspace."""
    if request.folder_id != "workspace":
        raise HTTPException(status_code=400, detail="Only workspace folder is supported")

    root = Path(workspace.root).resolve()
    target = resolve_workspace_path(root, request.path)

    target.mkdir(parents=True, exist_ok=True)
    return {"path": str(target)}


@router.put("/files")
def write_file(
    request: FileContentUpdateRequest,
    workspace=Depends(get_workspace),
) -> dict:
    """Create or update a file in the workspace."""
    if request.folder_id != "workspace":
        raise HTTPException(status_code=400, detail="Only workspace folder is supported")

    root = Path(workspace.root).resolve()
    target = resolve_workspace_path(root, request.path)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    target.write_text(request.content, encoding="utf-8")
    return {"path": str(target)}
