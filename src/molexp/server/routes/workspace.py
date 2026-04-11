"""Workspace routes for MolExp API."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from molexp.workspace import Workspace

from ..dependencies import get_workspace, set_workspace_path_override
from ..schemas import (
    FileContentResponse,
    WorkspaceInfoResponse,
    WorkspaceOpenRequest,
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
        assetCount=len(workspace.assets.list_assets()),
    )

@router.get("/files")
def list_workspace_files(
    path: str = Query("", description="Workspace-relative path to list"),
    max_depth: int = Query(4, ge=0, le=8, description="Maximum recursion depth"),
    workspace=Depends(get_workspace),
) -> dict:
    """Return a nested file tree rooted at the requested path."""
    root = Path(workspace.root).resolve()
    requested = resolve_workspace_path(root, path.lstrip("/"))
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    def build_node(node_path: Path, depth: int) -> dict:
        is_file = node_path.is_file()
        node = {
            "id": str(node_path),
            "name": node_path.name or str(node_path),
            "path": str(node_path),
            "type": "file" if is_file else "folder",
            "size": node_path.stat().st_size if is_file else None,
            "modified": node_path.stat().st_mtime,
        }
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
    workspace = Workspace.from_path(path)
    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(workspace.list_projects()),
        assetCount=len(workspace.assets.list_assets()),
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
