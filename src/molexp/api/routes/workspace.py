"""Workspace routes for MolExp API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends

from molexp.workspace.scanner import FolderScanner

from ..dependencies import (WorkspaceFolderStore, get_workspace,
                            get_workspace_folder_store)
from ..exceptions import (BinaryFileError, FileNotFoundError,
                          FileTooLargeError, FolderAlreadyAddedError,
                          FolderNotFoundError, InvalidPathError,
                          PathExistsError, PathNotDirectoryError,
                          PathNotFileError, PathOutsideWorkspaceError)
from ..schemas import (DashboardStatsResponse, DirectoryCreateRequest,
                       EntityClassificationResponse, FileContentResponse,
                       FileContentUpdateRequest, FolderBrowseResponse,
                       FolderEntryResponse, MessageResponse,
                       WorkspaceFolderAddRequest, WorkspaceFolderResponse,
                       WorkspaceInfoResponse, WorkspaceScanResponse)

router = APIRouter(prefix="/api", tags=["workspace"])


# ============================================================================
# Workspace Info
# ============================================================================


@router.get("/workspace/info", response_model=WorkspaceInfoResponse)
def get_workspace_info(workspace=Depends(get_workspace)) -> WorkspaceInfoResponse:
    """Get workspace information."""
    projects = workspace.list_projects()
    assets = workspace.list_assets()

    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(projects),
        assetCount=len(assets),
    )


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(workspace=Depends(get_workspace)) -> DashboardStatsResponse:
    """Get dashboard statistics."""
    projects = workspace.list_projects()
    assets = workspace.list_assets()

    total_experiments = 0
    total_runs = 0
    recent_experiments = []

    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        total_experiments += len(experiments)

        for exp in experiments:
            runs = workspace.list_runs(project.project_id, exp.experiment_id)
            total_runs += len(runs)

            if len(recent_experiments) < 5:
                recent_experiments.append(
                    {
                        "id": exp.experiment_id,
                        "name": exp.name,
                        "status": "Active",
                        "details": f"Project: {project.name}",
                    }
                )

    return DashboardStatsResponse(
        totalExperiments=total_experiments,
        activeWorkflows=total_runs,
        dataUsage=f"{len(assets) * 1.5:.1f} MB",
        computeHours=f"{total_runs * 0.5:.1f}h",
        recentExperiments=recent_experiments,
    )


@router.get("/workspace/tree")
def get_workspace_tree(workspace=Depends(get_workspace)):
    """Get complete workspace tree structure."""
    projects = workspace.list_projects()
    tree_items = []

    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        experiment_items = []

        for exp in experiments:
            runs = workspace.list_runs(project.project_id, exp.experiment_id)
            run_items = []

            for run in runs:
                run_items.append(
                    {
                        "id": f"{project.project_id}/{exp.experiment_id}/{run.run_id}",
                        "name": run.run_id,
                        "type": "run",
                        "indexed": True,
                        "kind": "run",
                        "schema_version": run.schema_version,
                        "status": run.status.value,
                        "created": run.created_at.isoformat(),
                        "finished": (
                            run.finished_at.isoformat() if run.finished_at else None
                        ),
                        "parameters": run.parameters,
                    }
                )

            # Scan for workflow files
            exp_dir = (
                workspace.root
                / "projects"
                / project.project_id
                / "experiments"
                / exp.experiment_id
            )
            workflow_items = []
            if exp_dir.exists():
                for item in exp_dir.iterdir():
                    if item.is_file() and item.name.endswith(".flow"):
                        workflow_items.append(
                            {
                                "id": f"workspace:projects/{project.project_id}/experiments/{exp.experiment_id}/{item.name}",
                                "name": item.name,
                                "type": "file",
                                "path": f"projects/{project.project_id}/experiments/{exp.experiment_id}/{item.name}",
                                "size": item.stat().st_size,
                            }
                        )

            experiment_items.append(
                {
                    "id": f"{project.project_id}/{exp.experiment_id}",
                    "name": exp.name,
                    "type": "experiment",
                    "indexed": True,
                    "kind": "experiment",
                    "schema_version": exp.schema_version,
                    "experimentId": exp.experiment_id,
                    "workflow": exp.workflow_template.source,
                    "created": exp.created_at.isoformat(),
                    "runCount": len(runs),
                    "children": workflow_items + run_items,
                }
            )

        tree_items.append(
            {
                "id": project.project_id,
                "name": project.name,
                "type": "project",
                "indexed": True,
                "kind": "project",
                "schema_version": project.schema_version,
                "projectId": project.project_id,
                "owner": project.owner,
                "tags": project.tags,
                "created": project.created_at.isoformat(),
                "experimentCount": len(experiments),
                "children": experiment_items,
            }
        )

    return {
        "id": "workspace",
        "name": "Workspace",
        "type": "workspace",
        "children": tree_items,
    }


# ============================================================================
# Workspace Folders
# ============================================================================


@router.get("/workspace/folders", response_model=list[WorkspaceFolderResponse])
def list_workspace_folders(
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> list[WorkspaceFolderResponse]:
    """List all workspace folders."""
    folders = store.list_all()
    return [
        WorkspaceFolderResponse(
            id=f["id"],
            path=f["path"],
            name=f["name"],
            added_at=f["added_at"],
        )
        for f in folders
    ]


@router.post(
    "/workspace/folders", response_model=WorkspaceFolderResponse, status_code=201
)
def add_workspace_folder(
    folder: WorkspaceFolderAddRequest,
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> WorkspaceFolderResponse:
    """Add a workspace folder."""
    folder_path = Path(folder.path).resolve()

    if not folder_path.exists():
        raise InvalidPathError(folder.path, "Path does not exist")

    if not folder_path.is_dir():
        raise PathNotDirectoryError(folder.path)

    existing = store.find_by_path(folder_path)
    if existing:
        raise FolderAlreadyAddedError(folder.path)

    folder_id = str(uuid4())[:8]
    folder_name = folder.name or folder_path.name
    added_at = datetime.now().isoformat()

    store.add(folder_id, str(folder_path), folder_name, added_at)

    return WorkspaceFolderResponse(
        id=folder_id,
        path=str(folder_path),
        name=folder_name,
        added_at=added_at,
    )


@router.delete("/workspace/folders/{folder_id}", response_model=MessageResponse)
def remove_workspace_folder(
    folder_id: str,
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> MessageResponse:
    """Remove a workspace folder."""
    if not store.remove(folder_id):
        raise FolderNotFoundError(folder_id)
    return MessageResponse(message="Folder removed")


@router.get(
    "/workspace/folders/{folder_id}/browse", response_model=FolderBrowseResponse
)
def browse_workspace_folder(
    folder_id: str,
    path: str = "",
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> FolderBrowseResponse:
    """Browse contents of a workspace folder."""
    folder = store.get(folder_id)
    if not folder:
        raise FolderNotFoundError(folder_id)

    base_path = Path(folder["path"])
    full_path = (base_path / path).resolve() if path else base_path

    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise PathOutsideWorkspaceError(str(full_path))

    if not full_path.exists():
        raise FileNotFoundError(path)

    if not full_path.is_dir():
        raise PathNotDirectoryError(path)

    entries = []
    try:
        for item in sorted(
            full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
        ):
            entry = FolderEntryResponse(
                name=item.name,
                path=str(item.relative_to(base_path)),
                type="directory" if item.is_dir() else "file",
                size=item.stat().st_size if item.is_file() else None,
            )
            entries.append(entry)
    except PermissionError:
        raise PathOutsideWorkspaceError(str(full_path))

    return FolderBrowseResponse(path=path, entries=entries)


# ============================================================================
# File Operations
# ============================================================================

MAX_FILE_SIZE = 1024 * 1024  # 1MB


def _resolve_file_path(
    folder_id: str, path: str, workspace, store: WorkspaceFolderStore
) -> tuple[Path, Path]:
    """Resolve full path and base path for a file operation."""
    if folder_id == "workspace":
        base_path = workspace.root
    else:
        folder = store.get(folder_id)
        if not folder:
            raise FolderNotFoundError(folder_id)
        base_path = Path(folder["path"])

    full_path = (base_path / path).resolve()

    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise PathOutsideWorkspaceError(str(full_path))

    return full_path, base_path


@router.get("/workspace/files/content", response_model=FileContentResponse)
def read_workspace_file(
    folder_id: str,
    path: str,
    workspace=Depends(get_workspace),
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> FileContentResponse:
    """Read content of a file."""
    full_path, _ = _resolve_file_path(folder_id, path, workspace, store)

    if not full_path.exists():
        raise FileNotFoundError(path)

    if not full_path.is_file():
        raise PathNotFileError(path)

    if full_path.stat().st_size > MAX_FILE_SIZE:
        raise FileTooLargeError(path, full_path.stat().st_size, MAX_FILE_SIZE)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return FileContentResponse(content=content)
    except UnicodeDecodeError:
        raise BinaryFileError(path)


@router.put("/workspace/files/content", response_model=MessageResponse)
def write_workspace_file(
    update: FileContentUpdateRequest,
    workspace=Depends(get_workspace),
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> MessageResponse:
    """Write content to a file."""
    full_path, _ = _resolve_file_path(update.folder_id, update.path, workspace, store)

    if not full_path.parent.exists():
        raise FileNotFoundError(str(full_path.parent))

    if full_path.exists() and not full_path.is_file():
        raise PathNotFileError(update.path)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(update.content)

    return MessageResponse(message="File saved successfully")


@router.post("/workspace/files/directory", response_model=MessageResponse)
def create_workspace_directory(
    data: DirectoryCreateRequest,
    workspace=Depends(get_workspace),
    store: WorkspaceFolderStore = Depends(get_workspace_folder_store),
) -> MessageResponse:
    """Create a directory."""
    full_path, _ = _resolve_file_path(data.folder_id, data.path, workspace, store)

    if full_path.exists():
        raise PathExistsError(data.path)

    full_path.mkdir(parents=True, exist_ok=True)
    return MessageResponse(message="Directory created")


# ============================================================================
# Entity Classification
# ============================================================================


@router.get("/entities/{kind}/{entity_id}")
def get_entity_metadata(
    kind: str,
    entity_id: str,
    workspace=Depends(get_workspace),
):
    """Get metadata for any indexed entity."""
    scanner = FolderScanner(workspace.root)

    if kind == "project":
        folder_path = workspace.root / "projects" / entity_id
    elif kind == "asset":
        folder_path = workspace.root / "assets" / entity_id
    else:
        raise InvalidPathError(entity_id, f"Cannot directly access {kind} entities")

    if not folder_path.exists():
        raise FileNotFoundError(entity_id)

    entity_info = scanner.scan_folder(folder_path)
    if not entity_info:
        raise FileNotFoundError(entity_id)

    return {
        "kind": entity_info["kind"],
        "indexed": True,
        "path": entity_info["path"],
        "metadata": entity_info["entity"].model_dump(mode="json"),
    }


@router.get("/workspace/classify", response_model=EntityClassificationResponse)
def classify_folder(
    path: str, workspace=Depends(get_workspace)
) -> EntityClassificationResponse:
    """Classify a folder as an indexed entity or generic folder."""
    scanner = FolderScanner(workspace.root)
    folder_path = workspace.root / path

    if not folder_path.exists():
        raise FileNotFoundError(path)

    if not folder_path.is_dir():
        raise PathNotDirectoryError(path)

    entity_info = scanner.scan_folder(folder_path)

    if entity_info:
        return EntityClassificationResponse(
            indexed=True,
            kind=entity_info["kind"],
            path=entity_info["path"],
            metadata=entity_info["entity"].model_dump(mode="json"),
        )
    else:
        return EntityClassificationResponse(
            indexed=False,
            kind="folder",
            path=str(folder_path.relative_to(workspace.root)),
        )


@router.get("/workspace/scan", response_model=WorkspaceScanResponse)
def scan_workspace(workspace=Depends(get_workspace)) -> WorkspaceScanResponse:
    """Scan entire workspace and return all indexed entities."""
    scanner = FolderScanner(workspace.root)
    entities = scanner.scan_workspace()

    return WorkspaceScanResponse(
        total=len(entities),
        entities=[
            EntityClassificationResponse(
                indexed=True,
                kind=e["kind"],
                path=e["path"],
                metadata=e["entity"].model_dump(mode="json"),
            )
            for e in entities
        ],
    )
