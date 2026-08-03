"""Workspace routes for MolExp API."""

from __future__ import annotations

import io
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONValue
from molexp.workspace import ContextFocus, Workspace, assemble_workspace_context
from molexp.workspace.events import WorkspaceEvent, WorkspaceEventType, read_workspace_events
from molexp.workspace.fs_cached import CachedRemoteFileSystem, prefetch_workspace_indices
from molexp.workspace.fs_local import LocalFileSystem

from ..dependencies import (
    get_remote_fs_factory,
    get_workspace,
    get_workspace_target_registry,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)
from ..preview import resolve_sidecar
from ..schemas import (
    FileContentResponse,
    TargetTestCheck,
    TargetTestResponse,
    WorkspaceContextResponse,
    WorkspaceInfoResponse,
    WorkspaceOpenLocalRequest,
    WorkspaceOpenRequest,
    WorkspaceRunRow,
    WorkspaceRunsResponse,
    WorkspaceSummaryResponse,
    WorkspaceTargetCreateRequest,
    WorkspaceTargetListResponse,
    WorkspaceTargetResponse,
    compute_workspace_runs_stats,
)
from ..workspace_targets import WorkspaceTarget

if TYPE_CHECKING:
    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest


class DirectoryCreateRequest(BaseModel):
    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path for new directory")


class FileContentUpdateRequest(BaseModel):
    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path within the folder")
    content: str = Field(..., description="New file content")


router = APIRouter(prefix="/workspace", tags=["workspace"])

# The activity stream mounts at the literal ``/api/events`` (no ``/workspace``
# prefix) — same flat-router precedent as ``plans.flat_router``.
events_router = APIRouter(tags=["workspace"])


class WorkspaceEventResponse(BaseModel):
    """One workspace-timeline event (read side of the event spine).

    The ONE wire shape for spine reads — the per-run route
    (``GET /runs/{run_id}/events``) aliases this model, so the two surfaces
    can never drift (vision-loop-12).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    seq: int
    type: str
    actor: str
    created_at: datetime
    payload: dict[str, JSONValue]
    refs: list[str]

    @classmethod
    def from_event(cls, event: WorkspaceEvent) -> WorkspaceEventResponse:
        """The one event→wire mapping (both routes call this — no drift)."""
        return cls(
            id=event.id,
            seq=event.seq,
            type=event.type,
            actor=event.actor,
            created_at=event.created_at,
            payload=event.payload,
            refs=event.refs,
        )


@events_router.get("/events", response_model=list[WorkspaceEventResponse])
def get_workspace_events(
    type: WorkspaceEventType | None = Query(default=None, description="Keep only this event type"),
    ref: str | None = Query(default=None, description="Keep only events referencing this id"),
    limit: int = Query(default=50, ge=1, le=500),
    workspace: Workspace = Depends(get_workspace),
) -> list[WorkspaceEventResponse]:
    """The workspace-wide activity stream, newest first.

    The global read over the event spine — the same shared
    :func:`molexp.workspace.events.read_workspace_events` code path the
    per-run route and ``molexp runs info`` use. A workspace with no timeline
    yet answers ``[]`` without creating the DB (reading is side-effect free).
    """
    events = read_workspace_events(workspace.root, type=type, ref=ref, limit=limit)
    return [WorkspaceEventResponse.from_event(e) for e in events]


MAX_TEXT_BYTES = 2_000_000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_workspace_path(root: Path, path_str: str) -> Path:
    """Resolve a workspace-relative or absolute path within the workspace root."""
    raw_path = Path(path_str).expanduser()
    target = raw_path.resolve() if raw_path.is_absolute() else (root / path_str).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Path is outside workspace root")
    return target


def resolve_workspace_path_via_fs(workspace, path_str: str) -> str:  # noqa: ANN001
    """Filesystem-aware variant of :func:`resolve_workspace_path`.

    Works for both local and remote workspaces by going through
    ``workspace._fs`` rather than ``pathlib.Path``.  For pure local
    workspaces (``_fs is LocalFileSystem``) it preserves the existing
    ``Path.resolve()`` containment check so symlink escapes are still
    caught.  For any non-local backend (e.g. a remote workspace wrapped
    in :class:`CachedRemoteFileSystem`) it does string-level
    containment against the remote root.
    """
    fs = workspace._fs
    root = str(workspace.root)
    if isinstance(fs, LocalFileSystem):
        resolved = resolve_workspace_path(Path(root).resolve(), path_str)
        return str(resolved)

    normalized_root = root.rstrip("/") or "/"
    if not path_str or path_str in {"/", "."}:
        return normalized_root

    if path_str.startswith("/"):
        candidate = path_str
    else:
        candidate = fs.join(normalized_root, path_str)
    candidate = candidate.rstrip("/")
    if candidate != normalized_root and not candidate.startswith(normalized_root + "/"):
        raise HTTPException(status_code=400, detail="Path is outside workspace root")
    return candidate


@router.get("/info", response_model=WorkspaceInfoResponse)
def get_workspace_info(workspace=Depends(get_workspace)) -> WorkspaceInfoResponse:  # noqa: ANN001
    """Get workspace information."""
    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(workspace.list_projects()),
        assetCount=len(workspace.assets.list()),
    )


@router.get("/context", response_model=WorkspaceContextResponse)
def get_workspace_context(
    project_id: str | None = Query(default=None, alias="projectId"),
    experiment_id: str | None = Query(default=None, alias="experimentId"),
    run_id: str | None = Query(default=None, alias="runId"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> WorkspaceContextResponse:
    """The canonical structural workspace read-model (integration.md §1).

    A read-only projection assembled from authoritative workspace state — the one
    shape agents/planners/CLI/UI observe. ``ContextFocus`` is supplied by the
    caller via optional query params and is never persisted. ``/runs`` remains the
    specialized detailed run view (richer per-execution rows); this endpoint is the
    canonical *structure* and stays consistent with it.
    """
    focus = ContextFocus(project_id=project_id, experiment_id=experiment_id, run_id=run_id)
    context = assemble_workspace_context(workspace, focus=focus)
    return WorkspaceContextResponse.from_context(context)


@router.get("/copilot", response_model=WorkspaceSummaryResponse)
def get_workspace_copilot(workspace=Depends(get_workspace)) -> WorkspaceSummaryResponse:  # noqa: ANN001
    """The read-only Workspace Copilot summary — structured state + ranked next-actions.

    A pure projection over the canonical ``WorkspaceContext``; it mutates nothing.
    Next-actions are **advisory** and separated from execution — high-risk ones are
    flagged ``requiresProposal`` (they must go through a ``ChangeProposal`` first).
    """
    from molexp.harness.copilot import summarize_workspace

    summary = summarize_workspace(assemble_workspace_context(workspace))
    return WorkspaceSummaryResponse.from_summary(summary)


@router.get("/runs", response_model=WorkspaceRunsResponse)
def list_workspace_runs(
    project_id: str | None = Query(default=None, alias="projectId"),
    experiment_id: str | None = Query(default=None, alias="experimentId"),
    backend: str | None = Query(default=None, description="Filter by executor backend"),
    status: str | None = Query(default=None, description="Filter by run status"),
    limit: int = Query(default=500, ge=1, le=2000),
    workspace=Depends(get_workspace),  # noqa: ANN001
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
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> dict:
    """Return a nested file tree rooted at the requested path.

    With ``include=catalog``, file nodes that match a registered asset
    are enriched with ``assetId``, ``assetKind``, ``producerRunId`` and
    ``producerTaskId`` so the UI can render lineage chips inline.

    Children matching the workspace ``.gitignore`` cascade (plus a safety
    floor for ``node_modules`` / ``.git`` / venvs) are omitted so git-managed
    workspaces do not dump dependency trees into the UI.
    """
    from molexp.workspace.gitignore import load_gitignore_matcher

    root = Path(workspace.root).resolve()
    requested = resolve_workspace_path(root, path.lstrip("/"))
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    ignore = load_gitignore_matcher(root, fs=getattr(workspace, "_fs", None))

    include_set = {part.strip() for part in (include or "").split(",") if part.strip()}
    asset_index_by_abs: dict[Path, dict] = {}
    if "catalog" in include_set:
        from molexp.workspace.assets import scan

        from ._scope import resolve_scope_dir

        for asset in scan.scan_assets(workspace.root):
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
                "hasPreviewSidecar": resolve_sidecar(abs_path) is not None,
            }

    def _rel_for(node_path: Path) -> str | None:
        try:
            return node_path.resolve().relative_to(root).as_posix()
        except OSError, ValueError:
            return None

    def build_node(
        node_path: Path, depth: int, *, _visited: set[Path] | None = None
    ) -> dict[str, Any]:
        visited = _visited if _visited is not None else set()
        try:
            real = node_path.resolve()
        except OSError:
            real = node_path
        if real in visited:
            return {
                "id": str(node_path),
                "name": node_path.name or str(node_path),
                "path": str(node_path),
                "type": "folder",
                "size": None,
                "modified": None,
                "children": [],
            }
        visited.add(real)

        try:
            is_file = node_path.is_file()
            st = node_path.stat()
        except OSError:
            return {
                "id": str(node_path),
                "name": node_path.name or str(node_path),
                "path": str(node_path),
                "type": "folder",
                "size": None,
                "modified": None,
                "children": [],
            }

        node: dict[str, Any] = {
            "id": str(node_path),
            "name": node_path.name or str(node_path),
            "path": str(node_path),
            "type": "file" if is_file else "folder",
            "size": st.st_size if is_file else None,
            "modified": st.st_mtime,
        }
        if asset_index_by_abs:
            enrich = asset_index_by_abs.get(real)
            if enrich is not None:
                node.update(enrich)
        if not is_file and depth < max_depth:
            children = []
            try:
                kids = sorted(node_path.iterdir(), key=lambda p: (p.is_file(), p.name))
            except OSError:
                kids = []
            for child in kids:
                rel = _rel_for(child)
                if rel is None:
                    continue
                try:
                    child_is_dir = child.is_dir()
                except OSError:
                    continue
                if ignore.is_ignored(rel, is_dir=child_is_dir):
                    continue
                children.append(build_node(child, depth + 1, _visited=visited))
            node["children"] = children
        else:
            node["children"] = []
        return node

    root_node = build_node(requested, 0)
    return {"path": str(requested), "children": root_node.get("children", [])}


@router.get("/file", response_model=FileContentResponse)
def read_workspace_file(
    path: str = Query("", description="Workspace-relative path to read"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> FileContentResponse:
    """Read a text file from the workspace.

    Routes through ``workspace._fs`` so remote workspaces (and the
    :class:`CachedRemoteFileSystem` mirror) take effect.
    """
    target = resolve_workspace_path_via_fs(workspace, path)
    fs = workspace._fs
    if not fs.exists(target) or not fs.is_file(target):
        raise HTTPException(status_code=404, detail="File not found")

    size = fs.getsize(target)
    if size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="File too large for text preview")

    try:
        content = fs.read_text(target, encoding="utf-8")
    except UnicodeDecodeError:
        content = fs.read_bytes(target).decode("utf-8", errors="replace")
    return FileContentResponse(content=content)


@router.get("/file/blob")
def read_workspace_file_blob(
    path: str = Query("", description="Workspace-relative path to read"),
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> StreamingResponse:
    """Read a binary file from the workspace.

    Routes through ``workspace._fs`` so remote workspaces (and the
    :class:`CachedRemoteFileSystem` mirror) take effect.
    """
    target = resolve_workspace_path_via_fs(workspace, path)
    fs = workspace._fs
    if not fs.exists(target) or not fs.is_file(target):
        raise HTTPException(status_code=404, detail="File not found")

    name = fs.basename(target)
    suffix = ("." + name.rsplit(".", 1)[-1]).lower() if "." in name else ""
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported binary preview type")

    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    data = fs.read_bytes(target)
    return StreamingResponse(io.BytesIO(data), media_type=media_type)


@router.post("/open", response_model=WorkspaceInfoResponse)
def open_workspace(
    request: WorkspaceOpenRequest,
    registry=Depends(get_workspace_target_registry),  # noqa: ANN001
) -> WorkspaceInfoResponse:
    """Set the active workspace — local path or registered remote descriptor.

    Switching the active workspace drains any registered workspace
    subscribers (SSE streams, file watchers — registered via
    :func:`~molexp.server.dependencies.register_workspace_subscriber`)
    *before* the cache is reset, so the new workspace starts from a
    clean subscriber slate.
    """
    if isinstance(request, WorkspaceOpenLocalRequest):
        path = Path(request.path).expanduser().resolve()
        created = False
        if not path.exists():
            if not request.create_if_missing:
                raise HTTPException(status_code=404, detail="Workspace path not found")
            path.mkdir(parents=True, exist_ok=True)
            created = True

        set_workspace_path_override(path)
        workspace = Workspace(path)
        if created:
            # Only a just-created directory is materialized — opening an
            # existing path must never write workspace.json on its own.
            workspace.materialize()
        return WorkspaceInfoResponse(
            root=str(workspace.root),
            projectCount=len(workspace.list_projects()),
            assetCount=len(workspace.assets.list()),
        )

    # Remote branch
    try:
        target = registry.get(request.name)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"workspace target {request.name!r} not found",
        ) from exc

    from ..workspace_targets import target_to_filesystem_for_workspace_target

    fs = target_to_filesystem_for_workspace_target(target)
    set_active_workspace_descriptor(target.name)
    workspace = Workspace(target.root_path, fs=fs)
    warnings = prefetch_workspace_indices(workspace)
    return WorkspaceInfoResponse(
        root=str(workspace.root),
        projectCount=len(workspace.list_projects()),
        assetCount=len(workspace.assets.list()),
        warnings=[f"{w.path}: {w.reason}" for w in warnings],
    )


@router.post("/directories")
def create_directory(
    request: DirectoryCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
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
    workspace=Depends(get_workspace),  # noqa: ANN001
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


# ============================================================================
# Workspace-target registry endpoints
# ============================================================================
#
# A *workspace target* is a server-process-scoped descriptor that names
# a remote workspace root.  These endpoints CRUD the registry and probe
# connectivity; the active-workspace switch (which actually mounts the
# remote root) lives in sub-spec 02.


@router.get("/targets", response_model=WorkspaceTargetListResponse)
def list_workspace_targets(
    registry=Depends(get_workspace_target_registry),  # noqa: ANN001
) -> WorkspaceTargetListResponse:
    rows = [WorkspaceTargetResponse.from_model(t) for t in registry.list()]
    return WorkspaceTargetListResponse(targets=rows, total=len(rows))


@router.post("/targets", response_model=WorkspaceTargetResponse, status_code=201)
def create_workspace_target(
    payload: WorkspaceTargetCreateRequest,
    registry=Depends(get_workspace_target_registry),  # noqa: ANN001
) -> WorkspaceTargetResponse:
    try:
        target = WorkspaceTarget(
            name=payload.name,
            host=payload.host,
            root_path=payload.root_path,
            port=payload.port,
            identity_file=payload.identity_file,
            ssh_opts=tuple(payload.ssh_opts),
            cache_dir=payload.cache_dir,
            cache_ttl_seconds=payload.cache_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        registry.add(target)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return WorkspaceTargetResponse.from_model(target)


@router.delete("/targets/{name}", status_code=204)
def delete_workspace_target(
    name: str,
    registry=Depends(get_workspace_target_registry),  # noqa: ANN001
) -> None:
    try:
        registry.remove(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"workspace target {name!r} not found") from exc


@router.post("/targets/{name}/test", response_model=TargetTestResponse)
def test_workspace_target(
    name: str,
    registry=Depends(get_workspace_target_registry),  # noqa: ANN001
    fs_factory=Depends(get_remote_fs_factory),  # noqa: ANN001
) -> TargetTestResponse:
    """Connectivity probe for a workspace-target descriptor.

    Returns HTTP 200 with ``ok=False`` on probe failure (matches the
    ``/api/targets/{name}/test`` pattern) so the UI can render failures
    inline rather than parsing HTTP error envelopes.
    """
    try:
        target = registry.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"workspace target {name!r} not found") from exc

    fs = fs_factory(target)
    checks: list[TargetTestCheck] = []

    # 1. mkdir root_path
    try:
        fs.mkdir(target.root_path, parents=True, exist_ok=True)
        checks.append(TargetTestCheck(label=f"mkdir {target.root_path}", ok=True))
    except Exception as exc:
        checks.append(
            TargetTestCheck(
                label=f"mkdir {target.root_path}",
                ok=False,
                detail=str(exc),
            )
        )
        return TargetTestResponse(
            name=name,
            ok=False,
            checks=checks,
            error=f"mkdir failed: {exc}",
        )

    # 2. file round-trip (write → read → remove)
    probe_path = f"{target.root_path.rstrip('/')}/.molexp-workspace-test"
    try:
        fs.write_text(probe_path, "ok")
        if fs.read_text(probe_path) != "ok":
            checks.append(
                TargetTestCheck(
                    label="file round-trip",
                    ok=False,
                    detail="content mismatch",
                )
            )
            return TargetTestResponse(
                name=name,
                ok=False,
                checks=checks,
                error="file round-trip mismatch",
            )
        fs.remove(probe_path)
        checks.append(TargetTestCheck(label="file round-trip", ok=True))
    except Exception as exc:
        checks.append(
            TargetTestCheck(
                label="file round-trip",
                ok=False,
                detail=str(exc),
            )
        )
        return TargetTestResponse(
            name=name,
            ok=False,
            checks=checks,
            error=f"round-trip failed: {exc}",
        )

    return TargetTestResponse(name=name, ok=True, checks=checks, error=None)


# ============================================================================
# Remote-workspace cache control
# ============================================================================
#
# The active workspace's :class:`CachedRemoteFileSystem` mirrors remote
# bytes locally.  These endpoints let the UI invalidate or refresh the
# mirror without having to re-open the workspace.  Local workspaces have
# no cache; the endpoints respond ``409 Conflict`` rather than 404 so the
# UI can distinguish "no such cache" from "workspace not found".


class CacheControlRequest(BaseModel):
    """Body for ``POST /api/workspace/cache/{invalidate,refresh}``."""

    path: str | None = Field(
        default=None,
        description="Drop this entry only (and its descendants if a directory).",
    )
    scope: str = Field(
        default="all",
        description="When ``path`` is null: 'all' drops everything; 'indices' drops navigation-index entries only.",
    )


class CacheControlResponse(BaseModel):
    dropped: int = Field(..., description="Number of cache entries removed")
    warnings: list[str] = Field(
        default_factory=list,
        description="Per-node warnings raised by the post-invalidate refresh (refresh endpoint only).",
    )


def _require_cached_fs(workspace) -> CachedRemoteFileSystem:  # noqa: ANN001
    fs = getattr(workspace, "_fs", None)
    if not isinstance(fs, CachedRemoteFileSystem):
        raise HTTPException(
            status_code=409,
            detail="Active workspace has no cache (local workspaces are not cached).",
        )
    return fs


@router.post("/cache/invalidate", response_model=CacheControlResponse)
def invalidate_workspace_cache(
    request: CacheControlRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> CacheControlResponse:
    """Drop cached entries from the active workspace's mirror.

    ``scope="indices"`` is the "I added a run on the remote, refresh
    navigation" knob — it drops only entries whose basename identifies
    a navigation-index file, leaving log/blob bytes intact.
    """
    fs = _require_cached_fs(workspace)
    dropped = fs.invalidate(request.path, scope=request.scope)
    return CacheControlResponse(dropped=dropped, warnings=[])


@router.post("/cache/refresh", response_model=CacheControlResponse)
def refresh_workspace_cache(
    request: CacheControlRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> CacheControlResponse:
    """Invalidate, then walk the navigation indices again.

    Saves the UI from issuing a follow-up call after a refresh button
    click.  Per-node failures during the walk surface as ``warnings`` —
    the response is still 200 so a single bad project does not blank
    the whole tree.
    """
    fs = _require_cached_fs(workspace)
    dropped = fs.invalidate(request.path, scope=request.scope)
    warnings = prefetch_workspace_indices(workspace)
    return CacheControlResponse(
        dropped=dropped,
        warnings=[f"{w.path}: {w.reason}" for w in warnings],
    )


# ── Guarded curation (deterministic, LLM-free) — curate-unify-03 ──────────────


class CurateRequest(BaseModel):
    """A structured, LLM-free destructive-curation request.

    Builds a §8 ``ChangeProposal`` directly from typed args and drives it through
    the shared ``run_curation_proposal`` backend (the same one the CLI + NL flow
    use). ``approve`` defaults to ``False`` so a destructive mutation over HTTP
    never auto-executes — the proposal is recorded and refused unless the caller
    opts in.
    """

    op: Literal["move_run", "delete_folder", "rehome_asset"]
    run: str | None = None
    target_experiment: str | None = None
    folder: str | None = None
    asset: str | None = None
    source: dict[str, str] | None = None
    target: dict[str, str] | None = None
    action: str = "copy"
    approve: bool = False
    project: str = "curations"
    experiment: str = "curate"


class CurateResponse(BaseModel):
    """The gated-execution outcome for a deterministic curation request."""

    proposalId: str
    status: str
    reason: str | None = None
    resultArtifactIds: list[str] = Field(default_factory=list)


async def _curate_reject_approver(request: ApprovalRequest) -> ApprovalDecision:
    from datetime import UTC, datetime

    from molexp.harness.schemas import ApprovalDecision

    return ApprovalDecision(
        request_id=request.id,
        granted=False,
        decided_by="http-operator",
        decided_at=datetime.now(tz=UTC),
        reason="approve=false",
    )


async def _curate_grant_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Grant carried by the HTTP request body's explicit ``approve: true``.

    An explicit per-request decision by the HTTP caller — NOT a silent
    default — so ``decided_by`` names the caller, never "auto-approver".
    """
    from datetime import UTC, datetime

    from molexp.harness.schemas import ApprovalDecision

    return ApprovalDecision(
        request_id=request.id,
        granted=True,
        decided_by="http-operator",
        decided_at=datetime.now(tz=UTC),
        reason="approve=true (explicit in the request body)",
    )


@router.post("/curate", response_model=CurateResponse)
async def curate_workspace(
    request: CurateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> CurateResponse:
    """Gate + execute one deterministic destructive-curation op (single stack).

    Shares the ``run_curation_proposal`` backend with ``molexp curate`` (Python ≡
    UI). ``approve=false`` (default) records the proposal and refuses; ``true``
    executes the mutation. Either way the §8 ``change_proposal`` artifact is the audit.
    """
    from molexp.services.curate_runtime import build_curation_proposal, run_curation_proposal
    from molexp.workspace.utils import derive_run_id

    try:
        proposal = build_curation_proposal(
            request.op,
            run=request.run,
            target_experiment=request.target_experiment,
            folder=request.folder,
            asset=request.asset,
            source=request.source,
            target=request.target,
            action=request.action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    params: dict[str, Any] = {"mode": "curate-propose", "op": request.op, "proposal": proposal.id}
    audit_run = (
        workspace.add_project(request.project)
        .add_experiment(request.experiment)
        .add_run(params, id=derive_run_id(params))
    )
    approver = _curate_grant_approver if request.approve else _curate_reject_approver
    result = await run_curation_proposal(
        proposal, workspace=workspace, run=audit_run, approve=approver
    )
    outcome = result.execution_result
    return CurateResponse(
        proposalId=proposal.id,
        status=outcome.status if outcome is not None else "failed",
        reason=outcome.reason if outcome is not None else None,
        resultArtifactIds=list(outcome.result_artifact_ids) if outcome is not None else [],
    )
