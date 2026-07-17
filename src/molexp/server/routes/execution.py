"""Execution routes for MolExp API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from molexp.workflow import (
    Caching,
    default_binding_registry,
    default_codec,
)
from molexp.workspace import (
    ExperimentNotFoundError as WorkspaceExperimentNotFoundError,
)
from molexp.workspace import (
    ProjectNotFoundError as WorkspaceProjectNotFoundError,
)

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError, ProjectNotFoundError
from ..schemas import CacheClearResponse, CacheStatsResponse, ExecutionCreateRequest, RunResponse

router = APIRouter(prefix="", tags=["execution"])


@router.post("/executions", response_model=RunResponse)
def create_execution(
    request: ExecutionCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> RunResponse:
    """Create a new execution in a specific project/experiment.

    If ``request.workflow_json`` is supplied and the experiment has no
    workflow bound, compile and persist the IR before the run is
    materialized so worker processes can pick it up off disk.
    """
    try:
        project = workspace.get_project(request.project_id)
    except WorkspaceProjectNotFoundError:
        raise ProjectNotFoundError(request.project_id) from None

    try:
        experiment = project.get_experiment(request.experiment_id)
    except WorkspaceExperimentNotFoundError:
        raise ExperimentNotFoundError(request.project_id, request.experiment_id) from None

    if (
        request.workflow_json is not None
        and default_binding_registry.for_experiment(experiment) is None
    ):
        # The IR is the durable artifact — compile it here so the bound
        # spec lives in the workflow-layer binding registry, and re-emit it
        # as opaque JSON so the worker can pick it up without re-running
        # the user script.
        spec = default_codec.ir_to_spec(request.workflow_json)
        default_binding_registry.bind(experiment, spec)

    new_run = experiment.add_run(params=request.parameters)
    return RunResponse.from_model(new_run)


@router.post("/plan")
def get_execution_plan() -> JSONResponse:
    """Get execution plan for a workflow (not yet implemented)."""
    return JSONResponse(
        status_code=501,
        content={"error": "not_implemented", "message": "Execution planning not yet implemented."},
    )


# ============================================================================
# Cache Routes
# ============================================================================

# Process-local cache — NOT shared across workers.  Use --workers 1.
_cache_instance: Caching | None = None


def _get_cache() -> Caching:
    global _cache_instance
    if _cache_instance is None:
        # Process-local FileCacheStore under ~/.molexp/cache (not workspace-rooted).
        # Caching no longer exposes initialize() / _store_dir — storage is the
        # pluggable CacheStore and stats() is the public entry-count path.
        store_dir = Path.home() / ".molexp" / "cache"
        store_dir.mkdir(parents=True, exist_ok=True)
        _cache_instance = Caching(store_dir=store_dir)
    return _cache_instance


@router.get("/cache/stats", response_model=CacheStatsResponse)
def get_cache_stats() -> CacheStatsResponse:
    cache = _get_cache()
    stats = cache.stats
    store = cache.store
    store_dir = getattr(store, "store_dir", None)
    return CacheStatsResponse(
        storeDir=str(store_dir) if store_dir is not None else "",
        entryCount=int(stats["entry_count"]),
    )


@router.delete("/cache", response_model=CacheClearResponse)
def clear_cache() -> CacheClearResponse:
    cache = _get_cache()
    removed = cache.clear()
    return CacheClearResponse(removedCount=removed)
