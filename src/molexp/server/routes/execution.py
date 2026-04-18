"""Execution routes for MolExp API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from molexp.workflow.cache import Caching

from ..dependencies import get_workspace
from ..exceptions import ExperimentNotFoundError, ProjectNotFoundError
from ..schemas import CacheClearResponse, CacheStatsResponse, ExecutionCreateRequest, RunResponse

router = APIRouter(prefix="", tags=["execution"])


@router.post("/executions", response_model=RunResponse)
def create_execution(
    request: ExecutionCreateRequest,
    workspace=Depends(get_workspace),
) -> RunResponse:
    """Create a new execution in a specific project/experiment."""
    project = workspace.get_project(request.project_id)
    if not project:
        raise ProjectNotFoundError(request.project_id)

    experiment = project.get_experiment(request.experiment_id)
    if not experiment:
        raise ExperimentNotFoundError(request.project_id, request.experiment_id)

    new_run = experiment.run(parameters=request.parameters)
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
_cache_instance = None


def _get_cache():
    global _cache_instance
    if _cache_instance is None:
        store_dir = Path.home() / ".molexp" / "cache"
        _cache_instance = Caching(store_dir=store_dir)
        _cache_instance.initialize()
    return _cache_instance


@router.get("/cache/stats", response_model=CacheStatsResponse)
def get_cache_stats() -> CacheStatsResponse:
    cache = _get_cache()
    entry_count = 0
    if cache._store_dir.exists():
        entry_count = len(list(cache._store_dir.glob("*.json")))
    return CacheStatsResponse(storeDir=str(cache._store_dir), entryCount=entry_count)


@router.delete("/cache", response_model=CacheClearResponse)
def clear_cache() -> CacheClearResponse:
    cache = _get_cache()
    removed = cache.clear()
    return CacheClearResponse(removedCount=removed)
