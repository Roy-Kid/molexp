"""Remote Operations API for the molq plugin.

Read-only endpoints powering the workspace-level Remote Operations overview.
All routes are mounted under ``/api/plugins/molq``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from molq import JobNotFoundError

from molexp.plugins.submit_molq import dashboard

from ..schemas import (
    MolqJobDetailResponse,
    MolqJobsResponse,
    MolqTargetListResponse,
    MolqTargetSummary,
)

router = APIRouter(prefix="/plugins/molq", tags=["molq"])

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/targets", response_model=MolqTargetListResponse)
def list_targets() -> MolqTargetListResponse:
    """List configured molq targets (one per profile in ``~/.molq/config.toml``)."""
    try:
        targets = dashboard.list_targets()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    summaries = [MolqTargetSummary.from_dataclass(t) for t in targets]
    return MolqTargetListResponse(targets=summaries, total=len(summaries))


@router.get("/jobs", response_model=MolqJobsResponse)
def list_jobs(
    target: str | None = Query(default=None, description="Profile name to filter by."),
    include_terminal: bool = Query(default=True, alias="includeTerminal"),
    limit: int = Query(default=200, ge=1, le=1000),
) -> MolqJobsResponse:
    """List jobs across one or all targets, plus aggregate queue stats."""
    try:
        page = dashboard.fetch_page(
            target,
            include_terminal=include_terminal,
            limit=limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MolqJobsResponse.from_page(page)


@router.get("/jobs/{job_id}", response_model=MolqJobDetailResponse)
def get_job(
    job_id: str,
    target: str = Query(..., description="Profile name owning the job."),
) -> MolqJobDetailResponse:
    """Return a single job's detail including transitions and dependency state."""
    try:
        detail = dashboard.get_job(target, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MolqJobDetailResponse.from_dataclass(detail)


@router.get("/jobs/{job_id}/logs")
async def stream_logs(
    job_id: str,
    target: str = Query(..., description="Profile name owning the job."),
    stream: str = Query(default="stdout", pattern="^(stdout|stderr)$"),
) -> StreamingResponse:
    """SSE stream of newline-terminated log chunks.

    Each event payload is ``data: {"line": "..."}\\n\\n`` so the client's
    EventSource ``message`` handler parses one log line per event.
    """
    try:
        # Fail fast: invalid target / unknown job becomes a 404 instead of an
        # SSE stream that opens then closes empty (harder to debug client-side).
        dashboard.get_job(target, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _generate() -> AsyncGenerator[str, None]:
        async for line in dashboard.tail_log(target, job_id, stream=stream):
            payload = json.dumps({"line": line})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
