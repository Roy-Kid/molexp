"""Review queue routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_workspace
from ..schemas import (
    MessageResponse,
    PlanDecisionRequest,
    ReviewDecisionRequest,
    ReviewItemResponse,
    ReviewListResponse,
    ReviewTargetRefResponse,
)
from . import agent as agent_routes
from .review_store import (
    PersistedReviewItem,
    list_review_metadata,
    read_review_metadata,
    resolve_review,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


def _workspace_root(workspace) -> str:
    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    return str(root)


def _to_response(item: PersistedReviewItem) -> ReviewItemResponse:
    return ReviewItemResponse(
        id=item.review_id,
        kind=item.kind,
        title=item.title,
        description=item.description,
        riskLevel=item.risk_level,
        status=item.status,
        targetRef=ReviewTargetRefResponse(
            type=item.target_type,
            id=item.target_id,
            taskId=item.task_id,
            sessionId=item.session_id,
        ),
        createdAt=item.created_at,
        resolvedAt=item.resolved_at,
        resolutionComment=item.resolution_comment,
        metadata=item.metadata,
    )


def _get_review_or_404(workspace, review_id: str) -> PersistedReviewItem:
    item = read_review_metadata(_workspace_root(workspace), review_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return item


@router.get("", response_model=ReviewListResponse)
def list_reviews(
    status: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    workspace=Depends(get_workspace),
) -> ReviewListResponse:
    """List persisted review items."""
    rows = list_review_metadata(_workspace_root(workspace))
    if status is not None:
        rows = [row for row in rows if row.status == status]
    if kind is not None:
        rows = [row for row in rows if row.kind == kind]
    return ReviewListResponse(reviews=[_to_response(row) for row in rows], total=len(rows))


@router.get("/{review_id}", response_model=ReviewItemResponse)
def get_review(
    review_id: str,
    workspace=Depends(get_workspace),
) -> ReviewItemResponse:
    """Get one persisted review item."""
    return _to_response(_get_review_or_404(workspace, review_id))


@router.post("/{review_id}/approve", response_model=MessageResponse)
async def approve_review(
    review_id: str,
    request: ReviewDecisionRequest,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Approve a review item and apply its target decision."""
    item = _get_review_or_404(workspace, review_id)
    if item.status != "pending":
        raise HTTPException(status_code=409, detail=f"Review {review_id} is already resolved")
    if item.kind == "plan":
        if not item.session_id:
            raise HTTPException(status_code=409, detail="Plan review has no runtime session")
        await agent_routes.respond_plan(
            item.session_id,
            PlanDecisionRequest(
                request_id=item.target_id,
                approved=True,
                edited_plan=request.edited_plan,
                edited_workflow_ir=request.edited_workflow_ir,
                feedback=request.comment,
            ),
            workspace=workspace,
        )
    resolve_review(_workspace_root(workspace), item, status="approved", comment=request.comment)
    return MessageResponse(message="approved")


@router.post("/{review_id}/reject", response_model=MessageResponse)
async def reject_review(
    review_id: str,
    request: ReviewDecisionRequest,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Reject a review item and notify its target when possible."""
    item = _get_review_or_404(workspace, review_id)
    if item.status != "pending":
        raise HTTPException(status_code=409, detail=f"Review {review_id} is already resolved")
    if item.kind == "plan":
        if not item.session_id:
            raise HTTPException(status_code=409, detail="Plan review has no runtime session")
        await agent_routes.respond_plan(
            item.session_id,
            PlanDecisionRequest(
                request_id=item.target_id,
                approved=False,
                feedback=request.comment,
            ),
            workspace=workspace,
        )
    resolve_review(_workspace_root(workspace), item, status="rejected", comment=request.comment)
    return MessageResponse(message="rejected")
