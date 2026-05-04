"""Tests for /api/reviews."""

from __future__ import annotations

import pytest

from molexp.server.routes import agent as agent_route
from molexp.server.routes.review_store import PersistedReviewItem, write_review_metadata


@pytest.fixture(autouse=True)
def _clean_agent_state():
    agent_route._sessions.clear()
    agent_route._live_sessions.clear()
    yield
    agent_route._sessions.clear()
    agent_route._live_sessions.clear()


@pytest.mark.integration
def test_list_reviews_filters_by_status(client, workspace):
    write_review_metadata(
        workspace.root,
        PersistedReviewItem(
            review_id="review-a",
            kind="plan",
            title="Plan A",
            status="pending",
            target_type="plan",
            target_id="plan-a",
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    write_review_metadata(
        workspace.root,
        PersistedReviewItem(
            review_id="review-b",
            kind="plan",
            title="Plan B",
            status="approved",
            target_type="plan",
            target_id="plan-b",
            created_at="2026-01-02T00:00:00Z",
        ),
    )

    body = client.get("/api/reviews", params={"status": "pending"}).json()

    assert body["total"] == 1
    assert body["reviews"][0]["id"] == "review-a"


@pytest.mark.integration
def test_approve_plan_review_resolves_live_plan(client, workspace):
    captured = {}

    class _Live:
        async def respond_plan(
            self,
            request_id: str,
            approved: bool,
            edited_plan=None,
            edited_workflow_ir=None,
            feedback: str = "",
        ) -> None:
            captured["request_id"] = request_id
            captured["approved"] = approved
            captured["feedback"] = feedback

    agent_route._sessions["sess-plan"] = agent_route.AgentSessionResponse(
        sessionId="sess-plan",
        status="running",
        goalDescription="Draft workflow",
        createdAt="2026-01-01T00:00:00Z",
        events=[],
        stats=agent_route.SessionStatsResponse(),
    )
    agent_route._live_sessions["sess-plan"] = _Live()
    write_review_metadata(
        workspace.root,
        PersistedReviewItem(
            review_id="review-plan-task-1-plan-1",
            kind="plan",
            title="Plan",
            status="pending",
            target_type="plan",
            target_id="plan-1",
            task_id="task-1",
            session_id="sess-plan",
            created_at="2026-01-01T00:01:00Z",
        ),
    )

    res = client.post("/api/reviews/review-plan-task-1-plan-1/approve", json={"comment": "ok"})

    assert res.status_code == 200
    assert captured == {"request_id": "plan-1", "approved": True, "feedback": "ok"}
    review = client.get("/api/reviews/review-plan-task-1-plan-1").json()
    assert review["status"] == "approved"
    assert review["resolutionComment"] == "ok"
