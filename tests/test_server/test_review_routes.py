"""Tests for /api/reviews."""

from __future__ import annotations

import pytest

from molexp.server.routes import agent as agent_route
from molexp.server.routes.review_store import PersistedReviewItem, write_review_metadata


@pytest.fixture(autouse=True)
def _clean_agent_state():
    agent_route.reset_agent_service_cache()
    yield
    agent_route.reset_agent_service_cache()


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
    """The reviews layer routes plan approvals into the live AgentSession.

    Per spec §6.3 (Decision O1) the live session sits on
    :class:`AgentService`; the review layer resolves the plan via the
    same ``respond_plan`` pathway the agent route uses.
    """
    from molexp.agent import AgentMode, Goal
    from molexp.agent.orchestration import PlanStateMachine
    from molexp.workflow import WorkflowPreviewView

    service = agent_route._service_for(workspace)
    session = service.start_session(Goal(description="Draft workflow", mode=AgentMode.PLAN))
    # Park the session on a plan emission so respond_plan has work to do.
    preview = WorkflowPreviewView()
    session.plan = (
        PlanStateMachine()
        .request_plan()
        .emit_plan(request_id="plan-1", plan_markdown="...", preview=preview)
    )
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
            session_id=session.session_id,
            created_at="2026-01-01T00:01:00Z",
        ),
    )

    res = client.post("/api/reviews/review-plan-task-1-plan-1/approve", json={"comment": "ok"})

    assert res.status_code == 200
    # The session state machine should have transitioned to PLAN_APPROVED.
    from molexp.agent.orchestration import PlanState

    assert session.plan.state is PlanState.PLAN_APPROVED
    review = client.get("/api/reviews/review-plan-task-1-plan-1").json()
    assert review["status"] == "approved"
    assert review["resolutionComment"] == "ok"
