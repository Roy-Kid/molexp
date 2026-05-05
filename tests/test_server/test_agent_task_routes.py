"""Tests for the user-facing /api/agent-tasks routes."""

from __future__ import annotations

import json

import pytest

from molexp.server.routes import agent as agent_route
from molexp.server.routes.agent_task_store import (
    PersistedAgentTask,
    write_agent_task_metadata,
)


@pytest.fixture(autouse=True)
def _clean_agent_state():
    agent_route.reset_agent_service_cache()
    yield
    agent_route.reset_agent_service_cache()


@pytest.mark.skip(reason="Legacy `sessions/` disk format; revisit after Phase 5 migration (R4).")
@pytest.mark.integration
def test_list_agent_tasks_wraps_persisted_sessions_and_writes_metadata(client, workspace):
    from molexp.plugins.agent_pydanticai.sessions_store import write_session_metadata

    write_session_metadata(
        workspace.root,
        "sess-historical",
        status="completed",
        goal_description="Analyze failed runs",
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:05:00Z",
    )

    res = client.get("/api/agent-tasks")

    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1
    task = body["tasks"][0]
    assert task["taskId"] == "sess-historical"
    assert task["sessionId"] == "sess-historical"
    assert task["title"] == "Analyze failed runs"
    assert task["goal"] == "Analyze failed runs"
    assert task["status"] == "completed"

    persisted = workspace.root / "agent_tasks" / "sess-historical" / "metadata.json"
    assert persisted.exists()
    raw = json.loads(persisted.read_text())
    assert raw["task_id"] == "sess-historical"
    assert raw["session_id"] == "sess-historical"


@pytest.mark.skip(
    reason="Stubbed registry.get path is gone; rewrite to use AgentService + FakeModelClient."
)
@pytest.mark.integration
def test_create_agent_task_starts_session_and_persists_task(client, workspace, monkeypatch):
    from molexp.plugins.agent_pydanticai.types import SessionStats

    from molexp.plugins import registry

    client.put("/api/agent/provider", json={"api_key": "sk-saved"})

    class _StubSession:
        session_id = "sess-created"
        status = "running"
        stats = SessionStats()

    class _StubService:
        @classmethod
        def from_workspace(cls, root):
            return cls()

        async def start(self, goal):
            return _StubSession()

    monkeypatch.setattr(registry, "is_available", lambda cap: True)
    monkeypatch.setattr(registry, "get", lambda cap: _StubService)

    res = client.post("/api/agent-tasks", json={"description": "Create NEMD workflow"})

    assert res.status_code == 200
    body = res.json()
    assert body["taskId"].startswith("task-")
    assert body["sessionId"] == "sess-created"
    assert body["title"] == "Create NEMD workflow"
    assert body["goal"] == "Create NEMD workflow"
    assert body["status"] == "running"
    assert (workspace.root / "agent_tasks" / body["taskId"] / "metadata.json").exists()


@pytest.mark.integration
def test_get_agent_task_falls_back_to_task_metadata_when_session_missing(client, workspace):
    write_agent_task_metadata(
        workspace.root,
        PersistedAgentTask(
            task_id="task-orphan",
            session_id="sess-missing",
            title="Recovered task",
            goal="Recover from task metadata",
            status="blocked",
            created_at="2026-02-01T00:00:00Z",
            updated_at="2026-02-01T00:03:00Z",
            plan_mode=True,
            skill_id="builtin-plan",
        ),
    )

    res = client.get("/api/agent-tasks/task-orphan")

    assert res.status_code == 200
    body = res.json()
    assert body["taskId"] == "task-orphan"
    assert body["sessionId"] == "sess-missing"
    assert body["title"] == "Recovered task"
    assert body["goal"] == "Recover from task metadata"
    assert body["status"] == "blocked"
    assert body["planMode"] is True
    assert body["skillId"] == "builtin-plan"


@pytest.mark.skip(
    reason="Used `_sessions` to inject a session shape; rewrite via AgentService + emitted PlanCreated event."
)
@pytest.mark.integration
def test_agent_task_plan_event_creates_pending_review(client, workspace):
    agent_route._sessions["sess-plan"] = agent_route.AgentSessionResponse(
        sessionId="sess-plan",
        status="running",
        goalDescription="Draft workflow",
        createdAt="2026-03-01T00:00:00Z",
        events=[
            agent_route.SessionEventResponse(
                type="PlanCreatedEvent",
                ts="2026-03-01T00:01:00Z",
                payload={
                    "request_id": "plan-1",
                    "plan_markdown": "1. Inspect workspace\n2. Draft workflow",
                    "workflow_preview": {"workflow_ir": {"task_configs": []}},
                },
            )
        ],
        stats=agent_route.SessionStatsResponse(),
    )

    body = client.get("/api/agent-tasks").json()

    task = body["tasks"][0]
    assert task["taskId"] == "sess-plan"
    assert task["status"] == "waiting_for_review"
    reviews = client.get("/api/reviews", params={"status": "pending"}).json()
    assert reviews["total"] == 1
    review = reviews["reviews"][0]
    assert review["kind"] == "plan"
    assert review["status"] == "pending"
    assert review["targetRef"]["taskId"] == "sess-plan"
    assert review["targetRef"]["sessionId"] == "sess-plan"
