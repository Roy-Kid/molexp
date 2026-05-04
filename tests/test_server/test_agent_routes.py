"""Tests for the FastAPI /api/agent routes after the AgentService cutover.

Phase 2 routes are thin translators between FastAPI schemas and
:class:`molexp.agent.AgentService`. These tests exercise that
translation against either the live in-process service or a stubbed
:class:`AgentService` injected via the FastAPI dependency override.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.agent import AgentMode, AgentService, Goal, SessionStatus
from molexp.agent.orchestration import (
    PlanCreated,
    SessionStarted,
    UserMessageRequested,
)
from molexp.agent.testing import FakeModelClient
from molexp.server.routes import agent as agent_route


@pytest.fixture(autouse=True)
def _clean_service_cache():
    """Drop the per-workspace AgentService cache between tests."""

    agent_route.reset_agent_service_cache()
    yield
    agent_route.reset_agent_service_cache()


# ── Event serializer ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_serialize_event_renders_harness_dataclasses() -> None:
    event = SessionStarted(session_id="s1", goal_description="hi")
    wire = agent_route._serialize_event(event)
    assert wire.type == "SessionStarted"
    assert wire.payload["session_id"] == "s1"
    assert wire.payload["goal_description"] == "hi"

    req = UserMessageRequested(request_id="r1", prompt="more?")
    wire = agent_route._serialize_event(req)
    assert wire.type == "UserMessageRequested"
    assert wire.payload["request_id"] == "r1"
    assert wire.payload["prompt"] == "more?"


# ── /sessions ↔ AgentService translation ────────────────────────────────────


@pytest.mark.integration
def test_post_message_unknown_session_404s(client) -> None:
    res = client.post("/api/agent/sessions/missing/messages", json={"content": "hi"})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_post_message_routes_to_live_session(client, workspace) -> None:
    """Sending a message via the route must reach the AgentService session."""

    service = agent_route.get_agent_service(workspace=workspace)
    session = service.start_session(Goal(description="hi"))

    res = client.post(
        f"/api/agent/sessions/{session.session_id}/messages",
        json={"content": "scope=project", "request_id": "req-7"},
    )
    assert res.status_code == 200
    assert res.json() == {"message": "queued"}

    inbound = await asyncio.wait_for(session.next_inbound(), timeout=1.0)
    assert inbound is not None
    assert inbound.content == "scope=project"
    assert inbound.request_id == "req-7"


@pytest.mark.integration
def test_list_sessions_includes_persisted(client, workspace) -> None:
    """Sessions persisted to .molexp-agent/sessions/ surface in the list."""

    service = agent_route.get_agent_service(workspace=workspace)
    s1 = service.start_session(Goal(description="first goal"))
    s2 = service.start_session(Goal(description="second goal"))

    res = client.get("/api/agent/sessions")
    assert res.status_code == 200
    body = res.json()
    ids = {s["sessionId"] for s in body["sessions"]}
    assert s1.session_id in ids
    assert s2.session_id in ids


@pytest.mark.integration
def test_get_session_falls_back_to_disk(client, workspace) -> None:
    """Even after dropping the live handle, ``GET /sessions/{id}`` finds disk metadata."""

    service = agent_route.get_agent_service(workspace=workspace)
    session = service.start_session(Goal(description="historical goal"))
    sid = session.session_id

    # Drop live handles to force the disk fallback path.
    service._sessions.clear()

    res = client.get(f"/api/agent/sessions/{sid}")
    assert res.status_code == 200
    body = res.json()
    assert body["sessionId"] == sid
    assert body["goalDescription"] == "historical goal"


@pytest.mark.integration
def test_get_session_404s_when_unknown(client) -> None:
    res = client.get("/api/agent/sessions/no-such-session")
    assert res.status_code == 404


# ── Plan-decision route ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_decision_route_resumes_parked_session(client, workspace) -> None:
    """POST /plan-decision must drive the runner past the parked plan state."""

    model = FakeModelClient()
    plan_text = (
        "Step 1: x\n"
        '```json\n{"workflow_ir": {"task_configs": [{"task_id": "t1"}]}}\n```'
    )
    model.queue_text(plan_text)
    model.queue_text("plan executed")

    # Build an isolated AgentService backed by FakeModelClient and inject it.
    isolated = AgentService.from_workspace(
        workspace.root, model=model, workspace=workspace
    )
    agent_route._service_cache[str(workspace.root)] = isolated

    session = isolated.start_session(
        Goal(description="please plan", mode=AgentMode.PLAN)
    )

    plan_created: list[PlanCreated] = []

    async def watch() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                plan_created.append(ev)
                break

    await asyncio.wait_for(watch(), timeout=2.0)
    request_id = plan_created[0].request_id

    res = client.post(
        f"/api/agent/sessions/{session.session_id}/plan-decision",
        json={"request_id": request_id, "approved": True},
    )
    assert res.status_code == 200
    assert res.json() == {"message": "approved"}
    await isolated.cancel(session.session_id)


@pytest.mark.integration
def test_plan_decision_returns_409_on_stale_request_id(client, workspace) -> None:
    service = agent_route.get_agent_service(workspace=workspace)
    session = service.start_session(Goal(description="not in plan mode"))

    res = client.post(
        f"/api/agent/sessions/{session.session_id}/plan-decision",
        json={"request_id": "stale", "approved": True},
    )
    assert res.status_code == 409


# ── Approval route ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_respond_approval_404s_unknown_session(client) -> None:
    res = client.post(
        "/api/agent/sessions/missing/approve",
        json={"request_id": "r1", "approved": True},
    )
    assert res.status_code == 404
