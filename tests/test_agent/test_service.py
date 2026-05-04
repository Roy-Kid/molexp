"""Phase 1a: AgentService stub surface (spec §6.3, Decisions O1 + T1)."""

from __future__ import annotations

import pytest

from molexp.agent import AgentService, Goal, SessionStatus, ToolRegistry


def test_service_owns_a_registry_per_decision_t1(workspace_path) -> None:
    a = AgentService.from_workspace(workspace_path / "a")
    b = AgentService.from_workspace(workspace_path / "b")
    assert isinstance(a.registry, ToolRegistry)
    assert isinstance(b.registry, ToolRegistry)
    assert a.registry is not b.registry


def test_start_session_persists_metadata(agent_service: AgentService) -> None:
    session = agent_service.start_session(Goal(description="hello"))
    assert session.session_id
    listed = agent_service.list_sessions()
    assert len(listed) == 1
    assert listed[0].session_id == session.session_id
    assert listed[0].status is SessionStatus.PENDING


def test_get_session_returns_handle(agent_service: AgentService) -> None:
    session = agent_service.start_session(Goal(description="hello"))
    again = agent_service.get_session(session.session_id)
    assert again is session


@pytest.mark.asyncio
async def test_emit_session_started_publishes_to_subscribers(
    agent_service: AgentService,
) -> None:
    from molexp.agent.orchestration.events import SessionStarted

    session = agent_service.start_session(Goal(description="hello"))
    received = []

    async def collect():
        async for event in session.stream_events():
            received.append(event)
            break

    import asyncio

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)  # let subscriber attach
    await agent_service.emit_session_started(session)
    await asyncio.wait_for(task, timeout=1.0)

    assert len(received) == 1
    assert isinstance(received[0], SessionStarted)
    assert received[0].session_id == session.session_id


@pytest.mark.asyncio
async def test_shutdown_marks_live_sessions_interrupted(
    agent_service: AgentService,
) -> None:
    session = agent_service.start_session(Goal(description="hi"))
    await agent_service.shutdown()
    assert session.status is SessionStatus.INTERRUPTED
