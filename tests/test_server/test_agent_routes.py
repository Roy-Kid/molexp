"""Tests for the FastAPI /api/agent routes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from molexp.server.routes import agent as agent_route


@pytest.fixture(autouse=True)
def _clean_agent_state():
    """Reset the in-memory session stores between tests."""
    agent_route._sessions.clear()
    agent_route._live_sessions.clear()
    yield
    agent_route._sessions.clear()
    agent_route._live_sessions.clear()


@pytest.mark.integration
def test_post_message_unknown_session_404s(client):
    res = client.post("/api/agent/sessions/missing/messages", json={"content": "hi"})
    assert res.status_code == 404


@pytest.mark.integration
def test_post_message_non_interactive_session_409s(client):
    """Sessions without respond_user_message return 409."""
    fake_response = SimpleNamespace()
    agent_route._sessions["sess-stale"] = fake_response  # type: ignore[assignment]
    # No matching live session → live = None → still 409 path.
    res = client.post("/api/agent/sessions/sess-stale/messages", json={"content": "hi"})
    assert res.status_code == 409


@pytest.mark.integration
def test_post_message_routes_to_live_session(client):
    captured = {}

    class _StubSession:
        async def respond_user_message(self, content: str, request_id: str | None = None) -> None:
            captured["content"] = content
            captured["request_id"] = request_id

    agent_route._sessions["sess-1"] = SimpleNamespace()  # type: ignore[assignment]
    agent_route._live_sessions["sess-1"] = _StubSession()

    res = client.post(
        "/api/agent/sessions/sess-1/messages",
        json={"content": "scope=project", "request_id": "req-7"},
    )
    assert res.status_code == 200
    assert res.json() == {"message": "queued"}
    assert captured == {"content": "scope=project", "request_id": "req-7"}


@pytest.mark.integration
def test_serialize_event_renders_known_dataclasses():
    from molexp.plugins.agent_pydanticai.types import (
        ResultArtifactEvent,
        UserMessageRequestEvent,
    )

    event = ResultArtifactEvent(kind="plot", title="t", payload={"data": [1]})
    serialized = agent_route._serialize_event(event)
    assert serialized.type == "ResultArtifactEvent"
    assert serialized.payload["kind"] == "plot"
    assert serialized.payload["title"] == "t"
    assert serialized.payload["payload"] == {"data": [1]}

    req = UserMessageRequestEvent(request_id="r1", prompt="hello?")
    serialized = agent_route._serialize_event(req)
    assert serialized.type == "UserMessageRequestEvent"
    assert serialized.payload["request_id"] == "r1"
    assert serialized.payload["prompt"] == "hello?"


@pytest.mark.integration
def test_drain_live_events_appends_to_response():
    from molexp.plugins.agent_pydanticai.types import UserMessageEvent

    response = SimpleNamespace(events=[], sessionId="x", status="running")
    pending = [UserMessageEvent(content="a"), UserMessageEvent(content="b")]

    class _Live:
        def drain_pending_events(self):
            return pending

    agent_route._drain_live_events(response, _Live())
    assert len(response.events) == 2
    assert response.events[0].payload["content"] == "a"


# ── Pre-flight: refuse to start without credentials ──────────────────────


@pytest.mark.integration
def test_create_session_returns_400_when_provider_unconfigured(client, monkeypatch):
    """No stored key + no env var = 400 with a structured ``code`` field."""
    # Defensive: scrub any developer-set env vars that would mask the gate.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    res = client.post("/api/agent/sessions", json={"description": "ping"})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["code"] == "agent_not_configured"
    assert detail["provider"] == "anthropic"
    assert "ANTHROPIC_API_KEY" in detail["envVar"] or detail["envVar"] == "ANTHROPIC_API_KEY"


@pytest.mark.integration
def test_create_session_passes_when_stored_key_present(client, monkeypatch):
    """A saved API key satisfies the pre-flight even if env vars are missing."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    client.put("/api/agent/provider", json={"api_key": "sk-saved"})

    # Stub out the actual session creation so we don't need a working LLM.
    from molexp.plugins import Capability, registry
    from molexp.plugins.agent_pydanticai.types import SessionStats

    class _StubSession:
        session_id = "sess-stub"
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
    res = client.post("/api/agent/sessions", json={"description": "ping"})
    assert res.status_code == 200
    assert res.json()["sessionId"] == "sess-stub"


# ── Historical sessions surface in list/get ──────────────────────────────


@pytest.mark.integration
def test_list_sessions_includes_persisted(client, workspace):
    """Sessions on disk show up in the list even with the in-memory store empty."""
    from molexp.plugins.agent_pydanticai.sessions_store import write_session_metadata

    write_session_metadata(
        workspace.root,
        "sess-historical-1",
        status="completed",
        goal_description="old goal",
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
    )
    write_session_metadata(
        workspace.root,
        "sess-historical-2",
        status="failed",
        goal_description="another old goal",
        created_at="2026-02-01T00:00:00Z",
    )
    res = client.get("/api/agent/sessions")
    assert res.status_code == 200
    body = res.json()
    ids = {s["sessionId"] for s in body["sessions"]}
    assert {"sess-historical-1", "sess-historical-2"} <= ids
    by_id = {s["sessionId"]: s for s in body["sessions"]}
    assert by_id["sess-historical-1"]["status"] == "completed"
    assert by_id["sess-historical-2"]["status"] == "failed"


@pytest.mark.integration
def test_get_session_falls_back_to_disk(client, workspace):
    from molexp.plugins.agent_pydanticai.sessions_store import write_session_metadata

    write_session_metadata(
        workspace.root,
        "sess-on-disk",
        status="completed",
        goal_description="historical goal",
        created_at="2026-04-01T10:00:00Z",
        completed_at="2026-04-01T10:05:00Z",
    )
    res = client.get("/api/agent/sessions/sess-on-disk")
    assert res.status_code == 200
    body = res.json()
    assert body["sessionId"] == "sess-on-disk"
    assert body["status"] == "completed"
    assert body["goalDescription"] == "historical goal"


@pytest.mark.integration
def test_in_memory_session_overrides_disk_listing(client, workspace):
    """A live session shadows its on-disk metadata so live status wins."""
    from molexp.plugins.agent_pydanticai.sessions_store import write_session_metadata

    write_session_metadata(
        workspace.root,
        "sess-double",
        status="failed",  # stale disk state
        goal_description="goal",
    )
    # Pretend it's currently running in memory.
    agent_route._sessions["sess-double"] = agent_route.AgentSessionResponse(
        sessionId="sess-double",
        status="running",
        goalDescription="goal",
        createdAt="2026-04-28T00:00:00Z",
        events=[],
        stats=agent_route.SessionStatsResponse(),
    )
    body = client.get("/api/agent/sessions").json()
    rows = [s for s in body["sessions"] if s["sessionId"] == "sess-double"]
    assert len(rows) == 1  # not duplicated
    assert rows[0]["status"] == "running"


# ── Health endpoint ──────────────────────────────────────────────────────


@pytest.mark.integration
def test_get_agent_health_reports_unconfigured(client, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    body = client.get("/api/agent/health").json()
    assert body["ready"] is False
    assert body["source"] == "none"
    assert body["envVar"] == "ANTHROPIC_API_KEY"
    assert "Save one in Agent Settings" in body["reason"]


@pytest.mark.integration
def test_get_agent_health_reports_stored_source(client):
    client.put("/api/agent/provider", json={"api_key": "sk-stored"})
    body = client.get("/api/agent/health").json()
    assert body["ready"] is True
    assert body["source"] == "stored"
    assert body["reason"] == ""


@pytest.mark.integration
def test_get_agent_health_reports_env_source(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    body = client.get("/api/agent/health").json()
    assert body["ready"] is True
    assert body["source"] == "env"
