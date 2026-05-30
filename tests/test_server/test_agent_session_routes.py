"""Relit agent session routes over the runtime registry (spec 00a).

Drives the real ``/api/agent-tasks`` HTTP surface (which delegates to the
relit ``routes/agent.py`` functions) with a scripted-Router runner factory so
no LLM is constructed. The client is used as a context manager so the app
lifespan runs — its shutdown cancels every background turn cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from molexp.agent.loops.interactive import InteractiveLoop, InteractiveLoopConfig
from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
)
from molexp.agent.runner import AgentRunner
from molexp.agent.types import UsageBreakdown
from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.server.routes import agent as agent_routes


class _ScriptedRouter:
    """A fake Router replaying a trivial reasoning-free turn."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="hello")
        yield FinalChunk(text="hello")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text="hello")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("not used")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _scripted_factory(workspace: object) -> AgentRunner:
    root = getattr(workspace, "root", None)
    config = InteractiveLoopConfig(workspace_root=Path(str(root)) if root else None)
    return AgentRunner(loop=InteractiveLoop(config=config), router=_ScriptedRouter())  # type: ignore[arg-type]


@pytest.fixture
def agent_client(workspace: object) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    agent_routes.set_runner_factory(_scripted_factory)
    with TestClient(app) as client:  # context-manager → lifespan runs (clean turn teardown)
        yield client
    agent_routes.reset_runner_factory()


def test_create_session_returns_200_and_is_listed(agent_client: TestClient) -> None:
    resp = agent_client.post("/api/agent-tasks", json={"description": "inspect the project"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["goal"] == "inspect the project"
    assert body["sessionId"]

    listed = agent_client.get("/api/agent-tasks").json()
    assert any(t["sessionId"] == body["sessionId"] for t in listed["tasks"])

    got = agent_client.get(f"/api/agent-tasks/{body['taskId']}")
    assert got.status_code == 200
    assert got.json()["sessionId"] == body["sessionId"]


def test_unknown_task_returns_404(agent_client: TestClient) -> None:
    assert agent_client.get("/api/agent-tasks/task-does-not-exist").status_code == 404


def test_missing_model_config_yields_503(
    workspace: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    # no runner factory installed + no configured model → 503 pre-flight
    agent_routes.reset_runner_factory()
    monkeypatch.setattr(agent_routes, "_configured_model", lambda: None)
    with TestClient(app) as client:
        resp = client.post("/api/agent-tasks", json={"description": "hi"})
    assert resp.status_code == 503
    assert "model" in resp.json()["detail"].lower()
