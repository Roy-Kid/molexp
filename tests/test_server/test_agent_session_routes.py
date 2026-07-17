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
from molexp.server.routes import agent_tasks as agent_task_routes


class _ScriptedRouter:
    """A fake Router replaying a trivial reasoning-free turn."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
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


def test_delete_task_removes_it_from_task_surface(agent_client: TestClient) -> None:
    created = agent_client.post("/api/agent-tasks", json={"description": "temporary task"})
    assert created.status_code == 200, created.text
    task_id = created.json()["taskId"]

    deleted = agent_client.delete(f"/api/agent-tasks/{task_id}")
    assert deleted.status_code == 200, deleted.text

    listed = agent_client.get("/api/agent-tasks").json()
    assert all(t["taskId"] != task_id for t in listed["tasks"])
    assert agent_client.get(f"/api/agent-tasks/{task_id}").status_code == 404


def test_cancel_task_is_idempotent_when_idle(agent_client: TestClient) -> None:
    """POST …/cancel succeeds even after the turn has already finished."""
    created = agent_client.post("/api/agent-tasks", json={"description": "quick"})
    assert created.status_code == 200, created.text
    task_id = created.json()["taskId"]

    # Scripted router finishes almost immediately; cancel must still 200.
    cancelled = agent_client.post(f"/api/agent-tasks/{task_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    again = agent_client.post(f"/api/agent-tasks/{task_id}/cancel")
    assert again.status_code == 200, again.text


def test_followup_message_stays_on_same_task(agent_client: TestClient) -> None:
    """POST …/messages continues the same session — does not mint a new taskId."""
    created = agent_client.post("/api/agent-tasks", json={"description": "first turn"})
    assert created.status_code == 200, created.text
    body = created.json()
    task_id = body["taskId"]
    session_id = body["sessionId"]

    # Wait for the first turn to settle so 409 doesn't fire.
    import time

    for _ in range(50):
        status = agent_client.get(f"/api/agent-tasks/{task_id}").json()["status"]
        if status != "running":
            break
        time.sleep(0.05)

    msg = agent_client.post(
        f"/api/agent-tasks/{task_id}/messages",
        json={"content": "follow-up on the same session"},
    )
    assert msg.status_code == 200, msg.text

    listed = agent_client.get("/api/agent-tasks").json()["tasks"]
    # Still one product task for this session — no duplicate task id.
    same = [t for t in listed if t["sessionId"] == session_id]
    assert len(same) == 1
    assert same[0]["taskId"] == task_id


def test_task_metadata_lives_under_interactive_home(
    agent_client: TestClient, workspace: object
) -> None:
    """Task metadata is colocated under agent/_tasks/."""
    created = agent_client.post("/api/agent-tasks", json={"description": "where on disk?"})
    assert created.status_code == 200, created.text
    task_id = created.json()["taskId"]
    root = Path(str(workspace.root))  # type: ignore[attr-defined]
    meta = root / "agent" / "_tasks" / task_id / "metadata.json"
    assert meta.is_file(), f"expected {meta}"


def test_plan_mode_is_same_task_and_asks_for_missing_experiment(
    agent_client: TestClient,
) -> None:
    created = agent_client.post(
        "/api/agent-tasks",
        json={"description": "plan a conductivity study", "mode": "plan"},
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["activeMode"] == "plan"
    assert body["status"] == "awaiting_user"
    assert any(event["type"] == "clarification_required" for event in body["events"])

    listed = agent_client.get("/api/agent-tasks").json()["tasks"]
    assert [task["taskId"] for task in listed] == [body["taskId"]]


def test_plan_context_reply_starts_plan_turn_on_same_task(
    agent_client: TestClient,
    workspace: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = workspace.add_project("Battery Project")  # type: ignore[attr-defined]
    experiment = project.add_experiment("Conductivity")
    launched: list[tuple[str, str, str]] = []

    def _capture_launch(*, workspace, task, draft, turn_id):
        launched.append((task.task_id, draft, turn_id))

    monkeypatch.setattr(agent_task_routes, "_launch_plan_turn", _capture_launch)
    created = agent_client.post(
        "/api/agent-tasks",
        json={"description": "plan a conductivity study", "mode": "plan"},
    ).json()

    replied = agent_client.post(
        f"/api/agent-tasks/{created['taskId']}/messages",
        json={"content": f"{project.id} / {experiment.id}", "mode": "chat"},
    )
    assert replied.status_code == 200, replied.text
    assert launched and launched[0][0] == created["taskId"]

    current = agent_client.get(f"/api/agent-tasks/{created['taskId']}").json()
    assert current["taskId"] == created["taskId"]
    assert current["activeMode"] == "plan"
    assert current["status"] == "running"


def test_chat_task_can_switch_to_plan_and_ask_for_context(agent_client: TestClient) -> None:
    created = agent_client.post(
        "/api/agent-tasks", json={"description": "inspect the workspace", "mode": "chat"}
    ).json()

    import time

    for _ in range(50):
        current = agent_client.get(f"/api/agent-tasks/{created['taskId']}").json()
        if current["status"] != "running":
            break
        time.sleep(0.05)

    switched = agent_client.post(
        f"/api/agent-tasks/{created['taskId']}/messages",
        json={"content": "now build the nine-step plan", "mode": "plan"},
    )
    assert switched.status_code == 200, switched.text
    current = agent_client.get(f"/api/agent-tasks/{created['taskId']}").json()
    assert current["taskId"] == created["taskId"]
    assert current["activeMode"] == "plan"
    assert current["status"] == "awaiting_user"
    assert any(event["type"] == "clarification_required" for event in current["events"])


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
