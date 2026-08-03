"""Live system-prompt route for agent-tasks (replaces retired /api/agent/sessions/.../system-prompt)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services.agent_task_store import PersistedAgentTask, write_agent_task_metadata
from molexp.workspace import Workspace


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    w = Workspace(root=tmp_path / "ws", name="lab")
    w.materialize()
    return w


@pytest.fixture()
def client(ws: Workspace) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: ws
    return TestClient(app)


def test_legacy_agent_system_prompt_path_is_503(client: TestClient) -> None:
    resp = client.get("/api/agent/sessions/any/system-prompt")
    assert resp.status_code == 503
    assert "retired" in resp.json()["detail"].lower()


def test_agent_tasks_system_prompt_returns_composed_prompt(
    client: TestClient, ws: Workspace
) -> None:
    write_agent_task_metadata(
        str(ws.root),
        PersistedAgentTask(
            task_id="task-sysprompt1",
            session_id="task-sysprompt1",
            title="rg pe",
            goal="create pe chains",
            status="failed",
            created_at="2026-07-27T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
        ),
    )
    resp = client.get("/api/agent-tasks/task-sysprompt1/system-prompt")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("base")
    assert body["planMode"] is True
    assert "PLAN MODE" in body["effective"]
    assert body["effective"].startswith(body["base"][:20])


def test_agent_tasks_system_prompt_404_for_unknown(client: TestClient) -> None:
    resp = client.get("/api/agent-tasks/task-does-not-exist/system-prompt")
    assert resp.status_code == 404
