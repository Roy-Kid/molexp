"""Agent-session mount scope on create (vision-loop-11, RED set).

Session creation (``POST /api/agent-tasks``) accepts an optional mount scope
(``projectId`` / ``experimentId`` / ``runId``). A valid scope is persisted on
the agent-task metadata (``services/agent_task_store.PersistedAgentTask`` scope
fields) so follow-up messages rebuild the same mount; an unresolvable scope is
a 404 (mirroring the services builder's typed ``*NotFoundError``) and leaves
zero residue.

Bound loosely on purpose: the runner-factory seam may grow a context/scope
parameter, so the scripted factory below tolerates any extra arguments and the
assertions ride the route + task store only. The context-block *content*
contract lives in ``tests/test_services/test_agent_context.py``.

Spec: ``.claude/specs/vision-loop-11-mount-context.md`` (Design §3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

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
from molexp.services.agent_task_store import (
    list_agent_task_metadata,
    read_agent_task_metadata,
)
from molexp.workspace import Run, Workspace


class _ScriptedRouter:
    """A fake Router replaying a trivial reasoning-free turn."""

    async def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[object, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        yield TextDeltaChunk(text="hello")
        yield FinalChunk(text="hello")

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text="hello")

    async def complete_structured(self, **_: object) -> object:
        raise AssertionError("not used")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _scripted_factory(workspace: object, *args: object, **kwargs: object) -> AgentRunner:
    """Runner factory tolerant of a grown seam signature (scope/context args)."""
    root = getattr(workspace, "root", None)
    config = InteractiveLoopConfig(workspace_root=Path(str(root)) if root else None)
    return AgentRunner(loop=InteractiveLoop(config=config), router=_ScriptedRouter())  # type: ignore[arg-type]


@pytest.fixture
def agent_client(workspace: Workspace) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    agent_routes.set_runner_factory(_scripted_factory)
    with TestClient(app) as client:  # lifespan runs → clean background-turn teardown
        yield client
    agent_routes.reset_runner_factory()


def _scoped_payload(run: Run) -> dict[str, str]:
    return {
        "description": "inspect the mounted run",
        "projectId": "test-project",
        "experimentId": "test-exp",
        "runId": run.id,
    }


# ── Valid scope → 200 + persisted scope ─────────────────────────────────────


def test_create_with_run_scope_returns_200(
    agent_client: TestClient, workspace: Workspace, run: Run
) -> None:
    resp = agent_client.post("/api/agent-tasks", json=_scoped_payload(run))
    assert resp.status_code == 200, resp.text
    assert resp.json()["sessionId"]


def test_run_scope_persists_on_task_metadata(
    agent_client: TestClient, workspace: Workspace, run: Run
) -> None:
    resp = agent_client.post("/api/agent-tasks", json=_scoped_payload(run))
    assert resp.status_code == 200, resp.text

    task_id = resp.json()["taskId"]
    persisted = read_agent_task_metadata(str(workspace.root), task_id)
    assert persisted is not None
    assert persisted.project_id == "test-project"
    assert persisted.experiment_id == "test-exp"
    assert persisted.run_id == run.id


def test_experiment_scope_persists_without_run_id(
    agent_client: TestClient, workspace: Workspace, experiment: object
) -> None:
    resp = agent_client.post(
        "/api/agent-tasks",
        json={
            "description": "inspect the experiment",
            "projectId": "test-project",
            "experimentId": "test-exp",
        },
    )
    assert resp.status_code == 200, resp.text

    persisted = read_agent_task_metadata(str(workspace.root), resp.json()["taskId"])
    assert persisted is not None
    assert persisted.project_id == "test-project"
    assert persisted.experiment_id == "test-exp"
    assert persisted.run_id is None


def test_unscoped_create_still_works_and_persists_no_scope(
    agent_client: TestClient, workspace: Workspace
) -> None:
    resp = agent_client.post("/api/agent-tasks", json={"description": "no mount"})
    assert resp.status_code == 200, resp.text

    persisted = read_agent_task_metadata(str(workspace.root), resp.json()["taskId"])
    assert persisted is not None
    assert persisted.project_id is None
    assert persisted.experiment_id is None
    assert persisted.run_id is None


# ── Invalid scope → 404, zero residue ────────────────────────────────────────


def test_unknown_run_scope_returns_404(
    agent_client: TestClient, workspace: Workspace, experiment: object
) -> None:
    resp = agent_client.post(
        "/api/agent-tasks",
        json={
            "description": "mount a ghost run",
            "projectId": "test-project",
            "experimentId": "test-exp",
            "runId": "deadbeef",
        },
    )
    assert resp.status_code == 404, resp.text


def test_unknown_project_scope_returns_404(agent_client: TestClient, workspace: Workspace) -> None:
    resp = agent_client.post(
        "/api/agent-tasks",
        json={"description": "mount a ghost project", "projectId": "no-such-project"},
    )
    assert resp.status_code == 404, resp.text


def test_invalid_scope_persists_no_task(
    agent_client: TestClient, workspace: Workspace, experiment: object
) -> None:
    resp = agent_client.post(
        "/api/agent-tasks",
        json={
            "description": "mount a ghost run",
            "projectId": "test-project",
            "experimentId": "test-exp",
            "runId": "deadbeef",
        },
    )
    assert resp.status_code == 404, resp.text
    assert list_agent_task_metadata(str(workspace.root)) == []


# ── Storage mount + CLI/server parity (Design §3/§4) ─────────────────────────


def test_route_passes_run_dir_as_session_anchor(
    agent_client: TestClient, workspace: Workspace, run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route resolves the scope to the run dir and hands it to the runner."""
    captured: dict[str, object] = {}
    original = agent_routes._build_runner

    def _spy(
        workspace: object, context_block: str = "", session_anchor: Path | None = None
    ) -> AgentRunner:
        captured["anchor"] = session_anchor
        return original(workspace, context_block, session_anchor)

    monkeypatch.setattr(agent_routes, "_build_runner", _spy)
    resp = agent_client.post("/api/agent-tasks", json=_scoped_payload(run))
    assert resp.status_code == 200, resp.text
    assert captured["anchor"] == Path(str(run.run_dir))


def test_session_anchor_mounts_session_folder_under_run(run: Run) -> None:
    """A runner with a session anchor materializes the session inside the run dir."""

    class _NamedLoop:
        name = "default"

        async def run(self, **_: object) -> None:  # pragma: no cover — never driven
            raise AssertionError("not driven")

    run_dir = Path(str(run.run_dir))
    runner = AgentRunner(
        loop=_NamedLoop(),  # type: ignore[arg-type]
        router=_ScriptedRouter(),  # type: ignore[arg-type]
        workspace=run_dir.parents[1],
        session_anchor=run_dir,
    )
    runner.session("mounted-session")
    hits = list(run_dir.rglob("*mounted-session*"))
    assert hits, f"no session folder under {run_dir}"


def test_mount_context_parity(workspace: Workspace, run: Run) -> None:
    """CLI path and server path render byte-identical blocks (ac-004)."""
    from molexp.services.agent_context import build_mount_context

    cli_block = build_mount_context(
        workspace, project_id="test-project", experiment_id="test-exp", run_id=run.id
    )
    server_block, server_anchor = agent_routes._mount_context(
        workspace, project_id="test-project", experiment_id="test-exp", run_id=run.id
    )
    assert server_block == cli_block
    assert server_anchor == Path(str(run.run_dir))


def test_list_and_get_do_not_clobber_persisted_scope(
    agent_client: TestClient, workspace: Workspace, run: Run
) -> None:
    """Read paths refresh live-session state without wiping the stored scope."""
    resp = agent_client.post("/api/agent-tasks", json=_scoped_payload(run))
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["taskId"]

    assert agent_client.get("/api/agent-tasks").status_code == 200
    assert agent_client.get(f"/api/agent-tasks/{task_id}").status_code == 200

    persisted = read_agent_task_metadata(str(workspace.root), task_id)
    assert persisted is not None
    assert persisted.project_id == "test-project"
    assert persisted.experiment_id == "test-exp"
    assert persisted.run_id == run.id


def test_parentless_scope_id_is_rejected_not_downgraded(
    agent_client: TestClient, workspace: Workspace, run: Run
) -> None:
    """An experiment/run id without its parents is an error, never 'unscoped'."""
    resp = agent_client.post(
        "/api/agent-tasks",
        json={"description": "orphan scope", "experimentId": "test-exp"},
    )
    assert resp.status_code == 404, resp.text
    assert list_agent_task_metadata(str(workspace.root)) == []
