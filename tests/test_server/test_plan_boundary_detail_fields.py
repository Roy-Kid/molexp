"""Boundary 3 — Plan detail API fields (unit; TestClient, no e2e UI).

Pins ``PlanDetailResponse`` carries PlanOrchestrator artifacts and
``GET /api/.../plans/{run_id}`` serves a run that only has ``experiment_plan`` /
``plan_report`` (no legacy ``experiment_report`` required).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.server.routes.plans import PlanDetailResponse
from molexp.workspace import Workspace


@pytest.fixture()
def plan_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "ws", name="lab")
    ws.materialize()
    return ws


@pytest.fixture()
def seeded_plan_ids(plan_workspace: Workspace) -> tuple[str, str, str]:
    project = plan_workspace.add_project("demo")
    exp = project.add_experiment("exp")
    run = exp.add_run(params={"mode": "plan"}, id="plandetail1")

    store = FileArtifactStore(root=run.run_dir / "artifacts")
    store.put_json(
        kind="experiment_plan",
        obj={
            "spec": {"title": "Detail board", "objective": "serve plan API"},
            "board": {
                "version": 1,
                "tasks": [
                    {
                        "id": "t1",
                        "name": "build",
                        "acceptance": ["ok"],
                        "status": "pending",
                        "feasibility": None,
                    }
                ],
            },
        },
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        kind="plan_report",
        obj={"title": "Detail board", "summary_md": "# report"},
        created_by="test",
        parent_ids=[],
    )
    store.put_json(
        kind="frozen_experiment_plan",
        obj={
            "spec": {"title": "Detail board", "objective": "serve plan API"},
            "board": {"version": 1, "tasks": []},
        },
        created_by="test",
        parent_ids=[],
    )
    return project.id, exp.id, run.id


@pytest.fixture()
def plan_client(plan_workspace: Workspace) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: plan_workspace
    return TestClient(app)


class TestPlanDetailResponseSchema:
    def test_model_declares_plan_orchestrator_fields(self) -> None:
        fields = PlanDetailResponse.model_fields
        for name in (
            "experimentPlan",
            "frozenExperimentPlan",
            "planReport",
            "boundWorkflow",
            "interventionRequest",
            "artifactKinds",
        ):
            assert name in fields, f"missing PlanDetailResponse field {name}"


class TestPlanDetailRoute:
    def test_get_plan_without_legacy_experiment_report(
        self,
        plan_client: TestClient,
        seeded_plan_ids: tuple[str, str, str],
    ) -> None:
        project_id, experiment_id, run_id = seeded_plan_ids
        resp = plan_client.get(
            f"/api/projects/{project_id}/experiments/{experiment_id}/plans/{run_id}"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("experimentPlan") is not None
        assert body.get("planReport") is not None
        assert body.get("frozenExperimentPlan") is not None
        assert body.get("title")
        assert isinstance(body.get("tasks"), list)
        assert any(t.get("id") == "t1" for t in body["tasks"])

    def test_list_plans_includes_plan_orchestrator_run(
        self,
        plan_client: TestClient,
        seeded_plan_ids: tuple[str, str, str],
    ) -> None:
        project_id, experiment_id, run_id = seeded_plan_ids
        resp = plan_client.get(f"/api/projects/{project_id}/experiments/{experiment_id}/plans")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        ids = {p["runId"] for p in body.get("plans", [])}
        assert run_id in ids
