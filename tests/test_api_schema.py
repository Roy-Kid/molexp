from __future__ import annotations

from typing import Generator

from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.workspace import Workspace


def build_workspace(tmp_path) -> tuple[Workspace, str, str, str]:
    workspace = Workspace(root=tmp_path, name="Test Workspace")
    workspace.materialize()

    project = workspace.create_project(name="Project Alpha")
    experiment = project.create_experiment(name="Experiment One")
    run = experiment.create_run(parameters={"alpha": 1})

    return workspace, project.id, experiment.id, run.id


def build_client(workspace: Workspace) -> Generator[TestClient, None, None]:
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_project_experiment_run_schema_ids(tmp_path) -> None:
    workspace, project_id, experiment_id, run_id = build_workspace(tmp_path)

    for client in build_client(workspace):
        projects_response = client.get("/api/projects")
        assert projects_response.status_code == 200
        projects = projects_response.json()
        assert len(projects) == 1
        project = projects[0]
        assert project["id"] == project_id
        assert project["projectId"] == project_id

        experiment_list_response = client.get(f"/api/projects/{project_id}/experiments")
        assert experiment_list_response.status_code == 200
        experiments = experiment_list_response.json()
        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment["id"] == experiment_id
        assert experiment["experimentId"] == experiment_id
        assert experiment["projectId"] == project_id

        experiment_detail_response = client.get(
            f"/api/projects/{project_id}/experiments/{experiment_id}"
        )
        assert experiment_detail_response.status_code == 200
        experiment_detail = experiment_detail_response.json()
        assert experiment_detail["projectId"] == project_id
        assert experiment_detail["experimentId"] == experiment_id

        run_list_response = client.get(
            f"/api/projects/{project_id}/experiments/{experiment_id}/runs"
        )
        assert run_list_response.status_code == 200
        runs = run_list_response.json()
        assert len(runs) == 1
        run_item = runs[0]
        assert run_item["id"] == run_id
        assert run_item["runId"] == run_id
        assert run_item["projectId"] == project_id
        assert run_item["experimentId"] == experiment_id

        run_detail_response = client.get(
            f"/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}"
        )
        assert run_detail_response.status_code == 200
        run_detail = run_detail_response.json()
        assert run_detail["id"] == run_id
        assert run_detail["runId"] == run_id
        assert run_detail["projectId"] == project_id
        assert run_detail["experimentId"] == experiment_id
