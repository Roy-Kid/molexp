"""Tests for the /executions route consuming workflow IR.

When the agent posts an :class:`ExecutionCreateRequest` carrying
``workflow_json``, the server should bind it to the experiment (if no
workflow is bound yet) and persist it to disk before creating the run.
"""

from __future__ import annotations


def _ir() -> dict:
    return {
        "workflow_id": "workflow_00000000",
        "name": "exec_route_ir",
        "task_configs": [
            {
                "task_id": "k",
                "task_type": "core.constant",
                "config": {"value": 5},
                "status": "pending",
            },
            {
                "task_id": "doubled",
                "task_type": "core.multiply",
                "config": {"factor": 2.0},
                "status": "pending",
            },
        ],
        "links": [
            {"source": "k", "target": "doubled", "mapping": {}, "status": "pending"},
        ],
        "metadata": {"label": None, "description": None, "tags": [], "custom": {}},
    }


class TestExecutionIRRoute:
    def test_post_without_workflow_json_still_works(self, client, project, experiment):
        resp = client.post(
            "/api/executions",
            json={
                "project_id": project.id,
                "experiment_id": experiment.id,
                "parameters": {"lr": 1e-4},
            },
        )
        assert resp.status_code == 200
        # No workflow was bound and none was persisted
        assert experiment.workflow is None
        assert not (experiment.experiment_dir / "workflow.json").exists()

    def test_post_with_workflow_json_binds_and_persists(self, client, project, experiment):
        resp = client.post(
            "/api/executions",
            json={
                "project_id": project.id,
                "experiment_id": experiment.id,
                "parameters": {},
                "workflow_json": _ir(),
            },
        )
        assert resp.status_code == 200, resp.json()

        # Re-fetch the experiment via the project so we test the lazy-load
        # path, not the same instance we asserted on already.
        reloaded = project.get_experiment(experiment.id)
        assert reloaded is not None
        assert reloaded.workflow is not None
        assert reloaded.workflow.name == "exec_route_ir"
        assert (reloaded.experiment_dir / "workflow.json").exists()

    def test_second_post_does_not_overwrite_bound_workflow(self, client, project, experiment):
        first = _ir()
        client.post(
            "/api/executions",
            json={
                "project_id": project.id,
                "experiment_id": experiment.id,
                "parameters": {},
                "workflow_json": first,
            },
        )
        # Second POST sends a different IR; current behavior is to keep
        # the first binding (the on-disk IR wins).
        second = _ir()
        second["name"] = "different"
        resp = client.post(
            "/api/executions",
            json={
                "project_id": project.id,
                "experiment_id": experiment.id,
                "parameters": {},
                "workflow_json": second,
            },
        )
        assert resp.status_code == 200
        reloaded = project.get_experiment(experiment.id)
        assert reloaded.workflow.name == "exec_route_ir"  # first wins

    def test_unknown_project_404(self, client, experiment):
        resp = client.post(
            "/api/executions",
            json={
                "project_id": "nonexistent",
                "experiment_id": experiment.id,
                "parameters": {},
                "workflow_json": _ir(),
            },
        )
        assert resp.status_code in (404, 422)
