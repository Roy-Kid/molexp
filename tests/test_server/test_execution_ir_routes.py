"""Tests for the /executions route consuming workflow IR.

When the agent posts an :class:`ExecutionCreateRequest` carrying
``workflow_json``, the server compiles it into a ``WorkflowSpec`` and
binds it through the workflow-layer's process-local
:func:`molexp.workflow.set_workflow` registry — workspace itself
holds no workflow concept (rectification 2026-05-09).
"""

from __future__ import annotations

from molexp.workflow import get_workflow


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
        # No workflow was bound — registry stays empty for this experiment.
        assert get_workflow(experiment) is None

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

        # The compiled spec should be discoverable via the workflow-layer
        # registry, keyed by experiment.id (the registry is process-local
        # and survives across handler invocations within the test).
        bound = get_workflow(experiment)
        assert bound is not None
        assert bound.name == "exec_route_ir"

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
        # the first binding (the registry refuses to rebind once set).
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
        bound = get_workflow(experiment)
        assert bound is not None
        assert bound.name == "exec_route_ir"  # first wins

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
