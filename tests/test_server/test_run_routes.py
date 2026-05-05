"""Tests for run API routes."""

import pytest

from molexp.plugins.submit_molq.submit import SubmitHandler
from molexp.workspace import ComputeTarget, add_target


class TestRunRoutes:
    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_create(self, client, project, experiment):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {"lr": 1e-4}},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["parameters"] == {"lr": 1e-4}
        assert data["status"] == "pending"
        assert data["projectId"] == project.id
        assert data["experimentId"] == experiment.id

    def test_create_captures_workflow_snapshot(self, client, project, experiment):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}},
        )
        data = resp.json()
        assert data["workflow"] is not None
        assert data["workflow"]["source"] == "train.py"

    def test_list(self, client, project, experiment):
        client.post(self._prefix(project, experiment), json={"parameters": {}})
        client.post(self._prefix(project, experiment), json={"parameters": {}})
        resp = client.get(self._prefix(project, experiment))
        assert len(resp.json()) == 2

    def test_get(self, client, project, experiment, run):
        run._update_metadata(executor_info={"backend": "molq", "scheduler": "slurm"})
        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == run.id
        assert resp.json()["executorInfo"] == {"backend": "molq", "scheduler": "slurm"}

    def test_update_status(self, client, project, experiment, run):
        resp = client.patch(
            f"{self._prefix(project, experiment)}/{run.id}/status",
            json={"status": "succeeded"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "succeeded"
        assert data["finished"] is not None

    def test_get_exposes_results_and_history(self, client, project, experiment, run):
        # Drive a real execution so context.results and execution_history are populated.
        with run.start() as ctx:
            ctx.set_result("y", 9.0)

        resp = client.get(f"{self._prefix(project, experiment)}/{run.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == {"y": 9.0}
        assert data["workflowSource"] == "train.py"
        assert len(data["executionHistory"]) == 1
        record = data["executionHistory"][0]
        assert record["executionId"].startswith("exec-")
        assert record["status"] == "succeeded"
        assert record["startedAt"] is not None
        assert record["finishedAt"] is not None

    def test_list_summary_includes_results(self, client, project, experiment, run):
        with run.start() as ctx:
            ctx.set_result("y", 9.0)

        resp = client.get(
            f"/api/projects/{project.id}/experiments/{experiment.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"], "experiment should report at least one run"
        summary = next(r for r in data["runs"] if r["id"] == run.id)
        assert summary["results"] == {"y": 9.0}
        assert summary["finished"] is not None


@pytest.fixture
def local_target(workspace):
    add_target(
        workspace,
        ComputeTarget(name="laptop", scratch_root=str(workspace.root / "scratch")),
    )
    return "laptop"


@pytest.fixture
def captured_submits(monkeypatch):
    """Recorder for ``SubmitHandler.__call__``; each entry is ``(handler, args)``."""
    calls: list = []
    monkeypatch.setattr(
        SubmitHandler,
        "__call__",
        lambda self, *args, **kwargs: calls.append((self, args)),
    )
    return calls


class TestRunSubmissionWiring:
    """The API path must dispatch to molq when a target is given."""

    def _prefix(self, project, experiment):
        return f"/api/projects/{project.id}/experiments/{experiment.id}/runs"

    def test_create_with_target_invokes_submit_handler(
        self,
        client,
        project,
        experiment_with_entrypoint,
        local_target,
        captured_submits,
    ):
        resp = client.post(
            self._prefix(project, experiment_with_entrypoint),
            json={"parameters": {"lr": 1e-4}, "target": local_target},
        )
        assert resp.status_code == 201, resp.text
        assert len(captured_submits) == 1
        handler, args = captured_submits[0]
        assert handler._scheduler == "local"
        _script, mol_run, exp, proj = args
        assert mol_run.id == resp.json()["id"]
        assert exp.id == experiment_with_entrypoint.id
        assert proj.id == project.id

    def test_create_without_target_skips_submit(
        self, client, project, experiment, captured_submits
    ):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}},
        )
        assert resp.status_code == 201
        assert captured_submits == []

    def test_create_with_target_without_entrypoint_returns_422(
        self, client, project, experiment, local_target
    ):
        resp = client.post(
            self._prefix(project, experiment),
            json={"parameters": {}, "target": local_target},
        )
        assert resp.status_code == 422
        assert "entrypoint" in resp.json()["detail"].lower()

    def test_rerun_inherits_target_and_dispatches(
        self,
        client,
        project,
        experiment_with_entrypoint,
        local_target,
        captured_submits,
    ):
        src_run = experiment_with_entrypoint.run(parameters={"lr": 1e-4}, target=local_target)
        captured_submits.clear()  # ignore implicit submit on source-run creation

        resp = client.post(
            f"{self._prefix(project, experiment_with_entrypoint)}/{src_run.id}/rerun"
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["sourceRunId"] == src_run.id
        assert body["newRunId"] != src_run.id
        assert len(captured_submits) == 1

    def test_rerun_without_target_does_not_submit(
        self, client, project, experiment, run, captured_submits
    ):
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/rerun")
        assert resp.status_code == 201
        assert captured_submits == []

    def test_kill_routes_through_try_cancel(self, client, project, experiment, run, monkeypatch):
        seen: list = []

        def fake_try_cancel(r):
            seen.append(r.id)
            r.cancel()
            return None

        monkeypatch.setattr("molexp.server.routes.run.try_cancel", fake_try_cancel)
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/kill")
        assert resp.status_code == 200
        assert seen == [run.id]
        assert resp.json()["message"] == "Run cancelled"
        assert resp.json()["status"] == "cancelled"

    def test_kill_falls_back_when_try_cancel_warns(
        self, client, project, experiment, run, monkeypatch
    ):
        monkeypatch.setattr(
            "molexp.server.routes.run.try_cancel",
            lambda r: "no molq job id",
        )
        resp = client.post(f"{self._prefix(project, experiment)}/{run.id}/kill")
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "no molq job id"
        assert body["status"] == "cancelled"
