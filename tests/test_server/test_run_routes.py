"""Tests for run API routes."""


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
