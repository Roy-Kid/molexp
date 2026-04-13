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
