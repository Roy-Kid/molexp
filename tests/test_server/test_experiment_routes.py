"""Tests for experiment API routes."""


class TestExperimentRoutes:
    def test_list_empty(self, client, project):
        resp = client.get(f"/api/projects/{project.id}/experiments")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_with_workflow(self, client, project):
        resp = client.post(
            f"/api/projects/{project.id}/experiments",
            json={
                "name": "baseline",
                "workflow_source": "train.py",
                "parameter_space": {"lr": [1e-4]},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "baseline"
        assert data["workflow"] == "train.py"
        assert data["parameterSpace"] == {"lr": [1e-4]}

    def test_get(self, client, project, experiment):
        resp = client.get(f"/api/projects/{project.id}/experiments/{experiment.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == experiment.id
        assert data["projectId"] == project.id
        assert data["workflow"] == "train.py"

    def test_delete(self, client, project, experiment):
        resp = client.delete(f"/api/projects/{project.id}/experiments/{experiment.id}")
        assert resp.status_code == 200

    def test_create_with_default_target_succeeds(self, client, project):
        client.post(
            "/api/targets",
            json={"name": "hpc", "scratchRoot": "/tmp/x", "scheduler": "local"},
        )
        resp = client.post(
            f"/api/projects/{project.id}/experiments",
            json={
                "name": "with-target",
                "workflow_source": "train.py",
                "defaultTarget": "hpc",
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["defaultTarget"] == "hpc"

    def test_create_with_unknown_target_returns_422(self, client, project):
        resp = client.post(
            f"/api/projects/{project.id}/experiments",
            json={
                "name": "ghost",
                "workflow_source": "train.py",
                "defaultTarget": "no-such",
            },
        )
        assert resp.status_code == 422
        assert "no-such" in resp.json()["detail"]


class TestRunCreationWithTarget:
    def test_create_run_with_known_target(
        self, client, project, experiment_with_entrypoint
    ):
        experiment = experiment_with_entrypoint
        client.post(
            "/api/targets",
            json={"name": "hpc", "scratchRoot": "/tmp/x", "scheduler": "local"},
        )
        resp = client.post(
            f"/api/projects/{project.id}/experiments/{experiment.id}/runs",
            json={"parameters": {"lr": 1e-4}, "target": "hpc"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["target"] == "hpc"

    def test_create_run_with_unknown_target_returns_422(
        self, client, project, experiment
    ):
        resp = client.post(
            f"/api/projects/{project.id}/experiments/{experiment.id}/runs",
            json={"parameters": {}, "target": "ghost"},
        )
        assert resp.status_code == 422

    def test_create_run_inherits_experiment_default_target(self, client, project):
        client.post(
            "/api/targets",
            json={"name": "hpc", "scratchRoot": "/tmp/x", "scheduler": "local"},
        )
        exp_resp = client.post(
            f"/api/projects/{project.id}/experiments",
            json={
                "name": "inh",
                "workflow_source": "train.py",
                "defaultTarget": "hpc",
            },
        )
        exp_id = exp_resp.json()["id"]
        run_resp = client.post(
            f"/api/projects/{project.id}/experiments/{exp_id}/runs",
            json={"parameters": {}},
        )
        assert run_resp.status_code == 201
        assert run_resp.json()["target"] == "hpc"
