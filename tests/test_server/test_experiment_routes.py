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
        resp = client.get(
            f"/api/projects/{project.id}/experiments/{experiment.id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == experiment.id
        assert data["projectId"] == project.id
        assert data["workflow"] == "train.py"

    def test_delete(self, client, project, experiment):
        resp = client.delete(
            f"/api/projects/{project.id}/experiments/{experiment.id}"
        )
        assert resp.status_code == 200
