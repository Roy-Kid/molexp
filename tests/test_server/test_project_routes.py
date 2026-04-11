"""Tests for project API routes."""



class TestProjectRoutes:
    def test_list_empty(self, client):
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create(self, client):
        resp = client.post("/api/projects", json={"name": "QM9"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "QM9"
        assert data["id"] == "qm9"
        assert "created" in data

    def test_get(self, client, project):
        resp = client.get(f"/api/projects/{project.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == project.id

    def test_get_not_found(self, client):
        resp = client.get("/api/projects/nonexistent")
        assert resp.status_code == 404

    def test_delete(self, client, project):
        resp = client.delete(f"/api/projects/{project.id}")
        assert resp.status_code == 200
        resp2 = client.get(f"/api/projects/{project.id}")
        assert resp2.status_code == 404

    def test_list_after_create(self, client):
        client.post("/api/projects", json={"name": "A"})
        client.post("/api/projects", json={"name": "B"})
        resp = client.get("/api/projects")
        assert len(resp.json()) == 2
