"""HTTP auth routes + gate (filesystem users)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.services.auth import (
    AuthService,
    reset_auth_service,
    set_auth_enabled,
    set_auth_root,
)
from molexp.workspace import Workspace


@pytest.fixture
def auth_root(tmp_path: Path):
    root = tmp_path / "auth"
    set_auth_root(root)
    set_auth_enabled(False)
    yield root
    reset_auth_service()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(root=tmp_path / "ws", name="Test")


@pytest.fixture
def client(workspace: Workspace, auth_root: Path) -> TestClient:
    del auth_root
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)


def _seed_admin(password: str = "secret") -> None:
    AuthService().bootstrap_admin(password, username="admin")


class TestAuthOff:
    def test_projects_open_without_login(self, client: TestClient) -> None:
        resp = client.get("/api/projects")
        assert resp.status_code == 200

    def test_health_reports_auth_not_required(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["auth_required"] is False

    def test_status_disabled(self, client: TestClient) -> None:
        resp = client.get("/api/auth/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["authenticated"] is False


class TestAuthOn:
    def test_gate_requires_login(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        resp = client.get("/api/projects")
        assert resp.status_code == 401

    def test_login_then_access(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        login = client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
        assert login.status_code == 200
        assert login.json()["username"] == "admin"
        assert login.json()["role"] == "admin"
        # Cookie set → subsequent requests authenticated
        resp = client.get("/api/projects")
        assert resp.status_code == 200

    def test_health_reports_auth_required(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        resp = client.get("/api/health")
        assert resp.json()["auth_required"] is True

    def test_viewer_cannot_post(self, client: TestClient) -> None:
        svc = AuthService()
        svc.bootstrap_admin("secret")
        svc.create_user("view", "vpass", role="viewer")
        set_auth_enabled(True)
        login = client.post("/api/auth/login", json={"username": "view", "password": "vpass"})
        assert login.status_code == 200
        # GET ok
        assert client.get("/api/projects").status_code == 200
        # POST forbidden
        resp = client.post("/api/projects", json={"name": "nope"})
        assert resp.status_code == 403

    def test_logout_clears_access(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
        assert client.get("/api/projects").status_code == 200
        client.post("/api/auth/logout")
        assert client.get("/api/projects").status_code == 401

    def test_admin_can_list_users(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
        resp = client.get("/api/auth/users")
        assert resp.status_code == 200
        names = [u["username"] for u in resp.json()["users"]]
        assert "admin" in names

    def test_bearer_token_works(self, client: TestClient) -> None:
        _seed_admin()
        set_auth_enabled(True)
        login = client.post("/api/auth/login", json={"username": "admin", "password": "secret"})
        assert login.status_code == 200
        token_resp = client.get("/api/auth/token")
        assert token_resp.status_code == 200
        token = token_resp.json()["token"]
        # Fresh client without cookies
        app = client.app
        bare = TestClient(app)
        denied = bare.get("/api/projects")
        assert denied.status_code == 401
        ok = bare.get("/api/projects", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200


class TestWorkspaceAllowlist:
    def test_list_filters_by_allowlist(
        self, client: TestClient, workspace: Workspace, tmp_path: Path
    ) -> None:
        from molexp.server.dependencies import ServedWorkspace, set_served_workspaces

        other = Workspace(root=tmp_path / "other", name="Other")
        set_served_workspaces(
            [
                ServedWorkspace(
                    key="ws-a",
                    label=str(workspace.root),
                    is_remote=False,
                    path=str(workspace.root),
                ),
                ServedWorkspace(
                    key="ws-b",
                    label=str(other.root),
                    is_remote=False,
                    path=str(other.root),
                ),
            ]
        )
        svc = AuthService()
        svc.bootstrap_admin("secret")
        svc.create_user("bob", "bobpass", role="operator", workspaces=["ws-a"])
        set_auth_enabled(True)
        client.post("/api/auth/login", json={"username": "bob", "password": "bobpass"})
        resp = client.get("/api/workspaces")
        assert resp.status_code == 200
        keys = [row["key"] for row in resp.json()]
        assert keys == ["ws-a"]
