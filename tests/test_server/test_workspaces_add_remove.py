"""Runtime served-set add/remove (VS Code multi-root explorer)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.dependencies import (
    reset_workspace_cache,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)
from molexp.server.deps.served import (
    ServedWorkspace,
    get_served_workspaces,
    set_served_workspaces,
)


@pytest.fixture
def client(tmp_path: Path):
    """App client with one local served workspace."""
    from molexp.server.app import create_app
    from molexp.workspace import Workspace

    root_a = tmp_path / "ws-a"
    Workspace(root_a).materialize()
    set_served_workspaces(
        [
            ServedWorkspace(
                key="local-ws-a",
                label=str(root_a.resolve()),
                is_remote=False,
                path=str(root_a.resolve()),
            )
        ]
    )
    # Path override alone (descriptor is mutually exclusive — do not clear after).
    set_workspace_path_override(root_a.resolve())

    app = create_app()
    with TestClient(app) as c:
        yield c, root_a.resolve(), tmp_path

    set_served_workspaces([])
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    reset_workspace_cache()


@pytest.mark.unit
class TestWorkspacesAddRemove:
    def test_list_includes_served(self, client):
        c, root_a, _ = client
        rows = c.get("/api/workspaces").json()
        assert len(rows) == 1
        assert rows[0]["key"] == "local-ws-a"
        assert rows[0]["active"] is True
        assert rows[0]["path"] == str(root_a)

    def test_add_local_folder(self, client):
        c, _root_a, tmp_path = client
        root_b = tmp_path / "ws-b"
        root_b.mkdir()
        # bare dir without materialize — still addable
        resp = c.post(
            "/api/workspaces/add",
            json={"kind": "local", "path": str(root_b), "activate": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["isRemote"] is False
        assert body["path"] == str(root_b.resolve())
        assert body["active"] is True

        rows = c.get("/api/workspaces").json()
        assert len(rows) == 2
        keys = {r["key"] for r in rows}
        assert "local-ws-a" in keys
        assert body["key"] in keys

    def test_add_local_dedupes_same_path(self, client):
        c, root_a, _ = client
        resp = c.post(
            "/api/workspaces/add",
            json={"kind": "local", "path": str(root_a), "activate": False},
        )
        assert resp.status_code == 200
        assert resp.json()["key"] == "local-ws-a"
        assert len(get_served_workspaces()) == 1

    def test_add_missing_path_404(self, client):
        c, _, tmp_path = client
        missing = tmp_path / "no-such-dir"
        resp = c.post(
            "/api/workspaces/add",
            json={"kind": "local", "path": str(missing), "create_if_missing": False},
        )
        assert resp.status_code == 404

    def test_add_create_if_missing(self, client):
        c, _, tmp_path = client
        new_path = tmp_path / "brand-new"
        resp = c.post(
            "/api/workspaces/add",
            json={
                "kind": "local",
                "path": str(new_path),
                "create_if_missing": True,
                "activate": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert new_path.is_dir()
        assert (new_path / "workspace.json").exists() or (new_path / "meta.json").exists()

    def test_remove_folder_switches_active(self, client):
        c, _root_a, tmp_path = client
        root_b = tmp_path / "ws-b"
        root_b.mkdir()
        add = c.post(
            "/api/workspaces/add",
            json={"kind": "local", "path": str(root_b), "activate": True},
        )
        key_b = add.json()["key"]
        assert add.json()["active"] is True

        rm = c.delete(f"/api/workspaces/{key_b}")
        assert rm.status_code == 200, rm.text
        rows = rm.json()
        assert len(rows) == 1
        assert rows[0]["key"] == "local-ws-a"
        assert rows[0]["active"] is True

    def test_remove_unknown_404(self, client):
        c, _, _ = client
        resp = c.delete("/api/workspaces/does-not-exist")
        assert resp.status_code == 404
