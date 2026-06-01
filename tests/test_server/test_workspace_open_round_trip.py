"""Local↔remote round-trip integration for the active-workspace switch.

This is the binding end-to-end test for sub-spec 02: it switches the active
workspace local → remote → local through the HTTP route and asserts that
each phase returns the descriptor's own state (different project lists),
proving the cache rekey + descriptor-aware get_workspace branching wires
together correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import (
    get_workspace_target_registry,
    reset_workspace_cache,
    reset_workspace_target_registry,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)
from molexp.server.workspace_targets import (
    WorkspaceTarget,
    WorkspaceTargetRegistry,
)
from molexp.workspace import Workspace
from molexp.workspace.fs_local import LocalFileSystem


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import molexp.server.dependencies as deps

    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    reset_workspace_target_registry()
    registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
    monkeypatch.setattr(deps, "_workspace_target_registry", registry)
    yield
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    reset_workspace_target_registry()


@pytest.mark.integration
def test_local_to_remote_to_local_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Open local → list projects → switch remote → list projects → switch
    back; each phase exposes its own project set."""
    # Build LOCAL workspace with one project "p-local"
    local_root = tmp_path / "local_ws"
    local_ws = Workspace(root=local_root, name="local")
    local_ws.materialize()
    local_ws.add_project("p-local")
    # Build REMOTE root (still a local fs in test) with project "p-remote"
    remote_root = tmp_path / "remote_ws"
    remote_ws = Workspace(root=remote_root, name="remote")
    remote_ws.materialize()
    remote_ws.add_project("p-remote")

    # Register a remote descriptor whose FS factory returns LocalFileSystem.
    registry = get_workspace_target_registry()
    registry.add(WorkspaceTarget(name="lab", host="me@example.org", root_path=str(remote_root)))
    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        lambda _t: LocalFileSystem(),
    )

    client = TestClient(create_app())

    # Open local
    resp = client.post(
        "/api/workspace/open",
        json={"kind": "local", "path": str(local_root), "create_if_missing": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["projectCount"] == 1
    # Verify the project listed via the workspace API is "p-local"
    list_resp = client.get("/api/projects")
    assert list_resp.status_code == 200
    body = list_resp.json()
    rows = body.get("projects", body) if isinstance(body, dict) else body
    names = [p["name"] for p in rows]
    assert "p-local" in names

    # Switch to remote
    resp = client.post("/api/workspace/open", json={"kind": "remote", "name": "lab"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["projectCount"] == 1
    list_resp = client.get("/api/projects")
    assert list_resp.status_code == 200
    body = list_resp.json()
    rows = body.get("projects", body) if isinstance(body, dict) else body
    names = [p["name"] for p in rows]
    assert "p-remote" in names
    assert "p-local" not in names, names

    # Switch back to local
    resp = client.post(
        "/api/workspace/open",
        json={"kind": "local", "path": str(local_root)},
    )
    assert resp.status_code == 200, resp.text
    list_resp = client.get("/api/projects")
    body = list_resp.json()
    rows = body.get("projects", body) if isinstance(body, dict) else body
    names = [p["name"] for p in rows]
    assert "p-local" in names
    assert "p-remote" not in names, names
