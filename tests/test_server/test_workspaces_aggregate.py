"""Aggregate workspace surface — ``/api/workspaces/{ws}/…`` namespaced routes.

Covers the serve-multi-remote-workspaces spec phase 3 acceptance criteria:

- ac-004: two local workspaces each with a same-named project resolve distinctly
- ac-005: a remote workspace is served read-through its fs
- ac-006: a mutating request to a remote workspace is rejected (405)
- ac-007: an unreachable remote is flagged in /api/workspaces (and 502s on data)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import molexp.server.deps.targets as targets_deps
from molexp.server.app import create_app
from molexp.server.dependencies import (
    ServedWorkspace,
    get_workspace_target_registry,
    reset_workspace_cache,
    reset_workspace_target_registry,
    set_active_workspace_descriptor,
    set_served_workspaces,
    set_workspace_path_override,
)
from molexp.server.workspace_targets import WorkspaceTarget, WorkspaceTargetRegistry
from molexp.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    set_served_workspaces([])
    reset_workspace_target_registry()
    registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
    monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)
    yield
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    set_served_workspaces([])
    reset_workspace_target_registry()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(serve_static=False))


def _local_ws_with_project(root: Path, project: str) -> str:
    root.mkdir(parents=True, exist_ok=True)
    Workspace(root).add_project(project)
    return str(root)


# ── ac-004: two local workspaces, same-named project, no collision ──────────


def test_namespaced_routes_resolve_distinct_workspaces(client: TestClient, tmp_path: Path):
    a = _local_ws_with_project(tmp_path / "a", "matrix")
    b = _local_ws_with_project(tmp_path / "b", "matrix")
    set_served_workspaces(
        [
            ServedWorkspace(key="local-a", label=a, is_remote=False, path=a),
            ServedWorkspace(key="local-b", label=b, is_remote=False, path=b),
        ]
    )
    set_workspace_path_override(Path(a))  # active = a (flat back-compat)

    ra = client.get("/api/workspaces/local-a/projects")
    rb = client.get("/api/workspaces/local-b/projects")
    assert ra.status_code == 200 and rb.status_code == 200
    assert [p["id"] for p in ra.json()] == ["matrix"]
    assert [p["id"] for p in rb.json()] == ["matrix"]
    # distinct on-disk roots → distinct created timestamps, no shared instance
    assert ra.json()[0]["created"] != rb.json()[0]["created"]

    # flat routes still address the active workspace (a)
    flat = client.get("/api/projects")
    assert flat.status_code == 200
    assert flat.json()[0]["created"] == ra.json()[0]["created"]

    # the served list marks which workspace the flat routes / tree address
    listing = {w["key"]: w["active"] for w in client.get("/api/workspaces").json()}
    assert listing == {"local-a": True, "local-b": False}


def test_unknown_workspace_is_404(client: TestClient, tmp_path: Path):
    a = _local_ws_with_project(tmp_path / "a", "matrix")
    set_served_workspaces([ServedWorkspace(key="local-a", label=a, is_remote=False, path=a)])
    r = client.get("/api/workspaces/nope/projects")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "UNKNOWN_WORKSPACE"


# ── ac-005 / ac-006: remote read-through + write rejection ──────────────────


def _stub_reachable_remote(monkeypatch: pytest.MonkeyPatch):
    """Make remote targets resolve to a real local FileSystem (read-through)."""
    from molexp.workspace.fs_local import LocalFileSystem

    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        lambda _t: LocalFileSystem(),
    )


def test_remote_workspace_served_read_through(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    remote_root = tmp_path / "remote"
    _local_ws_with_project(remote_root, "prod")
    get_workspace_target_registry().add(
        WorkspaceTarget(name="hpc", host="h", root_path=str(remote_root))
    )
    _stub_reachable_remote(monkeypatch)
    set_served_workspaces(
        [ServedWorkspace(key="hpc", label="me@h:/remote", is_remote=True, target_name="hpc")]
    )

    r = client.get("/api/workspaces/hpc/projects")
    assert r.status_code == 200, r.text
    assert [p["id"] for p in r.json()] == ["prod"]


def test_remote_workspace_rejects_writes(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    remote_root = tmp_path / "remote"
    _local_ws_with_project(remote_root, "prod")
    get_workspace_target_registry().add(
        WorkspaceTarget(name="hpc", host="h", root_path=str(remote_root))
    )
    _stub_reachable_remote(monkeypatch)
    set_served_workspaces(
        [ServedWorkspace(key="hpc", label="me@h:/remote", is_remote=True, target_name="hpc")]
    )

    # any mutating method on a scoped remote route is rejected before the body
    r = client.post("/api/workspaces/hpc/executions", json={})
    assert r.status_code == 405, r.text
    assert r.json()["error"]["code"] == "REMOTE_WORKSPACE_READ_ONLY"


# ── ac-007: unreachable remote degrades gracefully ──────────────────────────


def _stub_unreachable_remote(monkeypatch: pytest.MonkeyPatch):
    def _boom(_t):
        raise ConnectionError("ssh: connect to host h port 22: timed out")

    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        _boom,
    )


def test_unreachable_remote_flagged_in_list_and_502_on_data(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    get_workspace_target_registry().add(WorkspaceTarget(name="hpc", host="h", root_path="/remote"))
    _stub_unreachable_remote(monkeypatch)
    set_served_workspaces(
        [ServedWorkspace(key="hpc", label="me@h:/remote", is_remote=True, target_name="hpc")]
    )

    listing = client.get("/api/workspaces")
    assert listing.status_code == 200
    row = listing.json()[0]
    assert row["key"] == "hpc"
    assert row["unreachable"] is True  # still listed, flagged

    data = client.get("/api/workspaces/hpc/projects")
    assert data.status_code == 502
    assert data.json()["error"]["code"] == "REMOTE_WORKSPACE_UNREACHABLE"
