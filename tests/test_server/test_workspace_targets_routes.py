"""Tests for the /api/workspace/targets registry endpoints.

These tests override :func:`get_workspace_target_registry` so the
registry is rooted under ``tmp_path`` rather than the developer's real
``~/.molexp/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import (
    get_remote_fs_factory,
    get_workspace,
    get_workspace_target_registry,
)
from molexp.server.workspace_targets import WorkspaceTargetRegistry
from molexp.workspace import Workspace


@pytest.fixture
def isolated_registry(tmp_path: Path) -> WorkspaceTargetRegistry:
    return WorkspaceTargetRegistry(store_path=tmp_path / "workspace_targets.json")


@pytest.fixture
def client(tmp_path: Path, isolated_registry: WorkspaceTargetRegistry) -> TestClient:
    """TestClient with both the workspace and the workspace-target
    registry redirected to per-test temp storage."""
    ws = Workspace(root=tmp_path / "ws", name="Test")
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: ws
    app.dependency_overrides[get_workspace_target_registry] = lambda: isolated_registry
    return TestClient(app)


def _add_descriptor(client: TestClient, **overrides: object) -> dict:
    payload = {
        "name": "hpc1",
        "host": "me@hpc.example.org",
        "root_path": "/scratch/me/molexp",
    }
    payload.update(overrides)
    resp = client.post("/api/workspace/targets", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── GET ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_list_empty(client: TestClient):
    resp = client.get("/api/workspace/targets")
    assert resp.status_code == 200
    assert resp.json() == {"targets": [], "total": 0}


@pytest.mark.unit
def test_list_after_add(client: TestClient):
    _add_descriptor(client, name="a")
    _add_descriptor(client, name="b")
    body = client.get("/api/workspace/targets").json()
    assert body["total"] == 2
    assert [t["name"] for t in body["targets"]] == ["a", "b"]


# ── POST ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_create_201(client: TestClient):
    body = _add_descriptor(client)
    assert body["name"] == "hpc1"
    assert body["host"] == "me@hpc.example.org"
    assert body["root_path"] == "/scratch/me/molexp"


@pytest.mark.unit
def test_create_duplicate_409(client: TestClient):
    _add_descriptor(client, name="dup")
    resp = client.post(
        "/api/workspace/targets",
        json={
            "name": "dup",
            "host": "other.host",
            "root_path": "/other/path",
        },
    )
    assert resp.status_code == 409
    assert "dup" in resp.json()["detail"]


@pytest.mark.unit
def test_create_invalid_name_422(client: TestClient):
    resp = client.post(
        "/api/workspace/targets",
        json={
            "name": "has space",
            "host": "h",
            "root_path": "/r",
        },
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_create_missing_required_422(client: TestClient):
    resp = client.post(
        "/api/workspace/targets",
        json={"name": "incomplete"},
    )
    assert resp.status_code == 422


# ── DELETE ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_delete_204(client: TestClient):
    _add_descriptor(client, name="tmp")
    resp = client.delete("/api/workspace/targets/tmp")
    assert resp.status_code == 204
    assert client.get("/api/workspace/targets").json()["total"] == 0


@pytest.mark.unit
def test_delete_unknown_404(client: TestClient):
    resp = client.delete("/api/workspace/targets/missing")
    assert resp.status_code == 404


# ── POST {name}/test ───────────────────────────────────────────────────


@pytest.mark.unit
def test_probe_unknown_404(client: TestClient):
    resp = client.post("/api/workspace/targets/no-such/test")
    assert resp.status_code == 404


@pytest.mark.unit
def test_probe_happy_path(
    client: TestClient, isolated_registry: WorkspaceTargetRegistry, tmp_path: Path
):
    """Stub the FS factory to return a LocalFileSystem rooted under
    tmp_path so we can exercise the full probe without SSH."""
    from molexp.workspace.fs_local import LocalFileSystem

    remote_root = tmp_path / "remote_root"

    _add_descriptor(client, name="lab", root_path=str(remote_root))

    def fake_fs_factory(_target):
        return LocalFileSystem()

    client.app.dependency_overrides[get_remote_fs_factory] = lambda: fake_fs_factory
    try:
        resp = client.post("/api/workspace/targets/lab/test")
    finally:
        client.app.dependency_overrides.pop(get_remote_fs_factory, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True, body
    assert body["error"] is None
    labels = [c["label"] for c in body["checks"]]
    assert any("mkdir" in label for label in labels)
    assert any("round-trip" in label for label in labels)
    # The probe file must have been cleaned up afterwards.
    probe = remote_root / ".molexp-workspace-test"
    assert not probe.exists()


@pytest.mark.unit
def test_probe_failure_returns_200_with_ok_false(client: TestClient, tmp_path: Path):
    """When the FS raises on mkdir, the probe returns HTTP 200 + ok=False
    so the UI can render inline (matches /api/targets test pattern)."""
    _add_descriptor(client, name="lab", root_path="/should/be/unreachable")

    class _ExplodingFS:
        def mkdir(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise OSError("simulated mkdir failure")

    def fake_fs_factory(_target):
        return _ExplodingFS()

    client.app.dependency_overrides[get_remote_fs_factory] = lambda: fake_fs_factory
    try:
        resp = client.post("/api/workspace/targets/lab/test")
    finally:
        client.app.dependency_overrides.pop(get_remote_fs_factory, None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] is not None
    failing = [c for c in body["checks"] if c["ok"] is False]
    assert failing, body
    assert "mkdir" in failing[0]["label"]


# ── independence from get_workspace ────────────────────────────────────


@pytest.mark.unit
def test_registry_endpoints_work_without_workspace_open(
    isolated_registry: WorkspaceTargetRegistry, tmp_path: Path
):
    """The registry endpoints must function before any workspace is open."""
    app = create_app()
    app.dependency_overrides[get_workspace_target_registry] = lambda: isolated_registry
    # Deliberately do NOT override get_workspace.
    bare_client = TestClient(app)

    # GET still works.
    resp = bare_client.get("/api/workspace/targets")
    assert resp.status_code == 200

    # POST still works.
    resp = bare_client.post(
        "/api/workspace/targets",
        json={
            "name": "first",
            "host": "h",
            "root_path": "/r",
        },
    )
    assert resp.status_code == 201
