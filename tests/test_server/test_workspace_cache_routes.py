"""Server cache-control routes + the now-FS-routed /file/blob path.

Verifies:
- ``POST /api/workspace/cache/invalidate`` drops entries (scope=all / indices / specific path).
- ``POST /api/workspace/cache/refresh`` invalidates *then* prefetches and returns warnings.
- ``GET /api/workspace/file/blob`` routes through ``workspace._fs`` (regression — used to
  open(target, "rb") which is broken for remote workspaces).
- Local workspaces return 409 from cache routes (they have no mirror).
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
from molexp.workspace.fs_cached import CachedRemoteFileSystem
from molexp.workspace.fs_local import LocalFileSystem


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import molexp.server.deps.targets as targets_deps

    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    reset_workspace_target_registry()
    registry = WorkspaceTargetRegistry(store_path=tmp_path / "wt.json")
    monkeypatch.setattr(targets_deps, "_workspace_target_registry", registry)
    yield
    reset_workspace_cache()
    set_workspace_path_override(None)
    set_active_workspace_descriptor(None)
    reset_workspace_target_registry()


def _wire_cached_remote(monkeypatch: pytest.MonkeyPatch, mirror_root: Path):
    """Stub the factory so the 'remote' workspace uses LocalFileSystem under a CachedRemoteFileSystem."""

    def _factory(_target: WorkspaceTarget) -> CachedRemoteFileSystem:
        return CachedRemoteFileSystem(LocalFileSystem(), mirror_root=mirror_root, ttl_seconds=300)

    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        _factory,
    )


def _make_remote_ws_on_disk(remote_root: Path) -> Workspace:
    remote_root.mkdir(parents=True, exist_ok=True)
    ws = Workspace(root=remote_root, name="remote")
    ws.materialize()
    proj = ws.add_project("alpha")
    exp = proj.add_experiment("first", workflow_source="train.py")
    exp.add_run(params={"lr": 1e-3})
    return ws


def _open_remote(client: TestClient, name: str = "lab"):
    return client.post("/api/workspace/open", json={"kind": "remote", "name": name})


@pytest.fixture
def remote_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Open a remote workspace via the route; returns (client, remote_root, mirror_root)."""
    remote_root = tmp_path / "remote_ws"
    mirror_root = tmp_path / "mirror"
    _make_remote_ws_on_disk(remote_root)
    _wire_cached_remote(monkeypatch, mirror_root)

    get_workspace_target_registry().add(
        WorkspaceTarget(name="lab", host="h", root_path=str(remote_root))
    )
    client = TestClient(create_app())
    resp = _open_remote(client)
    assert resp.status_code == 200, resp.text
    return client, remote_root, mirror_root


# ── Cache invalidate ──────────────────────────────────────────────────


@pytest.mark.integration
def test_invalidate_all_drops_everything(remote_client):
    client, _root, mirror_root = remote_client
    # Open already prefetched some entries — confirm a clear works.
    resp = client.post("/api/workspace/cache/invalidate", json={"scope": "all"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dropped"] >= 1
    assert not (mirror_root / "files").exists()


# ── Cache refresh ─────────────────────────────────────────────────────


@pytest.mark.integration
def test_refresh_invalidates_then_prefetches(remote_client):
    client, _root, _mirror = remote_client
    resp = client.post("/api/workspace/cache/refresh", json={"scope": "indices"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dropped"] >= 1
    assert body["warnings"] == []


# ── Local workspace rejected ─────────────────────────────────────────


@pytest.mark.integration
def test_cache_routes_reject_local_workspace(tmp_path: Path):
    client = TestClient(create_app())
    resp = client.post(
        "/api/workspace/open",
        json={"kind": "local", "path": str(tmp_path / "ws"), "create_if_missing": True},
    )
    assert resp.status_code == 200, resp.text
    bad = client.post("/api/workspace/cache/invalidate", json={"scope": "all"})
    assert bad.status_code == 409, bad.text


# ── /file/blob regression ─────────────────────────────────────────────


@pytest.mark.integration
def test_file_blob_serves_via_fs_for_remote_workspace(remote_client, tmp_path: Path):
    """Regression: /file/blob used to open(target, "rb") which crashes for remote workspaces."""
    client, root, _mirror = remote_client
    img = root / "projects" / "alpha" / "preview.png"
    # 1x1 PNG header bytes (not a real image — just non-empty bytes routed through StreamingResponse).
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)

    resp = client.get(f"/api/workspace/file/blob?path={img}")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/")
    assert resp.content.startswith(b"\x89PNG")


@pytest.mark.integration
def test_file_blob_rejects_non_image_extensions(remote_client, tmp_path: Path):
    client, root, _mirror = remote_client
    txt = root / "projects" / "alpha" / "data.bin"
    txt.write_bytes(b"binary garbage")
    resp = client.get(f"/api/workspace/file/blob?path={txt}")
    assert resp.status_code == 400, resp.text


# ── Warnings surface on open ─────────────────────────────────────────


@pytest.mark.integration
def test_open_remote_surfaces_transport_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A factory whose CachedRemoteFileSystem wraps a flaky FS must surface warnings, not 500."""
    remote_root = tmp_path / "ws"
    remote_root.mkdir()
    ws = Workspace(root=remote_root, name="ws")
    ws.materialize()
    proj = ws.add_project("solo")
    proj.add_experiment("first", workflow_source="train.py")

    class _FlakyLocalFS(LocalFileSystem):
        """LocalFileSystem that raises on the experiments-index read.

        Overrides ``read_bytes`` because CachedRemoteFileSystem.read_text
        routes through ``inner.read_bytes`` rather than ``inner.read_text``.
        """

        def read_bytes(self, path) -> bytes:  # type: ignore[override]
            if str(path).endswith("/projects/solo/experiment.json"):
                raise ConnectionError("simulated ssh blip")
            return super().read_bytes(path)

        def read_text(self, path, encoding: str = "utf-8") -> str:  # type: ignore[override]
            if str(path).endswith("/projects/solo/experiment.json"):
                raise ConnectionError("simulated ssh blip")
            return super().read_text(path, encoding=encoding)

    def _factory(_target):
        return CachedRemoteFileSystem(
            _FlakyLocalFS(), mirror_root=tmp_path / "mirror", ttl_seconds=300
        )

    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        _factory,
    )
    get_workspace_target_registry().add(
        WorkspaceTarget(name="lab", host="h", root_path=str(remote_root))
    )
    client = TestClient(create_app())
    resp = _open_remote(client)
    assert resp.status_code == 200, resp.text
    warnings = resp.json().get("warnings", [])
    assert any("simulated ssh blip" in w for w in warnings), warnings
