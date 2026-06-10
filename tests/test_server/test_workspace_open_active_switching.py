"""Tests for the discriminated WorkspaceOpenRequest + active-workspace
switching contract (sub-spec 02)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import (
    get_active_workspace,
    get_workspace_target_registry,
    register_workspace_subscriber,
    reset_workspace_cache,
    reset_workspace_target_registry,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)
from molexp.server.workspace_targets import (
    WorkspaceTarget,
    WorkspaceTargetRegistry,
)


@pytest.fixture(autouse=True)
def _isolate_dependency_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reset cache + overrides + registry for each test."""
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


@pytest.fixture
def client():
    return TestClient(create_app())


def _stub_local_fs_factory(monkeypatch: pytest.MonkeyPatch):
    """Make target_to_filesystem_for_workspace_target return a LocalFileSystem."""
    from molexp.workspace.fs_local import LocalFileSystem

    monkeypatch.setattr(
        "molexp.server.workspace_targets.target_to_filesystem_for_workspace_target",
        lambda _t: LocalFileSystem(),
    )


# ── Request discriminator ──────────────────────────────────────────────


@pytest.mark.unit
def test_default_kind_is_local(client: TestClient, tmp_path: Path):
    body = {"path": str(tmp_path / "ws-local"), "create_if_missing": True}
    resp = client.post("/api/workspace/open", json=body)
    assert resp.status_code == 200, resp.text
    assert str(tmp_path / "ws-local") in resp.json()["root"]


@pytest.mark.unit
def test_request_discriminator_local(client: TestClient, tmp_path: Path):
    body = {"kind": "local", "path": str(tmp_path / "ws-local"), "create_if_missing": True}
    resp = client.post("/api/workspace/open", json=body)
    assert resp.status_code == 200, resp.text


@pytest.mark.unit
def test_request_discriminator_remote(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _stub_local_fs_factory(monkeypatch)
    remote_root = tmp_path / "remote-root"
    remote_root.mkdir()
    get_workspace_target_registry().add(
        WorkspaceTarget(name="hpc1", host="me@host", root_path=str(remote_root))
    )

    resp = client.post("/api/workspace/open", json={"kind": "remote", "name": "hpc1"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["root"] == str(remote_root)


@pytest.mark.unit
def test_unknown_descriptor_404(client: TestClient):
    resp = client.post("/api/workspace/open", json={"kind": "remote", "name": "missing"})
    assert resp.status_code == 404
    assert "missing" in resp.json()["detail"]


@pytest.mark.unit
def test_missing_name_422(client: TestClient):
    resp = client.post("/api/workspace/open", json={"kind": "remote"})
    assert resp.status_code == 422


@pytest.mark.unit
def test_missing_path_422(client: TestClient):
    resp = client.post("/api/workspace/open", json={"kind": "local"})
    assert resp.status_code == 422


# ── Cache + override invariants ────────────────────────────────────────


@pytest.mark.unit
def test_cache_rekeyed_on_switch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """After local→remote switch, no key in _workspace_cache has kind == 'local'."""
    import molexp.server.dependencies as deps

    _stub_local_fs_factory(monkeypatch)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    get_workspace_target_registry().add(
        WorkspaceTarget(name="hpc1", host="h", root_path=str(remote_root))
    )

    # Open local
    local_root = tmp_path / "local"
    client.post(
        "/api/workspace/open",
        json={"kind": "local", "path": str(local_root), "create_if_missing": True},
    )
    # Make sure cache contains the local entry
    _ = get_active_workspace()
    assert any(k[0] == "local" for k in deps._workspace_cache)

    # Switch to remote — cache should be cleared (so no stale local entry left)
    client.post("/api/workspace/open", json={"kind": "remote", "name": "hpc1"})

    stale_local = [k for k in deps._workspace_cache if k[0] == "local"]
    assert stale_local == [], f"stale local cache keys after switch: {stale_local}"


@pytest.mark.unit
def test_overrides_mutually_exclusive(tmp_path: Path):
    import molexp.server.dependencies as deps

    set_workspace_path_override(tmp_path / "x")
    assert deps._workspace_path_override is not None
    assert deps._workspace_descriptor_override is None

    set_active_workspace_descriptor("hpc1")
    assert deps._workspace_descriptor_override == "hpc1"
    assert deps._workspace_path_override is None

    set_workspace_path_override(tmp_path / "y")
    assert deps._workspace_descriptor_override is None
    assert deps._workspace_path_override is not None


# ── Subscriber drain ───────────────────────────────────────────────────


@pytest.mark.unit
def test_subscribers_drained_before_cache_reset(tmp_path: Path):
    """Registered closers must run *before* the workspace cache is reset."""
    order: list[str] = []

    def closer():
        order.append("closer")

    # Plant a synthetic entry so we can observe cache eviction
    import molexp.server.dependencies as deps

    deps._workspace_cache[("local", "/tmp/sentinel")] = object()  # type: ignore[assignment]
    register_workspace_subscriber(closer)
    set_workspace_path_override(tmp_path / "x")
    # After the setter returns, closer must have run and cache must be empty.
    assert "closer" in order
    assert ("local", "/tmp/sentinel") not in deps._workspace_cache


@pytest.mark.unit
def test_subscribers_support_awaitable_closers(tmp_path: Path):
    """Closers may return an awaitable; the drain awaits them."""
    order: list[str] = []

    async def async_closer():
        order.append("async-closer")

    register_workspace_subscriber(async_closer)
    set_workspace_path_override(tmp_path / "x")
    assert order == ["async-closer"]


@pytest.mark.unit
def test_subscribers_cleared_after_drain(tmp_path: Path):
    """Drain must clear the subscriber list — switching twice doesn't double-fire."""
    calls: list[int] = []

    def closer():
        calls.append(1)

    register_workspace_subscriber(closer)
    set_workspace_path_override(tmp_path / "a")
    set_workspace_path_override(tmp_path / "b")
    assert calls == [1]
