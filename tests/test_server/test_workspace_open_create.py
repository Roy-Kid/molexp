"""Tests for the ``create_if_missing`` branch of ``POST /api/workspace/open``.

Spec ui-creation-entries: a create-open (``create_if_missing: true`` for a
missing local path) must *materialize* the new workspace — i.e. write
``<root>/workspace.json`` — while every other open flavor stays write-free:
refusing a missing path (404) must not create the directory, and opening an
already-existing directory must never materialize it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import (
    reset_workspace_cache,
    reset_workspace_target_registry,
    set_active_workspace_descriptor,
    set_workspace_path_override,
)
from molexp.server.workspace_targets import WorkspaceTargetRegistry


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
    """Plain app client.

    Workspace resolution goes through the real path-override machinery;
    the conftest ``client`` fixture (which overrides ``get_workspace``)
    is deliberately shadowed.
    """
    return TestClient(create_app())


# ── create_if_missing: true ────────────────────────────────────────────


@pytest.mark.unit
def test_create_open_returns_resolved_root_and_zero_projects(client: TestClient, tmp_path: Path):
    """Create-open of a missing path → 200 with resolved root and 0 projects."""
    missing = tmp_path / "fresh-ws"
    assert not missing.exists()

    resp = client.post(
        "/api/workspace/open",
        json={"path": str(missing), "create_if_missing": True},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert Path(body["root"]) == missing.resolve()
    assert body["projectCount"] == 0


@pytest.mark.unit
def test_create_open_materializes_workspace_json(client: TestClient, tmp_path: Path):
    """Create-open must write ``workspace.json`` at the just-created root."""
    missing = tmp_path / "fresh-ws"

    resp = client.post(
        "/api/workspace/open",
        json={"path": str(missing), "create_if_missing": True},
    )

    assert resp.status_code == 200, resp.text
    marker = Path(resp.json()["root"]) / "workspace.json"
    assert marker.is_file(), "create-open must materialize workspace.json at the new root"


# ── missing path without create_if_missing ─────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra_body",
    [
        pytest.param({"create_if_missing": False}, id="explicit-false"),
        pytest.param({}, id="omitted"),
    ],
)
def test_open_missing_path_without_create_is_404_and_side_effect_free(
    client: TestClient, tmp_path: Path, extra_body: dict[str, bool]
):
    """Missing path + no create consent → 404 and the directory stays absent."""
    missing = tmp_path / "absent-ws"

    resp = client.post("/api/workspace/open", json={"path": str(missing), **extra_body})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Workspace path not found"
    assert not missing.exists(), "refused open must not create the directory"


# ── open-existing is not materialize ───────────────────────────────────


@pytest.mark.unit
def test_open_existing_empty_dir_never_materializes(client: TestClient, tmp_path: Path):
    """Opening a pre-existing empty directory must not write workspace.json."""
    existing = tmp_path / "plain-dir"
    existing.mkdir()

    resp = client.post("/api/workspace/open", json={"path": str(existing)})

    assert resp.status_code == 200, resp.text
    assert not (existing / "workspace.json").exists(), (
        "opening an existing directory must never materialize workspace.json"
    )


# ── create then reopen lifecycle ───────────────────────────────────────


@pytest.mark.unit
def test_reopen_after_create_succeeds_without_create_flag(client: TestClient, tmp_path: Path):
    """After a create-open, the same path reopens with create_if_missing false."""
    target = tmp_path / "fresh-ws"

    first = client.post(
        "/api/workspace/open",
        json={"path": str(target), "create_if_missing": True},
    )
    assert first.status_code == 200, first.text

    second = client.post(
        "/api/workspace/open",
        json={"path": str(target), "create_if_missing": False},
    )

    assert second.status_code == 200, second.text
    assert second.json()["root"] == first.json()["root"]
