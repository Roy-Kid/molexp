"""Tests for ``ws.catalog`` — the AssetCatalog property on workspace.

Sub-spec ``unify-folder-abstraction-03`` retired the PascalCase
``Workspace.Catalog()`` accessor and the standalone ``CatalogFolder``
wrapper in favour of the lowercase ``ws.catalog`` property that returns
the :class:`AssetCatalog` directly. The on-disk layout
(``<workspace_root>/catalog/index.json``, no dotfile prefix) is
unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from molexp.workspace import (
    ArtifactAsset,
    AssetCatalog,
    AssetScope,
    Producer,
    Workspace,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab")
    ws.materialize()
    return ws


def _make_artifact(scope: AssetScope) -> ArtifactAsset:
    now = datetime.now()
    return ArtifactAsset(
        asset_id="ws/a1",
        name="a.bin",
        scope=scope,
        path=Path("artifacts/a.bin"),
        created_at=now,
        updated_at=now,
        producer=Producer(execution_id="e1", task_id="t1"),
        mime="application/octet-stream",
        size=1,
    )


# ── New path: <root>/catalog/index.json (no dotfile prefix) ──────────────────


def test_catalog_index_lands_at_root_catalog_index(workspace: Workspace) -> None:
    """``index.json`` lives at ``<root>/catalog/``, not ``<root>/.catalog/``."""
    catalog = workspace.catalog
    catalog.upsert_workspace(
        {
            "workspace_id": "ws-id",
            "root_path": "/tmp/lab",
            "name": "lab",
            "created_at": "2026-05-11T00:00:00Z",
            "updated_at": "2026-05-11T00:00:00Z",
        }
    )
    assert catalog.path.exists()
    assert catalog.path.name == "index.json"
    assert catalog.path.parent.name == "catalog"
    assert not Path(workspace.root / ".catalog").exists()


def test_workspace_catalog_is_idempotent_property(workspace: Workspace) -> None:
    """``ws.catalog`` is identity-stable across calls (singleton property)."""
    assert workspace.catalog is workspace.catalog


# ── AssetCatalog round-trip through the new path ──────────────────────────────


def test_register_and_query_round_trip(workspace: Workspace) -> None:
    catalog = workspace.catalog
    scope = AssetScope(kind="workspace", ids=())
    catalog.register(_make_artifact(scope))

    results = catalog.query_assets(scope=scope)
    assert len(results) == 1
    assert results[0].asset_id == "ws/a1"


def test_rebuild_uses_new_path(workspace: Workspace) -> None:
    catalog = workspace.catalog
    catalog.rebuild()
    payload = json.loads(Path(workspace.root / "catalog" / "index.json").read_text())
    assert isinstance(payload, dict)
    assert "workspaces" in payload


# ── AssetCatalog is re-exported from the assets package via a single owner ────


def test_assets_reexport_is_same_class() -> None:
    """``from molexp.workspace.assets import AssetCatalog`` is the same class."""
    from molexp.workspace.assets import AssetCatalog as AssetsReexport

    assert AssetsReexport is AssetCatalog


def test_workspace_catalog_property_uses_new_location(workspace: Workspace) -> None:
    """``workspace.catalog`` returns an ``AssetCatalog`` rooted at ``<root>/catalog``."""
    cat = workspace.catalog
    assert cat.dir == workspace.root / "catalog"
    assert cat.path == workspace.root / "catalog" / "index.json"
