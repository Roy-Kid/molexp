"""Tests for ``ws.catalog`` — the AssetCatalog property on workspace.

Sub-spec ``unify-folder-abstraction-03`` retired the PascalCase
``Workspace.Catalog()`` accessor and the standalone ``CatalogFolder``
wrapper in favour of the lowercase ``ws.catalog`` property that returns
the :class:`AssetCatalog` directly. The on-disk layout
(``<workspace_root>/catalog/index.json``, no dotfile prefix) is
unchanged.
"""

from __future__ import annotations

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
    assert catalog.path.name == "index.sqlite"
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
    report = catalog.rebuild()
    assert report.errors == []
    assert Path(workspace.root / "catalog" / "index.sqlite").exists()


# ── AssetCatalog is re-exported from the assets package via a single owner ────


def test_assets_reexport_is_same_class() -> None:
    """``from molexp.workspace.assets import AssetCatalog`` is the same class."""
    from molexp.workspace.assets import AssetCatalog as AssetsReexport

    assert AssetsReexport is AssetCatalog


def test_workspace_catalog_property_uses_new_location(workspace: Workspace) -> None:
    """``workspace.catalog`` returns an ``AssetCatalog`` rooted at ``<root>/catalog``."""
    cat = workspace.catalog
    assert cat.dir == workspace.root / "catalog"
    assert cat.path == workspace.root / "catalog" / "index.sqlite"


# ── workspace-slim-02: _catalog_upsert convergence onto a Folder skeleton ─────
#
# The four entity ``_catalog_upsert`` bodies build a *different* per-kind
# catalog record and call a *different* ``catalog.upsert_<kind>``.  Only the
# dispatch — walk to the root workspace catalog, then hand it the
# kind-specific record — is shared and lifted onto ``Folder``.  These tests
# pin two contracts:
#
# * Payload equivalence (ac-004): the record handed to the catalog for each
#   kind keeps its exact field set and deterministic values.
# * Dispatch shape (ac-003): ``Folder`` owns ``_catalog_upsert`` plus a
#   per-kind ``_write_catalog_row`` hook; the four entities provide their
#   payload through the hook and no longer carry a divergent body.

from molexp.workspace.experiment import Experiment  # noqa: E402
from molexp.workspace.folder import Folder  # noqa: E402
from molexp.workspace.project import Project  # noqa: E402
from molexp.workspace.run import Run  # noqa: E402

_VOLATILE = {"created_at", "updated_at", "started_at", "finished_at"}


class _RecordingCatalog:
    """Captures the dict passed to each ``upsert_<kind>`` call."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def _capture(self, kind: str, record: dict) -> None:
        self.records[kind] = dict(record)

    def upsert_workspace(self, record: dict) -> None:
        self._capture("workspace", record)

    def upsert_project(self, record: dict) -> None:
        self._capture("project", record)

    def upsert_experiment(self, record: dict) -> None:
        self._capture("experiment", record)

    def upsert_run(self, record: dict) -> None:
        self._capture("run", record)

    def upsert_execution(self, record: dict) -> None:
        self._capture("execution", record)

    def upsert_run_with_executions(self, run_record: dict, execution_records: list[dict]) -> None:
        self._capture("run", run_record)
        for rec in execution_records:
            self._capture("execution", rec)

    # Removals are irrelevant to the create-path snapshot.
    def remove_project(self, *_a: object) -> None: ...
    def remove_experiment(self, *_a: object) -> None: ...
    def remove_run(self, *_a: object) -> None: ...
    def remove_execution(self, *_a: object) -> None: ...


@pytest.fixture
def recorded(tmp_path: Path):
    ws = Workspace(root=tmp_path / "ws", name="ws")
    rec = _RecordingCatalog()
    # Swap in the recorder before the first materialize triggers a catalog
    # access (the ``catalog`` property returns ``_catalog`` when set).
    ws._catalog = rec  # type: ignore[assignment]
    proj = ws.add_project("alpha")
    exp = proj.add_experiment("counter")
    run = exp.add_run(parameters={"x": 1})
    return ws, proj, exp, run, rec


def _assert_iso(record: dict) -> None:
    for key in _VOLATILE & set(record):
        value = record[key]
        assert value is None or isinstance(value, str), (key, value)


@pytest.mark.unit
def test_workspace_catalog_payload_snapshot(recorded) -> None:
    _ws, _proj, _exp, _run, rec = recorded
    record = rec.records["workspace"]
    assert set(record) == {"workspace_id", "root_path", "name", "created_at", "updated_at"}
    assert record["name"] == "ws"
    _assert_iso(record)


@pytest.mark.unit
def test_project_catalog_payload_snapshot(recorded) -> None:
    _ws, _proj, _exp, _run, rec = recorded
    record = rec.records["project"]
    assert set(record) == {
        "project_id",
        "workspace_id",
        "name",
        "description",
        "owner",
        "tags",
        "path",
        "created_at",
        "updated_at",
    }
    assert record["project_id"] == "alpha"
    assert record["name"] == "alpha"
    assert record["path"] == "projects/alpha"
    assert record["tags"] == []
    _assert_iso(record)


@pytest.mark.unit
def test_experiment_catalog_payload_snapshot(recorded) -> None:
    _ws, _proj, _exp, _run, rec = recorded
    record = rec.records["experiment"]
    assert set(record) == {
        "experiment_id",
        "project_id",
        "name",
        "description",
        "tags",
        "parameter_space",
        "n_replicas",
        "workflow_source",
        "workflow_type",
        "path",
        "created_at",
        "updated_at",
    }
    assert record["experiment_id"] == "counter"
    assert record["project_id"] == "alpha"
    assert record["path"] == "projects/alpha/experiments/counter"
    _assert_iso(record)


@pytest.mark.unit
def test_run_catalog_payload_snapshot(recorded) -> None:
    _ws, _proj, _exp, run, rec = recorded
    record = rec.records["run"]
    assert set(record) == {
        "run_id",
        "experiment_id",
        "status",
        "parameters",
        "profile",
        "config_hash",
        "labels",
        "path",
        "created_at",
        "finished_at",
        "workflow_snapshot",
    }
    assert record["run_id"] == run.id
    assert record["experiment_id"] == "counter"
    assert record["status"] == "pending"
    assert record["parameters"] == {"x": 1}
    assert record["profile"] is None
    assert record["config_hash"] is None
    assert record["workflow_snapshot"] is None
    assert record["path"] == f"runs/run-{run.id}"
    _assert_iso(record)


@pytest.mark.unit
def test_catalog_upsert_dispatch_skeleton_on_folder() -> None:
    """``Folder`` owns the dispatch; entities only supply the payload hook."""
    assert "_catalog_upsert" in vars(Folder), "Folder must own the dispatch skeleton"
    assert "_write_catalog_row" in vars(Folder), "Folder must declare the per-kind hook"
    for cls in (Workspace, Project, Experiment, Run):
        assert "_write_catalog_row" in vars(cls), f"{cls.__name__} must provide the payload hook"
        assert "_catalog_upsert" not in vars(cls), (
            f"{cls.__name__} must not keep its own _catalog_upsert body"
        )
