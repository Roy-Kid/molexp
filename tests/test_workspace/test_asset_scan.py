"""Manifest-scanning asset query layer (``assets/scan.py``).

These tests pin the scanner as a behaviour-preserving replacement for the
derived SQLite ``AssetCatalog``: for every query shape, scanning the
authoritative ``assets.json`` manifests must return the same asset set the
catalog does. (Spec: workspace-git-projection-01-drop-catalog.)
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import ArtifactAsset, scan


def _seed_workspace(root: Path, n_runs: int = 3) -> Workspace:
    ws = Workspace(root=root, name="Test")
    proj = ws.add_project("demo")
    exp = proj.add_experiment("baseline", params={"lr": 1e-3})
    for i in range(n_runs):
        r = exp.add_run(params={"seed": i})
        with r.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1 * i})
            ctx.log("train").append(f"run {i} starting")
            ctx.checkpoint("epoch1", data={"step": 1})
    return ws


def _ids(assets) -> set[str]:
    return {a.asset_id for a in assets}


class TestScannerEqualsCatalog:
    """The scanner returns the same asset set as the catalog for each query."""

    def test_all_assets(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        assert _ids(scan.scan_assets(root)) == _ids(ws.catalog.query_assets())
        assert len(scan.scan_assets(root)) >= 9  # 3 runs x (artifact+log+ckpt)

    def test_kind_filter(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        assert _ids(scan.scan_assets(root, kind="artifact")) == _ids(
            ws.catalog.query_assets(kind="artifact")
        )
        assert _ids(scan.scan_assets(root, kind=ArtifactAsset)) == _ids(
            ws.catalog.query_assets(kind=ArtifactAsset)
        )

    def test_run_scope_exact(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        run = ws.project("demo").experiment("baseline").list_runs()[0]
        assert _ids(scan.scan_assets(root, scope=run.scope)) == _ids(
            ws.catalog.query_assets(scope=run.scope)
        )

    def test_experiment_scope_recursive(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        exp = ws.project("demo").experiment("baseline")
        recursive = scan.scan_assets(root, scope=exp.scope, recursive=True)
        assert _ids(recursive) == _ids(ws.catalog.query_assets(scope=exp.scope, recursive=True))
        # non-recursive experiment scope sees no run-scoped assets here
        assert _ids(scan.scan_assets(root, scope=exp.scope)) == _ids(
            ws.catalog.query_assets(scope=exp.scope)
        )

    def test_limit(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        assert len(scan.scan_assets(root, limit=2)) == 2


class TestScannerLookups:
    def test_get_asset(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        some = scan.scan_assets(root)[0]
        assert scan.get_asset(root, some.asset_id).asset_id == some.asset_id
        assert scan.get_asset(root, "nonexistent") is None

    def test_find_by_content_hash(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        artifact = scan.scan_assets(root, kind="artifact")[0]
        assert artifact.content_hash
        found = scan.find_by_content_hash(root, artifact.content_hash)
        assert found is not None
        assert found.content_hash == artifact.content_hash
        assert scan.find_by_content_hash(root, "sha256:absent") is None
        assert scan.find_by_content_hash(root, "") is None


class TestScannerDeterministicOrder:
    def test_sorted_by_created_at_then_id(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        root = ws.root
        assets = scan.scan_assets(root)
        keys = [(a.created_at, a.asset_id) for a in assets]
        assert keys == sorted(keys)


class TestEmptyWorkspace:
    def test_no_manifests_returns_empty(self, tmp_path):
        ws = Workspace(tmp_path / "empty")
        ws.materialize()
        assert scan.scan_assets(ws.root) == []
        assert scan.get_asset(ws.root, "x") is None
