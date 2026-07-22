"""Manifest-scanning asset query layer (``assets/scan.py``).

The scanner is the asset query surface that replaced the derived SQLite
``AssetCatalog``: every query shape is answered by scanning the authoritative
``assets.json`` + ``assets/<id>/asset.json`` records
(spec: workspace-git-projection-01-drop-catalog).
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import ArtifactAsset, scan

# Each started run persists four assets: the user's artifact + "train" log +
# checkpoint, plus the lifecycle's auto-created "run" log.
ASSETS_PER_RUN = 4


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


def _kinds(assets) -> set[str]:
    return {a.kind for a in assets}


class TestScanAssets:
    """Each query shape returns the expected asset set from the manifests."""

    def test_returns_every_asset_across_scopes(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")  # 3 runs x 4 assets
        assets = scan.scan_assets(ws.root)
        assert len(assets) == 3 * ASSETS_PER_RUN
        assert _kinds(assets) == {"artifact", "log", "checkpoint"}

    def test_kind_filter_accepts_str_or_asset_subclass(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        by_str = scan.scan_assets(ws.root, kind="artifact")
        by_type = scan.scan_assets(ws.root, kind=ArtifactAsset)
        assert {a.asset_id for a in by_str} == {a.asset_id for a in by_type}
        assert len(by_str) == 3
        assert all(a.kind == "artifact" for a in by_str)

    def test_run_scope_matches_only_that_run(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        run = ws.project("demo").experiment("baseline").list_runs()[0]
        scoped = scan.scan_assets(ws.root, scope=run.scope)
        assert len(scoped) == ASSETS_PER_RUN
        assert all(a.scope == run.scope for a in scoped)

    def test_experiment_scope_recursive_vs_exact(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        exp = ws.project("demo").experiment("baseline")
        assert len(scan.scan_assets(ws.root, scope=exp.scope, recursive=True)) == 3 * ASSETS_PER_RUN
        # Non-recursive experiment scope sees no run-scoped assets.
        assert scan.scan_assets(ws.root, scope=exp.scope) == []

    def test_limit_caps_result_count(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        assert len(scan.scan_assets(ws.root, limit=2)) == 2

    def test_results_sorted_by_created_at_then_id(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        keys = [(a.created_at, a.asset_id) for a in scan.scan_assets(ws.root)]
        assert keys == sorted(keys)

    def test_empty_workspace_yields_no_assets(self, tmp_path):
        ws = Workspace(tmp_path / "empty")
        ws.materialize()
        assert scan.scan_assets(ws.root) == []
        assert scan.get_asset(ws.root, "x") is None


class TestGetAsset:
    def test_returns_asset_by_id_else_none(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        some = scan.scan_assets(ws.root)[0]
        assert scan.get_asset(ws.root, some.asset_id).asset_id == some.asset_id
        assert scan.get_asset(ws.root, "nonexistent") is None


class TestFindByContentHash:
    def test_matches_hash_and_rejects_absent_or_empty(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab")
        artifact = scan.scan_assets(ws.root, kind="artifact")[0]
        assert artifact.content_hash
        found = scan.find_by_content_hash(ws.root, artifact.content_hash)
        assert found is not None
        assert found.content_hash == artifact.content_hash
        assert scan.find_by_content_hash(ws.root, "sha256:absent") is None
        assert scan.find_by_content_hash(ws.root, "") is None


class TestNoDerivedSqliteIndex:
    """Invariant lock: the derived SQLite ``AssetCatalog`` is gone — the
    authoritative per-scope ``assets.json`` manifests are the only on-disk
    asset record (One-source-of-truth law)."""

    def test_seeded_workspace_writes_only_manifests_no_sqlite(self, tmp_path):
        from molexp.workspace.events import WORKSPACE_EVENTS_DB

        ws = _seed_workspace(tmp_path / "lab")  # 3 runs, assets persisted
        root = Path(str(ws.root))
        # Assets are queryable …
        assert len(scan.scan_assets(ws.root)) == 3 * ASSETS_PER_RUN
        # … yet nothing was written to a derived SQLite ASSET index. (The
        # run-lifecycle event spine is a sanctioned, non-asset sidecar — the
        # default-on run milestones legitimately create it.)
        assert not (root / "catalog").exists()
        assert {p.name for p in root.rglob("*.sqlite")} <= {WORKSPACE_EVENTS_DB}
        # The authoritative record is the per-scope assets.json manifest.
        assert list(root.rglob("assets.json"))
