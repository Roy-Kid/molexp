"""Invariant tests for the unified asset model.

Covers the asset-model classes (``Asset`` hierarchy, ``AssetManifest``,
``AssetsView``, ``DataAssetLibrary``, ``parse_asset``) and their success
criteria from ``docs/development/specs/unified-asset-model.md`` §8:

- Run directories are portable (assets discoverable from on-disk manifests).
- Manifest and disk stay consistent.
- Subclass dispatch survives serialization round-trips.
- Typed accessors populate ``Producer`` correctly.
- Concurrent asset writes all land in the manifest.
- The scope-bound ``AssetsView`` filters to its own scope; imports land there.

(Cross-cutting ``scan.py`` query shapes are owned by ``test_asset_scan.py``.)
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import (
    ArtifactAsset,
    AssetManifest,
    AssetScope,
    CheckpointAsset,
    DataAsset,
    ErrorTraceAsset,
    LogAsset,
    parse_asset,
    scan,
)


def _seed_workspace(root: Path, n_runs: int = 2) -> Workspace:
    ws = Workspace(root=root, name="Test")
    proj = ws.add_project("demo")
    exp = proj.add_experiment("baseline", params={"lr": 1e-3})
    for i in range(n_runs):
        r = exp.add_run(params={"seed": i})
        with r.start() as ctx:
            ctx.register_artifact({"loss": 0.1 * i}, name="metrics.json")
            ctx.log("train").append(f"run {i} starting")
            ctx.checkpoint("epoch1", data={"step": 1})
    return ws


class TestRunPortability:
    def test_moved_run_dir_stays_queryable_via_manifests(self, tmp_path):
        """A run directory copied under a *different* workspace stays queryable
        via the authoritative manifests — no absolute paths, no index to rebuild."""
        _seed_workspace(tmp_path / "source", n_runs=1)
        src_exp_dir = tmp_path / "source" / "projects" / "demo" / "experiments"
        actual_src_exp = next(src_exp_dir.iterdir())
        src_run_dir = next((actual_src_exp / "runs").iterdir())

        dst_ws = Workspace(tmp_path / "destination", name="Destination")
        dst_proj = dst_ws.add_project("demo")
        dst_exp = dst_proj.add_experiment("baseline", params={"lr": 1e-3})

        dst_run_dir = Path(dst_exp.experiment_dir) / "runs" / src_run_dir.name
        dst_run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_run_dir, dst_run_dir)

        found = scan.scan_assets(dst_ws.root)
        assert len(found) >= 3  # artifact + log + checkpoint


class TestAssetManifest:
    def test_every_entry_points_to_an_existing_file(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        exp = ws.project("demo").experiment("baseline")
        for run in exp.list_runs():
            run_dir = Path(run.run_dir)
            manifest = AssetManifest(run_dir)
            for asset in manifest.list():
                assert asset.absolute_path(run_dir).exists(), (
                    f"missing: {asset.uri} -> {asset.path}"
                )

    def test_parallel_saves_all_register(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.add_project("p").add_experiment("e").add_run()
        n = 20
        with run.start() as ctx, ThreadPoolExecutor(max_workers=4) as pool:
            futs = [
                pool.submit(lambda i=i: ctx.register_artifact({"i": i}, name=f"a{i}.json"))
                for i in range(n)
            ]
            results = [f.result() for f in as_completed(futs)]

        assert len(results) == n
        scanned = scan.scan_assets(ws.root, kind="artifact", producer_run=run.id)
        assert len(scanned) == n
        manifest_artifacts = [
            a for a in AssetManifest(Path(run.run_dir)).list() if a.kind == "artifact"
        ]
        assert len(manifest_artifacts) == n


class TestParseAsset:
    def test_round_trip_preserves_each_subclass(self):
        scope = AssetScope(kind="run", ids=("p", "e", "run-1"))
        now = datetime.now()
        cases = [
            ArtifactAsset(
                asset_id="a1",
                name="m.json",
                scope=scope,
                path=Path("artifacts/m.json"),
                created_at=now,
                updated_at=now,
                mime="application/json",
                size=10,
            ),
            LogAsset(
                asset_id="l1",
                name="run",
                scope=scope,
                path=Path("executions/ex-1/logs/run.log"),
                created_at=now,
                updated_at=now,
            ),
            CheckpointAsset(
                asset_id="c1",
                name="ckpt1",
                scope=scope,
                path=Path(".ckpt/c1.json"),
                created_at=now,
                updated_at=now,
                ckpt_id="ckpt_abc",
                parent_ckpt_id=None,
            ),
            ErrorTraceAsset(
                asset_id="e1",
                name="err",
                scope=scope,
                path=Path("executions/ex-1/error.txt"),
                created_at=now,
                updated_at=now,
                exception_type="RuntimeError",
                message="oops",
                execution_id="ex-1",
            ),
            DataAsset(
                asset_id="d1",
                name="ds",
                scope=scope,
                path=Path("assets/d1/payload"),
                created_at=now,
                updated_at=now,
                source_path="/tmp/ds",
                import_action="copy",
            ),
        ]
        for asset in cases:
            revived = parse_asset(json.loads(asset.model_dump_json()))
            assert type(revived) is type(asset)
            assert revived.asset_id == asset.asset_id


class TestProducer:
    def test_active_task_sets_producer_task_id(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            ctx.set_active_task("train")
            asset = ctx.register_artifact({"x": 1}, name="m.json")
        assert asset.producer.task_id == "train"


class TestAssetsView:
    def test_scope_view_returns_only_its_own_scope(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        proj = ws.list_projects()[0]
        exp = proj.list_experiments()[0]

        # All produced assets are run-scoped: non-run scopes see nothing.
        assert ws.assets.list() == []
        assert proj.assets.list() == []
        assert exp.assets.list() == []

        for run in exp.list_runs():
            kinds = {a.kind for a in run.assets.list()}
            assert {"artifact", "log", "checkpoint"} <= kinds

    def test_imported_data_asset_lands_at_workspace_scope(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        src = tmp_path / "input.txt"
        src.write_text("hello")
        asset = ws.data_assets.import_asset("greeting", src)
        assert isinstance(asset, DataAsset)
        assert asset.scope.kind == "workspace"

        # Visible through both the workspace view and the manifest scanner.
        assert ws.assets.get(asset.asset_id) is not None
        assert scan.get_asset(ws.root, asset.asset_id) is not None
