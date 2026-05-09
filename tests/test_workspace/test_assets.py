"""Invariant tests for the unified asset model.

Covers the success criteria from ``docs/development/specs/unified-asset-model.md`` §8:

- Catalog is regenerable from filesystem
- Run directories are portable
- Manifest/catalog/disk stay consistent under rebuild
- Subclass dispatch survives round-trips
- Typed accessors populate Producer correctly
- Concurrent asset writes all land in the catalog
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
    AssetCatalog,
    AssetManifest,
    AssetScope,
    CheckpointAsset,
    DataAsset,
    ErrorTraceAsset,
    LogAsset,
    parse_asset,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _seed_workspace(root: Path, n_runs: int = 2) -> Workspace:
    ws = Workspace(root=root, name="Test")
    proj = ws.Project("demo")
    exp = proj.Experiment("baseline", params={"lr": 1e-3})
    for i in range(n_runs):
        r = exp.Run(parameters={"seed": i})
        with r.start() as ctx:
            ctx.artifact.save("metrics.json", {"loss": 0.1 * i})
            ctx.log("train").append(f"run {i} starting")
            ctx.checkpoint("epoch1", data={"step": 1})
    return ws


# ── Regenerable catalog ────────────────────────────────────────────────────


class TestCatalogRebuild:
    def test_rebuild_matches_live_state(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=3)
        catalog = ws.catalog

        before = catalog._load()
        assert len(before["runs"]) == 3
        assert len(before["assets"]) >= 9  # 3 runs × (artifact + log + ckpt)

        # Wipe and rebuild
        shutil.rmtree(tmp_path / "lab" / ".catalog")
        fresh = Workspace(tmp_path / "lab")
        report = fresh.catalog.rebuild()

        assert report.errors == []
        after = fresh.catalog._load()
        assert set(after["runs"]) == set(before["runs"])
        assert set(after["assets"]) == set(before["assets"])
        assert set(after["projects"]) == set(before["projects"])
        assert set(after["experiments"]) == set(before["experiments"])

    def test_rebuild_idempotent(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        r1 = ws.catalog.rebuild()
        r2 = ws.catalog.rebuild()
        assert r1.assets == r2.assets
        assert r1.runs == r2.runs

    def test_rebuild_handles_missing_manifest(self, tmp_path):
        # A materialized workspace with no child assets still rebuilds cleanly
        ws = Workspace(tmp_path / "empty")
        ws.materialize()
        report = ws.catalog.rebuild()
        assert report.errors == []
        assert report.workspaces == 1
        assert report.assets == 0


# ── Run portability ────────────────────────────────────────────────────────


class TestRunPortability:
    def test_tar_move_rebuild(self, tmp_path):
        """A run directory moved under a new workspace stays queryable."""
        src_ws = _seed_workspace(tmp_path / "source", n_runs=1)
        runs = src_ws.catalog.query_runs()
        assert len(runs) == 1
        run_id = runs[0]["run_id"]

        # Build destination workspace scaffolding
        dst_root = tmp_path / "destination"
        dst_ws = Workspace(dst_root, name="Destination")
        dst_proj = dst_ws.Project("demo")
        dst_exp = dst_proj.Experiment("baseline", params={"lr": 1e-3})

        # Move the physical run directory
        src_run_dir = (
            tmp_path
            / "source"
            / "projects"
            / "demo"
            / "experiments"
            / dst_exp.id
            / "runs"
            / f"run-{run_id}"
        )
        # Source uses a different experiment slug — rediscover it
        src_exp_dir = tmp_path / "source" / "projects" / "demo" / "experiments"
        actual_src_exp = next(src_exp_dir.iterdir())
        src_run_dir = next((actual_src_exp / "runs").iterdir())

        dst_run_dir = dst_exp.experiment_dir / "runs" / src_run_dir.name
        dst_run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_run_dir, dst_run_dir)

        # Rewrite asset scope ids in the manifest (project/experiment slugs may differ).
        # Here both are identically "demo" + "baseline" so no rewrite needed.

        report = dst_ws.catalog.rebuild()
        assert report.errors == []
        assert report.runs == 1
        assert report.assets >= 3


# ── Manifest <-> disk consistency ──────────────────────────────────────────


class TestManifestConsistency:
    def test_every_manifest_entry_points_to_existing_file(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        for run_record in ws.catalog.query_runs():
            run_dir = ws.root / run_record["path"]
            manifest = AssetManifest(run_dir)
            for asset in manifest.list():
                assert asset.absolute_path(run_dir).exists(), (
                    f"missing: {asset.uri} -> {asset.path}"
                )


# ── Subclass dispatch ──────────────────────────────────────────────────────


class TestSubclassDispatch:
    def test_round_trip_preserves_type(self, tmp_path):
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
            dumped = json.loads(asset.model_dump_json())
            revived = parse_asset(dumped)
            assert type(revived) is type(asset)
            assert revived.asset_id == asset.asset_id


# ── Producer propagation ───────────────────────────────────────────────────


class TestProducerPropagation:
    def test_artifact_producer_set(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=1)
        run_id = ws.catalog.query_runs()[0]["run_id"]
        artifacts = ws.catalog.query_assets(kind="artifact")
        assert len(artifacts) == 1
        assert artifacts[0].producer is not None
        assert artifacts[0].producer.run_id == run_id
        assert artifacts[0].producer.execution_id is not None

    def test_task_id_set_via_set_active_task(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.Project("p").Experiment("e").Run()
        with run.start() as ctx:
            ctx.set_active_task("train")
            asset = ctx.artifact.save("m.json", {"x": 1})
        assert asset.producer.task_id == "train"


# ── Concurrent writes within a run ─────────────────────────────────────────


class TestConcurrentWrites:
    def test_parallel_artifact_writes_all_registered(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.Project("p").Experiment("e").Run()
        N = 20
        with run.start() as ctx, ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(ctx.artifact.save, f"a{i}.json", {"i": i}) for i in range(N)]
            results = [f.result() for f in as_completed(futs)]

        assert len(results) == N
        catalog_assets = ws.catalog.query_assets(kind="artifact", producer_run=run.id)
        assert len(catalog_assets) == N
        manifest_assets = AssetManifest(run.run_dir).list()
        # manifest also contains the auto-created "run" log
        artifact_in_manifest = [a for a in manifest_assets if a.kind == "artifact"]
        assert len(artifact_in_manifest) == N


# ── AssetsView scoping ─────────────────────────────────────────────────────


class TestAssetsView:
    def test_scope_filtering(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        proj = ws.list_projects()[0]
        exp = proj.list_experiments()[0]

        # Workspace scope should find zero produced assets (all are run-scoped)
        assert ws.assets.list() == []
        assert proj.assets.list() == []
        assert exp.assets.list() == []

        # Run scopes should each have artifact+log+ckpt
        for run in exp.list_runs():
            view_assets = AssetCatalog(ws.root).query_assets(
                scope=AssetScope(
                    kind="run",
                    ids=(proj.id, exp.id, run.id),
                )
            )
            kinds = {a.kind for a in view_assets}
            assert "artifact" in kinds
            assert "log" in kinds
            assert "checkpoint" in kinds

    def test_data_asset_import_scope(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        src = tmp_path / "input.txt"
        src.write_text("hello")
        asset = ws.data_assets.import_asset("greeting", src)
        assert isinstance(asset, DataAsset)
        assert asset.scope.kind == "workspace"

        # Visible in workspace view + catalog
        assert ws.assets.get(asset.asset_id) is not None
        assert ws.catalog.get(asset.asset_id) is not None
