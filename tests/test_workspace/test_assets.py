"""Invariant tests for the unified asset model.

Covers the success criteria from ``docs/development/specs/unified-asset-model.md`` §8:

- Run directories are portable (assets discoverable from on-disk manifests)
- Manifest/disk stay consistent
- Subclass dispatch survives round-trips
- Typed accessors populate Producer correctly
- Concurrent asset writes all land in the manifest
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

# ── Helpers ────────────────────────────────────────────────────────────────


def _seed_workspace(root: Path, n_runs: int = 2) -> Workspace:
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


# ── Run portability ────────────────────────────────────────────────────────


class TestRunPortability:
    def test_move_run_dir_stays_queryable(self, tmp_path):
        """A run directory moved under a new workspace stays queryable via the
        authoritative manifests (scanner), no derived index to rebuild."""
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

        # The moved run's assets are found by scanning the destination manifests.
        found = scan.scan_assets(dst_ws.root)
        assert len(found) >= 3  # artifact + log + checkpoint


# ── Manifest <-> disk consistency ──────────────────────────────────────────


class TestManifestConsistency:
    def test_every_manifest_entry_points_to_existing_file(self, tmp_path):
        ws = _seed_workspace(tmp_path / "lab", n_runs=2)
        exp = ws.project("demo").experiment("baseline")
        for run in exp.list_runs():
            run_dir = Path(run.run_dir)
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
        run_id = ws.project("demo").experiment("baseline").list_runs()[0].id
        artifacts = scan.scan_assets(ws.root, kind="artifact")
        assert len(artifacts) == 1
        assert artifacts[0].producer is not None
        assert artifacts[0].producer.run_id == run_id
        assert artifacts[0].producer.execution_id is not None

    def test_task_id_set_via_set_active_task(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            ctx.set_active_task("train")
            asset = ctx.artifact.save("m.json", {"x": 1})
        assert asset.producer.task_id == "train"


# ── Concurrent writes within a run ─────────────────────────────────────────


class TestConcurrentWrites:
    def test_parallel_artifact_writes_all_registered(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Test")
        run = ws.add_project("p").add_experiment("e").add_run()
        N = 20
        with run.start() as ctx, ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(ctx.artifact.save, f"a{i}.json", {"i": i}) for i in range(N)]
            results = [f.result() for f in as_completed(futs)]

        assert len(results) == N
        scanned = scan.scan_assets(ws.root, kind="artifact", producer_run=run.id)
        assert len(scanned) == N
        manifest_assets = AssetManifest(Path(run.run_dir)).list()
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
            view_assets = run.assets.list()
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

        # Visible in the workspace view + the manifest scanner
        assert ws.assets.get(asset.asset_id) is not None
        assert scan.get_asset(ws.root, asset.asset_id) is not None
