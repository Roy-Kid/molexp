"""Tests for asset content_hash + Producer.inputs lineage (spec: core-versioning).

Covers acceptance criteria:
- ac-006: register_data(consumed=...) records Producer.inputs
- ac-007: Asset.content_hash matches sha256 of payload
- ac-008: lineage.ancestors / descendants trace a 3-step DAG
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import lineage

KNOWN_BYTES = b"hello molexp\n"
KNOWN_SHA256 = "sha256:" + hashlib.sha256(KNOWN_BYTES).hexdigest()


class TestContentHash:
    def test_data_asset_content_hash_matches_payload(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_bytes(KNOWN_BYTES)

        ws = Workspace(tmp_path / "lab", name="Lab")
        asset = ws.data_assets.import_asset("greeting", src)

        assert asset.content_hash == KNOWN_SHA256

    def test_artifact_asset_content_hash_present(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            asset = ctx.artifact.save("payload.bin", KNOWN_BYTES)

        assert asset.content_hash == KNOWN_SHA256

    def test_log_asset_content_hash_is_none(self, tmp_path):
        ws = Workspace(tmp_path / "lab", name="Lab")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            ctx.log("train").append("hi")
            log_assets = list(Path(run.experiment.list_runs()[0].run_dir).iterdir())
            assert log_assets  # placeholder — actual log retrieval below
            from molexp.workspace.assets import LogAsset, scan

            manifest_assets = scan.scan_assets(ws.root, kind="log", producer_run=run.id)
            assert len(manifest_assets) >= 1
            assert all(isinstance(a, LogAsset) for a in manifest_assets)
            assert all(a.content_hash is None for a in manifest_assets)


class TestProducerInputs:
    def test_consumed_inputs_recorded_on_artifact(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_bytes(b"raw\n")

        ws = Workspace(tmp_path / "lab", name="Lab")
        upstream = ws.data_assets.import_asset("input", src)

        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            mid = ctx.artifact.save("mid.json", {"x": 1}, consumed=[upstream])
            final = ctx.artifact.save("final.json", {"y": 2}, consumed=[upstream, mid])

        assert mid.producer.inputs == (upstream.asset_id,)
        assert final.producer.inputs == (upstream.asset_id, mid.asset_id)


class TestLineageTraversal:
    def test_ancestors_three_step(self, tmp_path):
        src = tmp_path / "raw.txt"
        src.write_bytes(b"raw\n")

        ws = Workspace(tmp_path / "lab", name="Lab")
        a = ws.data_assets.import_asset("a", src)

        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            b = ctx.artifact.save("b.json", {"step": "b"}, consumed=[a])
            c = ctx.artifact.save("c.json", {"step": "c"}, consumed=[b])

        anc = lineage.ancestors(ws, c.asset_id)
        assert anc == {a.asset_id, b.asset_id}

        desc = lineage.descendants(ws, a.asset_id)
        assert desc == {b.asset_id, c.asset_id}

    def test_self_loop_terminates(self, tmp_path):
        # Defensive: if a producer somehow lists its own asset_id, traversal
        # must terminate via the visited set.
        ws = Workspace(tmp_path / "lab", name="Lab")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            asset = ctx.artifact.save("solo.json", {"x": 1})

        # Manually mutate the asset's producer to self-reference and persist.
        asset_loop = asset.model_copy(
            update={"producer": asset.producer.model_copy(update={"inputs": (asset.asset_id,)})}
        )
        from molexp.workspace.assets import AssetManifest

        AssetManifest(Path(run.run_dir)).update(asset_loop)

        anc = lineage.ancestors(ws, asset.asset_id)
        assert anc == set()  # self-loop excluded from ancestors of self
