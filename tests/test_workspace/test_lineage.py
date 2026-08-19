"""``molexp.workspace.assets.lineage`` — ancestors/descendants over the asset DAG.

Owns lineage traversal (``ancestors`` / ``descendants``) and the ``Producer.inputs``
edge data it walks (recorded via ``register_artifact(..., consumed=...)``). Asset
``content_hash`` correctness is owned by ``test_ids`` (``compute_content_hash``)
and ``test_asset_scan`` (``find_by_content_hash``), not here.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.assets import lineage


class TestProducerInputs:
    def test_consumed_inputs_recorded_on_artifact(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_bytes(b"raw\n")

        ws = Workspace(tmp_path / "lab", name="Lab")
        upstream = ws.data_assets.import_asset("input", src)

        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            mid = ctx.register_artifact({"x": 1}, name="mid.json", consumed=[upstream])
            final = ctx.register_artifact({"y": 2}, name="final.json", consumed=[upstream, mid])

        assert mid.producer.inputs == (upstream.asset_id,)
        assert final.producer.inputs == (upstream.asset_id, mid.asset_id)


class TestLineageTraversal:
    def test_ancestors_and_descendants_trace_three_step_dag(self, tmp_path):
        src = tmp_path / "raw.txt"
        src.write_bytes(b"raw\n")

        ws = Workspace(tmp_path / "lab", name="Lab")
        a = ws.data_assets.import_asset("a", src)

        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            b = ctx.register_artifact({"step": "b"}, name="b.json", consumed=[a])
            c = ctx.register_artifact({"step": "c"}, name="c.json", consumed=[b])

        assert lineage.ancestors(ws, c.asset_id) == {a.asset_id, b.asset_id}
        assert lineage.descendants(ws, a.asset_id) == {b.asset_id, c.asset_id}

    def test_self_loop_terminates(self, tmp_path):
        # Defensive: if a producer somehow lists its own asset_id, traversal
        # must terminate via the visited set.
        ws = Workspace(tmp_path / "lab", name="Lab")
        run = ws.add_project("p").add_experiment("e").add_run()
        with run.start() as ctx:
            asset = ctx.register_artifact({"x": 1}, name="solo.json")

        # Manually mutate the asset's producer to self-reference and persist.
        asset_loop = asset.model_copy(
            update={"producer": asset.producer.model_copy(update={"inputs": (asset.asset_id,)})}
        )
        from molexp.workspace.assets import AssetManifest

        AssetManifest(Path(run.run_dir)).update(asset_loop)

        assert lineage.ancestors(ws, asset.asset_id) == set()  # self-loop excluded
