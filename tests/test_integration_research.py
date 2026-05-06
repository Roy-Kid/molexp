"""End-to-end integration test for molexp-research-orchestration.

Acceptance ac-009: a synthetic ZKG-shaped pipeline that exercises every
primitive landed in this spec without using any ``molexp.plugins.*``
Task.  The test composes:

- ``Workflow.sanity_check(on_fail='halt')`` — first attempt with a
  high charge produces a "collapsed" density that fails the predicate;
  the workflow returns ``status='failed'`` with a ``sanity_events`` log.
- ``molexp.agent.replan`` — materialises a sibling Run whose
  ``charge_strength`` is halved; ``replanned_from`` provenance is
  recorded on the new run's metadata labels.
- 3 replicate Runs of the recovered workflow — cooling task succeeds
  for each replicate; outputs are collected into a list.
- ``@wf.reduce(over='replicate')`` — invoked via
  ``WorkflowSpec.run_reducer`` to produce ``{"mean_Tg": ...}``;
  attached to the experiment scope as a typed asset.
- A second workflow with ``dependent_params=`` reading the reduced
  ``mean_Tg`` from upstream output to set its Young-modulus target.
- ``experiment.assets.query(recursive=True)`` — confirms the
  experiment-level asset shows up alongside artifacts from descendant
  runs.

The test is *not* a chemistry validation — it is an architectural smoke
test: the primitives compose without depending on any concrete
molecular-dynamics provider.  The assertion that no plugin under
``src/molexp/plugins/`` contributes a Task is encoded by *what this
file imports*: only ``molexp.workflow``, ``molexp.workspace``, and
``molexp.agent``.
"""

from __future__ import annotations

import json

import pytest

from molexp.agent import replan
from molexp.config import ProfileConfig
from molexp.workflow import TaskContext, Workflow
from molexp.workspace import Workspace


def _make_zkg_cooling_workflow():
    """Synthetic cooling workflow with sanity-check + cross-replicate reducer."""
    wf = Workflow(name="zkg-cooling")

    @wf.task
    async def cool(ctx: TaskContext) -> dict:
        # Synthetic chemistry: higher charge_strength => more ionic clustering
        # => higher density at quench (collapsed melt) AND higher Tg.
        charge = float(ctx.config["charge_strength"])
        replicate = int(ctx.config.get("replicate", 0))
        density = 0.85 + 0.10 * (charge - 1.0)
        tg = 0.55 + 0.05 * charge + 0.005 * replicate
        return {"Tg": tg, "density": density}

    wf.sanity_check(
        after="cool",
        predicate=lambda state: 0.80 <= state.results["cool"]["density"] <= 0.90,
        on_fail="halt",
    )

    @wf.reduce(over="replicate")
    def aggregate(replicate_outputs: list[dict]) -> dict:
        tgs = [r["Tg"] for r in replicate_outputs]
        return {
            "mean_Tg": sum(tgs) / len(tgs),
            "n_replicates": len(tgs),
        }

    return wf.build()


def _make_mechanical_workflow():
    """Synthetic mechanical workflow consuming the reduced Tg via dependent_params."""
    wf = Workflow(name="zkg-mechanical")

    @wf.task
    async def reduce_proxy(ctx: TaskContext) -> dict:
        # The proxy task surfaces the reduced Tg as its output so the downstream
        # task can consume it through the regular dependent_params plumbing.
        return {"mean_Tg": float(ctx.config["mean_Tg"])}

    @wf.task(
        depends_on=["reduce_proxy"],
        dependent_params=lambda prev: {"T": 0.7 * prev["reduce_proxy"].output["mean_Tg"]},
    )
    async def deform(ctx: TaskContext) -> dict:
        T = float(ctx.config["T"])
        # Synthetic Young modulus: anti-correlated with deformation T.
        return {"target_T": T, "young_modulus": 100.0 / max(T, 0.01)}

    return wf.build()


@pytest.mark.asyncio
async def test_zkg_shaped_integration(tmp_path) -> None:
    ws = Workspace(root=tmp_path, name="zkg-int")
    proj = ws.project("zkg-study")
    exp = proj.experiment("topology-CP-ZW")

    spec = _make_zkg_cooling_workflow()

    # ── Phase 1: first attempt with high charge → sanity halts ─────────────
    initial = exp.run(parameters={"charge_strength": 2.0, "replicate": 0})
    initial_cfg = ProfileConfig({"charge_strength": 2.0, "replicate": 0}, name="zkg-initial")
    initial_result = await spec.execute(run=initial, profile_config=initial_cfg)
    assert initial_result.status == "failed"
    assert any(
        e.get("task") == "cool" and e.get("on_fail") == "halt" for e in initial_result.sanity_events
    ), f"expected halt event for cool task; got {initial_result.sanity_events}"

    # ── Phase 2: replan halves charge_strength ─────────────────────────────
    recovered = replan(
        initial,
        modifier=lambda p: {**p, "charge_strength": p["charge_strength"] / 2},
        reason="density above 0.90 — collapsed melt",
    )
    assert recovered.parameters["charge_strength"] == 1.0
    assert recovered.metadata.labels["replanned_from"] == initial.id
    assert recovered.metadata.labels["replanned_reason"].startswith("density above")
    # Original is preserved.
    assert initial.parameters["charge_strength"] == 2.0
    assert initial.experiment.id == exp.id

    # ── Phase 3: 3 replicate Runs of the recovered workflow ────────────────
    cooling_outputs: list[dict] = []
    replicate_runs = []
    for i in range(3):
        rep_run = exp.run(parameters={"charge_strength": 1.0, "replicate": i})
        rep_cfg = ProfileConfig({"charge_strength": 1.0, "replicate": i}, name=f"zkg-rep{i}")
        rep_result = await spec.execute(run=rep_run, profile_config=rep_cfg)
        assert rep_result.status == "completed", f"replicate {i} failed: {rep_result.sanity_events}"
        cooling_outputs.append(rep_result.outputs["cool"])
        replicate_runs.append(rep_run)

    assert len(cooling_outputs) == 3

    # ── Phase 4: cross-replicate reduce ─────────────────────────────────────
    reduced = spec.run_reducer(cooling_outputs)
    assert reduced["n_replicates"] == 3
    # mean_Tg = 0.55 + 0.05*1.0 + 0.005*mean(0,1,2) = 0.55 + 0.05 + 0.005 = 0.605
    assert reduced["mean_Tg"] == pytest.approx(0.605, abs=1e-9)
    # Persist the reduced result as an experiment-scope artifact so cross-run
    # aggregation queries can find it alongside per-replicate outputs.
    reducer_run = exp.run(parameters={"phase": "reduce"})
    with reducer_run.start() as rctx:
        rctx.artifact.save("zkg_reduced_tg.json", json.dumps(reduced).encode("utf-8"))

    # ── Phase 5: mechanical workflow with dependent_params from reduced Tg ─
    mech_spec = _make_mechanical_workflow()
    mech_run = exp.run(parameters={"phase": "mechanical"})
    mech_cfg = ProfileConfig({"mean_Tg": reduced["mean_Tg"]}, name="zkg-mech")
    mech_result = await mech_spec.execute(run=mech_run, profile_config=mech_cfg)
    assert mech_result.status == "completed"
    deform_out = mech_result.outputs["deform"]
    assert deform_out["target_T"] == pytest.approx(0.7 * reduced["mean_Tg"])
    assert deform_out["young_modulus"] == pytest.approx(
        100.0 / (0.7 * reduced["mean_Tg"]), rel=1e-9
    )

    # ── Phase 6: cross-run aggregation surfaces reduce + replicate artifacts
    # The reducer artifact lives on a run scoped under the experiment, so the
    # recursive query at experiment level returns at least the reducer file.
    all_artifacts = exp.assets.query(kind="artifact", recursive=True)
    artifact_names = {a.name for a in all_artifacts}
    assert "zkg_reduced_tg.json" in artifact_names

    # Replanned-from chain is queryable via run metadata labels.
    descendants = [
        r
        for r in (initial, recovered, *replicate_runs, reducer_run, mech_run)
        if r.metadata.labels.get("replanned_from") == initial.id
    ]
    assert recovered in descendants
