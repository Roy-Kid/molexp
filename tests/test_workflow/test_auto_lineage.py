"""Engine-automatic artifact lineage (vision-loop-09).

THE FAIR anchor: a tracked run's workflow DAG projects into
``Producer.inputs`` with ZERO task-author annotation — "which input produced
this result?" is answerable for every tracked run through the existing
``lineage.ancestors`` / ``GET /assets/{id}/lineage`` surfaces.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workflow import TaskContext, WorkflowCompiler
from molexp.workspace import Workspace
from molexp.workspace.assets import lineage, scan


def _workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "lab", name="lab")
    ws.materialize()
    return ws


def _artifact_for(ws: Workspace, run_id: str, task: str):
    found = scan.scan_assets(ws.root, kind="artifact", producer_run=run_id, producer_task=task)
    assert found, f"no artifact registered for task {task!r}"
    return found[0]


def _chain_workflow() -> WorkflowCompiler:
    wf = WorkflowCompiler(name="chain")

    @wf.task
    async def upstream(ctx: TaskContext) -> dict:
        return {"value": 21}

    @wf.task(depends_on=["upstream"])
    async def downstream(value: int) -> dict:
        return {"doubled": value * 2}

    return wf


def test_two_task_run_projects_the_dag_with_zero_annotation(tmp_path: Path) -> None:
    """THE test: downstream's producer.inputs carries upstream's asset_id."""
    ws = _workspace(tmp_path)
    run = ws.add_project("p").add_experiment("e").add_run(params=None)
    run.execute(_chain_workflow())

    up = _artifact_for(ws, run.id, "upstream")
    down = _artifact_for(ws, run.id, "downstream")
    assert up.asset_id in (down.producer.inputs if down.producer else ())
    # ...and the existing traversal answers it.
    assert up.asset_id in lineage.ancestors(ws, down.asset_id)


def test_root_task_artifact_has_empty_inputs(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    run = ws.add_project("p").add_experiment("e").add_run(params=None)
    run.execute(_chain_workflow())
    up = _artifact_for(ws, run.id, "upstream")
    assert up.producer is not None
    assert up.producer.inputs == ()


def test_diamond_fan_in_records_both_upstreams_deduped(tmp_path: Path) -> None:
    wf = WorkflowCompiler(name="diamond")

    @wf.task
    async def left(ctx: TaskContext) -> dict:
        return {"l": 1}

    @wf.task
    async def right(ctx: TaskContext) -> dict:
        return {"r": 2}

    @wf.task(depends_on=["left", "right"])
    async def join(l: int, r: int) -> dict:  # noqa: E741 — mirrors the upstream keys
        return {"sum": l + r}

    ws = _workspace(tmp_path)
    run = ws.add_project("p").add_experiment("e").add_run(params=None)
    run.execute(wf)

    join_asset = _artifact_for(ws, run.id, "join")
    inputs = set(join_asset.producer.inputs if join_asset.producer else ())
    left_id = _artifact_for(ws, run.id, "left").asset_id
    right_id = _artifact_for(ws, run.id, "right").asset_id
    assert {left_id, right_id} <= inputs
    assert len(join_asset.producer.inputs) == len(inputs)  # deduped
    # Independent parallel tasks record no cross-edges.
    left_asset = _artifact_for(ws, run.id, "left")
    assert right_id not in (left_asset.producer.inputs if left_asset.producer else ())


def test_cache_hit_keeps_the_chain(tmp_path: Path) -> None:
    """A second run whose upstream cache-hits still chains downstream→upstream."""
    ws = _workspace(tmp_path)
    exp = ws.add_project("p").add_experiment("e")

    run1 = exp.add_run(params=None, id="first")
    run1.execute(_chain_workflow())

    run2 = exp.add_run(params=None, id="second")
    run2.execute(_chain_workflow())

    down2 = _artifact_for(ws, "second", "downstream")
    up2 = _artifact_for(ws, "second", "upstream")
    assert up2.asset_id in (down2.producer.inputs if down2.producer else ())
    assert up2.asset_id in lineage.ancestors(ws, down2.asset_id)


def test_accessor_save_accepts_raw_id_strings(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    run = ws.add_project("p").add_experiment("e").add_run(params=None)
    with run.start() as ctx:
        first = ctx.artifact.save("a.json", {"x": 1})
        second = ctx.artifact.save("b.json", {"y": 2}, consumed=[first.asset_id])
    assert second.producer is not None
    assert second.producer.inputs == (first.asset_id,)


def test_raising_lineage_computation_never_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    """Fail-soft: lineage rides the persistence bonus channel — a raising
    upstream-id computation degrades (persist skipped, debug log) and the
    run still succeeds with its results intact."""
    from molexp.workflow._engine import node_cache

    def _boom(deps, registration):
        raise RuntimeError("lineage computation broken")

    monkeypatch.setattr(node_cache, "_upstream_asset_ids", _boom)
    ws = _workspace(tmp_path)
    run = ws.add_project("p").add_experiment("e").add_run(params=None)
    result = run.execute(_chain_workflow())
    assert result.status == "succeeded"
    assert result.outputs["downstream"] == {"doubled": 42}
