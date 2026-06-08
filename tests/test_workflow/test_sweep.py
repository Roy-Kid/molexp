"""Tests for the SweepMap workflow primitive.

Under the pure-task-context contract SweepMap maps a callable over each cell of a
ParamSpace and RETURNS one record per cell (``{name, tags, result, mime}``); the
engine's materialization layer persists the returned list. Tests assert on the
returned records (``result.outputs[...]``), not on per-cell artifact.save calls.
"""

import asyncio

import molexp as me
from molexp.workflow import (
    SweepMap,
    WorkflowCompiler,
    WorkflowRuntime,
    default_binding_registry,
)
from molexp.workspace import GridSpace
from molexp.workspace.run import RunContext


def _execute(spec, tmp_path):
    ws = me.Workspace(tmp_path)
    proj = ws.add_project("p")
    exp = proj.add_experiment("e", params={})
    default_binding_registry.bind(exp, spec)
    run = exp.add_run({})
    with RunContext(run) as ctx:
        result = asyncio.run(WorkflowRuntime().execute(spec, run_context=ctx))
    catalog = run.experiment.project.workspace.catalog
    artifacts = catalog.query_assets(kind="artifact", producer_run=run.id)
    return result, artifacts


def test_sweep_map_returns_one_record_per_cell(tmp_path):
    space = GridSpace({"scheme": ["int8", "int4"], "dataset": ["qm9"]})  # 2 cells

    def fn(cell):
        return {"scheme": cell["scheme"], "dataset": cell["dataset"], "value": 1}

    wf = WorkflowCompiler(name="sweep")
    wf.add(SweepMap(fn, space), name="sweep_cells")
    spec = wf.compile()

    result, _artifacts = _execute(spec, tmp_path)

    assert result.status == "completed"
    records = result.outputs["sweep_cells"]
    assert len(records) == 2
    schemes = {r["tags"].get("scheme") for r in records}
    assert schemes == {"int8", "int4"}
    # every record is tagged with its cell's params (stringified)
    for r in records:
        assert r["tags"].get("dataset") == "qm9"
        assert "sweep_index" in r["tags"]


def test_sweep_map_name_fn_controls_filename_and_extension(tmp_path):
    space = GridSpace({"scheme": ["int8", "int4"]})  # 2 cells

    # A binary payload saved under a caller-chosen name + extension — the case
    # the default ``<prefix>-<i>.json`` naming cannot express.
    def fn(cell):
        return cell["scheme"].encode()

    wf = WorkflowCompiler(name="sweepnamed")
    wf.add(
        SweepMap(
            fn,
            space,
            name_fn=lambda cell, _i: f"trajectory_{cell['scheme']}.pt",
            mime="application/octet-stream",
        ),
        name="sweep_cells",
    )
    spec = wf.compile()

    result, _artifacts = _execute(spec, tmp_path)

    assert result.status == "completed"
    records = result.outputs["sweep_cells"]
    # name_fn drives the full record name + extension (not the default "<prefix>-<i>.json")
    assert {r["name"] for r in records} == {"trajectory_int8.pt", "trajectory_int4.pt"}
    assert all(r["mime"] == "application/octet-stream" for r in records)
    # cell params are still tagged so downstream reduces can key on them
    assert {r["tags"].get("scheme") for r in records} == {"int8", "int4"}


def test_sweep_map_empty_space_returns_no_records(tmp_path):
    space = GridSpace({"scheme": []})  # 0 cells

    wf = WorkflowCompiler(name="sweep0")
    wf.add(SweepMap(lambda _cell: {"x": 1}, space), name="sweep_cells")
    spec = wf.compile()

    result, _artifacts = _execute(spec, tmp_path)

    assert result.status == "completed"
    assert result.outputs["sweep_cells"] == []


def test_sweep_map_propagates_callable_error(tmp_path):
    space = GridSpace({"scheme": ["int8"]})

    def boom(cell):
        raise ValueError("cell failed")

    wf = WorkflowCompiler(name="sweepboom")
    wf.add(SweepMap(boom, space), name="sweep_cells")
    spec = wf.compile()

    # the error must surface, not be silently swallowed
    raised = False
    try:
        result, _ = _execute(spec, tmp_path)
    except Exception:
        raised = True
    else:
        raised = result.status != "completed"
    assert raised


def test_sweep_map_is_public_symbol():
    import molexp.workflow as wf_mod

    assert "SweepMap" in wf_mod.__all__
