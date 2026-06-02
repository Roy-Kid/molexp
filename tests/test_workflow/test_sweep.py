"""Tests for the SweepMap workflow primitive.

SweepMap maps a callable over each cell of a ParamSpace and persists one
ArtifactAsset per cell, tagged with that cell's parameters.
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


def test_sweep_map_writes_one_asset_per_cell(tmp_path):
    space = GridSpace({"scheme": ["int8", "int4"], "dataset": ["qm9"]})  # 2 cells

    def fn(cell):
        return {"scheme": cell["scheme"], "dataset": cell["dataset"], "value": 1}

    wf = WorkflowCompiler(name="sweep")
    wf.add(SweepMap(fn, space), name="sweep_cells")
    spec = wf.compile()

    result, artifacts = _execute(spec, tmp_path)

    assert result.status == "completed"
    assert len(artifacts) == 2
    schemes = {a.tags.get("scheme") for a in artifacts}
    assert schemes == {"int8", "int4"}
    # every asset is tagged with its cell's params (stringified)
    for a in artifacts:
        assert a.tags.get("dataset") == "qm9"
        assert "sweep_index" in a.tags


def test_sweep_map_empty_space_writes_nothing(tmp_path):
    space = GridSpace({"scheme": []})  # 0 cells

    wf = WorkflowCompiler(name="sweep0")
    wf.add(SweepMap(lambda _cell: {"x": 1}, space), name="sweep_cells")
    spec = wf.compile()

    result, artifacts = _execute(spec, tmp_path)

    assert result.status == "completed"
    assert len(artifacts) == 0


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
