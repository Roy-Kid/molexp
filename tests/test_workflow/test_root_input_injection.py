"""Root-input injection — sweep params + content-addressed workdir as ctx.inputs.

Covers pure-task-context-02 ac-006 / ac-007: the engine injects a root task's
inputs as ``{"params": <run params>, "workdir": <Path>}``, and the injected
workdir is a bare, NON-navigable ``pathlib.Path`` (the RED LINE — no ambient
authority creeps back in).
"""

from __future__ import annotations

import pathlib

import pytest

from molexp.workflow import WorkflowCompiler, WorkflowRuntime
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace


def _new_run(tmp_path: pathlib.Path, params: dict):
    ws = Workspace(tmp_path / "lab")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params=params)


@pytest.mark.asyncio
async def test_root_task_receives_params_via_ctx_inputs(tmp_path: pathlib.Path) -> None:
    run = _new_run(tmp_path, {"mode": "block", "ratio": "r1", "n_litfsi": 54})
    captured: dict[str, object] = {}

    wf = WorkflowCompiler(name="rootinj")

    @wf.task
    async def root(ctx: TaskContext) -> str:
        captured["inputs"] = ctx.inputs
        return "ok"

    compiled = wf.compile()
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)

    assert result.status == "completed"
    injected = captured["inputs"]
    assert isinstance(injected, dict)
    assert injected["params"]["mode"] == "block"
    assert injected["params"]["n_litfsi"] == 54


@pytest.mark.asyncio
async def test_injected_workdir_is_non_navigable_path(tmp_path: pathlib.Path) -> None:
    run = _new_run(tmp_path, {"mode": "alt"})
    captured: dict[str, object] = {}

    wf = WorkflowCompiler(name="rootwd")

    @wf.task
    async def root(ctx: TaskContext) -> str:
        captured["workdir"] = ctx.inputs["workdir"]
        return "ok"

    compiled = wf.compile()
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)

    assert result.status == "completed"
    workdir = captured["workdir"]
    # RED LINE: a bare Path, NOT a navigable handle (no .folder()/.artifact/etc.).
    assert isinstance(workdir, pathlib.Path)
    assert not hasattr(workdir, "folder")
    assert not hasattr(workdir, "artifact")
    assert not hasattr(workdir, "run")


@pytest.mark.asyncio
async def test_every_task_gets_ctx_workdir(tmp_path: pathlib.Path) -> None:
    """``ctx.workdir`` is a content-addressed Path on EVERY task (not just roots),
    distinct per task, and matches a root task's ``ctx.inputs['workdir']``."""
    run = _new_run(tmp_path, {"mode": "alt"})
    captured: dict[str, object] = {}

    wf = WorkflowCompiler(name="wd-all")

    @wf.task
    async def root(ctx: TaskContext) -> int:
        captured["root_workdir"] = ctx.workdir
        captured["root_inputs_workdir"] = ctx.inputs["workdir"]
        return 1

    @wf.task(depends_on=["root"])
    async def child(ctx: TaskContext) -> int:
        captured["child_workdir"] = ctx.workdir
        return 2

    compiled = wf.compile()
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)

    assert result.status == "completed"
    # Non-root tasks now also receive a workdir (the whole point of ctx.workdir).
    assert isinstance(captured["root_workdir"], pathlib.Path)
    assert isinstance(captured["child_workdir"], pathlib.Path)
    # Root's ctx.workdir is consistent with the legacy ctx.inputs['workdir'].
    assert captured["root_workdir"] == captured["root_inputs_workdir"]
    # Content-addressed → distinct per task.
    assert captured["root_workdir"] != captured["child_workdir"]


@pytest.mark.asyncio
async def test_subworkflow_entry_merges_element_with_params_and_workdir(
    tmp_path: pathlib.Path,
) -> None:
    """A SubWorkflow forwarding a dict element into a workspace inner workflow
    MERGES it with the engine-injected ``{params, workdir}`` (element keys win),
    so the inner entry sees the element AND keeps params / workdir."""
    from molexp.workflow import SubWorkflow

    run = _new_run(tmp_path, {"mode": "blk"})
    captured: dict[str, object] = {}

    inner = WorkflowCompiler(name="inner-merge")

    @inner.task
    async def entry(ctx: TaskContext) -> str:
        captured["entry_inputs"] = ctx.inputs
        captured["entry_workdir"] = ctx.workdir
        return "ok"

    outer = WorkflowCompiler(name="outer-merge", entry="emit")

    @outer.task
    async def emit(ctx: TaskContext) -> list[dict]:
        return [{"label": "CAT", "smiles": "C"}]

    outer.add(SubWorkflow(inner), name="sub")

    @outer.task
    async def collect(ctx: TaskContext) -> list[str]:
        return list(ctx.inputs)

    outer.parallel(map_over="emit", body="sub", join="collect", max_concurrency=1)

    with run.start() as ctx:
        result = await WorkflowRuntime().execute(outer.compile(), run_context=ctx)

    assert result.status == "completed"
    merged = captured["entry_inputs"]
    assert isinstance(merged, dict)
    # Forwarded element keys are present...
    assert merged["label"] == "CAT"
    assert merged["smiles"] == "C"
    # ...and the engine-injected params / workdir survive the merge.
    assert merged["params"]["mode"] == "blk"
    assert isinstance(merged["workdir"], pathlib.Path)
    # ctx.workdir is independently available on the inner entry too.
    assert isinstance(captured["entry_workdir"], pathlib.Path)
