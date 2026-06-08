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
    return experiment.add_run(parameters=params)


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
