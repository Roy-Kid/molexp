"""Root-input injection — sweep params + content-addressed workdir as ctx inputs.

Covers pure-task-context-02 ac-006 / ac-007: the engine injects a root task's
inputs as ``{"params": <run params>, "workdir": <Path>}``, ``ctx.workdir`` is a
bare NON-navigable ``pathlib.Path`` on every task (the RED LINE — no ambient
authority creeps back in), and a SubWorkflow forwarding an element merges it
with the injected envelope (element keys win).
"""

from __future__ import annotations

import pathlib

import pytest

from molexp.workflow import SubWorkflow, WorkflowCompiler, WorkflowRuntime
from molexp.workflow.context import TaskContext
from molexp.workspace import Workspace


def _new_run(tmp_path: pathlib.Path, params: dict):
    ws = Workspace(tmp_path / "lab")
    project = ws.add_project(name="p")
    experiment = project.add_experiment(name="e")
    return experiment.add_run(params=params)


class TestRootInputInjection:
    @pytest.mark.asyncio
    async def test_root_task_binds_run_params_by_name(self, tmp_path: pathlib.Path) -> None:
        """The engine unwraps the ``{params, workdir}`` envelope and binds a root
        task's run params by name; ``**params`` absorbs the rest."""
        run = _new_run(tmp_path, {"mode": "block", "ratio": "r1", "n_litfsi": 54})
        captured: dict[str, object] = {}

        wf = WorkflowCompiler(name="rootinj")

        @wf.task
        async def root(ctx: TaskContext, mode: str, n_litfsi: int, **params: object) -> str:
            captured["mode"] = mode
            captured["n_litfsi"] = n_litfsi
            captured["params"] = params
            return "ok"

        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.status == "succeeded"
        assert captured["mode"] == "block"
        assert captured["n_litfsi"] == 54
        assert captured["params"]["ratio"] == "r1"  # rest absorbed by **params

    @pytest.mark.asyncio
    async def test_injected_workdir_is_non_navigable_path(self, tmp_path: pathlib.Path) -> None:
        """RED LINE: ``ctx.workdir`` is a bare ``pathlib.Path``, NOT a navigable
        handle (no ``.folder`` / ``.artifact`` / ``.run`` ambient authority)."""
        run = _new_run(tmp_path, {"mode": "alt"})
        captured: dict[str, object] = {}

        wf = WorkflowCompiler(name="rootwd")

        @wf.task
        async def root(ctx: TaskContext) -> str:
            captured["workdir"] = ctx.workdir
            return "ok"

        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.status == "succeeded"
        workdir = captured["workdir"]
        assert isinstance(workdir, pathlib.Path)
        assert not hasattr(workdir, "folder")
        assert not hasattr(workdir, "artifact")
        assert not hasattr(workdir, "run")

    @pytest.mark.asyncio
    async def test_every_task_gets_a_distinct_content_addressed_workdir(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``ctx.workdir`` is present on EVERY task (not just roots) and is
        content-addressed, so it is distinct per task."""
        run = _new_run(tmp_path, {"mode": "alt"})
        captured: dict[str, object] = {}

        wf = WorkflowCompiler(name="wd-all")

        @wf.task
        async def root(ctx: TaskContext) -> int:
            captured["root_workdir"] = ctx.workdir
            return 1

        @wf.task(depends_on=["root"])
        async def child(ctx: TaskContext, root: int) -> int:
            captured["child_workdir"] = ctx.workdir
            return 2

        with run.start() as ctx:
            result = await WorkflowRuntime().execute(wf.compile(), run_context=ctx)

        assert result.status == "succeeded"
        assert isinstance(captured["root_workdir"], pathlib.Path)
        assert isinstance(captured["child_workdir"], pathlib.Path)
        assert captured["root_workdir"] != captured["child_workdir"]

    @pytest.mark.asyncio
    async def test_subworkflow_entry_merges_element_with_injected_envelope(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A SubWorkflow forwarding a dict element MERGES it with the injected
        ``{params, workdir}`` (element keys win), so the inner entry sees the
        element AND keeps params / workdir."""
        run = _new_run(tmp_path, {"mode": "blk"})
        captured: dict[str, object] = {}

        inner = WorkflowCompiler(name="inner-merge")

        @inner.task
        async def entry(
            ctx: TaskContext, label: str, smiles: str, params: dict, **rest: object
        ) -> str:
            captured["entry_inputs"] = {
                "label": label,
                "smiles": smiles,
                "params": params,
                **rest,
            }
            captured["entry_workdir"] = ctx.workdir
            return "ok"

        outer = WorkflowCompiler(name="outer-merge", entry="emit")

        @outer.task
        async def emit(ctx: TaskContext) -> list[dict]:
            return [{"label": "CAT", "smiles": "C"}]

        outer.add(SubWorkflow(inner), name="sub")

        @outer.task
        async def collect(items: list[str]) -> list[str]:
            return list(items)

        outer.parallel(map_over="emit", body="sub", join="collect", max_concurrency=1)

        with run.start() as ctx:
            result = await WorkflowRuntime().execute(outer.compile(), run_context=ctx)

        assert result.status == "succeeded"
        merged = captured["entry_inputs"]
        assert isinstance(merged, dict)
        # Forwarded element keys are present...
        assert merged["label"] == "CAT"
        assert merged["smiles"] == "C"
        # ...and the engine-injected params / workdir survive the merge.
        assert merged["params"]["mode"] == "blk"
        assert isinstance(merged["workdir"], pathlib.Path)
        assert isinstance(captured["entry_workdir"], pathlib.Path)
