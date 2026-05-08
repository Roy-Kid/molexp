"""Tests for workflow execution through the molexp scheduler.

After the rectification, the runtime takes an opaque duck-typed
``run_context`` payload and a ``Mapping[str, Any]`` config — never a
``Workspace.Run`` or ``ProfileConfig``. The legacy ``run=`` kwarg is
gone; ``run_dir=`` accepts a path directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import Task, TaskContext, Workflow


class _RunContextStub:
    """Minimal duck-typed ``run_context`` — what the runtime now requires.

    Exposes ``.work_dir`` / ``.config`` / ``.run`` so the runtime can
    extract a run dir and forward the value to ``TaskContext.run_context``.
    No ``Workspace`` import.
    """

    def __init__(self, *, work_dir: Path, config: dict | None = None, run_id: str | None = None):
        self.work_dir = work_dir
        self.config = config or {}
        self.run = type(
            "RunStub",
            (),
            {"id": run_id or "stub-run", "run_dir": work_dir},
        )()


@pytest.mark.asyncio
class TestFunctionalExecution:
    async def test_single_task(self):
        wf = Workflow(name="single")

        @wf.task
        async def double(ctx: TaskContext) -> int:
            return (ctx.inputs or 5) * 2

        result = await wf.build().execute()
        assert result.status == "completed"
        assert result.outputs["double"] == 10

    async def test_chain(self):
        wf = Workflow(name="chain")

        @wf.task
        async def step_a(ctx: TaskContext) -> int:
            return 10

        @wf.task(depends_on=["step_a"])
        async def step_b(ctx: TaskContext) -> int:
            return ctx.inputs + 5

        result = await wf.build().execute()
        assert result.outputs == {"step_a": 10, "step_b": 15}

    async def test_failure_propagates(self):
        wf = Workflow(name="fail")

        @wf.task
        async def boom(ctx):
            raise RuntimeError("oops")

        result = await wf.build().execute()
        assert result.status == "failed"

    async def test_run_context_is_forwarded_to_tasks(self, tmp_path):
        wf = Workflow(name="with-run-context")

        run_ctx = _RunContextStub(
            work_dir=tmp_path / "run",
            config={"epochs": 1, "dataset": "md17"},
        )

        @wf.task
        async def inspect(ctx: TaskContext) -> bool:
            assert ctx.run_context is run_ctx
            return True

        result = await wf.build().execute(run_context=run_ctx)
        assert result.status == "completed"
        assert result.outputs["inspect"] is True

    async def test_runtime_runs_with_duck_typed_run_context_no_workspace(self, tmp_path):
        """Critical: runtime drives a workflow with a stub run_context that
        has no Workspace ancestry whatsoever, and writes workflow.json under
        run_dir/executions/<execution_id>/."""
        wf = Workflow(name="duck")

        run_ctx = _RunContextStub(work_dir=tmp_path / "stub-run")

        @wf.task
        async def step(ctx: TaskContext) -> str:
            return "ok"

        result = await wf.build().execute(run_context=run_ctx)
        assert result.status == "completed"
        assert result.outputs["step"] == "ok"

        executions = run_ctx.work_dir / "executions"
        assert executions.exists(), "runtime must materialize executions/ under run_dir"
        # Find the workflow.json under the single execution.
        wf_jsons = list(executions.rglob("workflow.json"))
        assert wf_jsons, "workflow.json must be written under run_dir/executions/<id>/"

    async def test_legacy_run_kwarg_is_rejected(self, tmp_path):
        """The runtime no longer accepts ``run=``. Use ``run_dir=`` or
        ``run_context=``."""
        wf = Workflow(name="no-run-kwarg")

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(TypeError):
            await wf.build().execute(run=object())

    async def test_legacy_profile_config_kwarg_is_rejected(self, tmp_path):
        """The runtime no longer accepts ``profile_config=``. Use ``config=``."""
        from molexp.config import ProfileConfig

        wf = Workflow(name="no-profile-config-kwarg")

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(TypeError):
            await wf.build().execute(profile_config=ProfileConfig({}, name=None))

    async def test_run_dir_kwarg_writes_workflow_json(self, tmp_path):
        wf = Workflow(name="run-dir-only")

        @wf.task
        async def step(ctx: TaskContext) -> int:
            return 7

        result = await wf.build().execute(run_dir=tmp_path / "rd")
        assert result.status == "completed"
        wf_jsons = list((tmp_path / "rd" / "executions").rglob("workflow.json"))
        assert wf_jsons

    async def test_config_is_plain_mapping(self, tmp_path):
        wf = Workflow(name="plain-config")

        @wf.task
        async def inspect(ctx: TaskContext) -> int:
            return ctx.config["epochs"]

        result = await wf.build().execute(config={"epochs": 42})
        assert result.outputs["inspect"] == 42


@pytest.mark.asyncio
class TestOOPExecution:
    async def test_task_subclass(self):
        class AddTen(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return (ctx.inputs or 0) + 10

        spec = Workflow(name="oop").add(AddTen()).build()
        result = await spec.execute()
        assert result.outputs["add_ten"] == 10


@pytest.mark.asyncio
class TestProtocolExecution:
    async def test_external_runnable(self):
        class External:
            async def execute(self, ctx) -> int:
                return 99

        spec = Workflow(name="ext").add(External(), name="ext").build()
        result = await spec.execute()
        assert result.outputs["ext"] == 99

    async def test_mixed_task_types(self):
        class OOP(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 10

        class External:
            async def execute(self, ctx) -> int:
                return ctx.inputs + 90

        spec = (
            Workflow(name="mixed")
            .add(OOP(), name="oop")
            .add(External(), name="ext", depends_on=["oop"])
            .build()
        )
        result = await spec.execute()
        assert result.outputs == {"oop": 10, "ext": 100}


@pytest.mark.asyncio
class TestParallelExecution:
    async def test_independent_tasks_parallel(self):
        wf = Workflow(name="parallel")

        @wf.task
        async def a(ctx):
            return "a"

        @wf.task
        async def b(ctx):
            return "b"

        result = await wf.build().execute()
        assert result.outputs == {"a": "a", "b": "b"}

    async def test_diamond_dependency(self):
        wf = Workflow(name="diamond")

        @wf.task
        async def root(ctx):
            return 1

        @wf.task(depends_on=["root"])
        async def left(ctx):
            return ctx.inputs + 10

        @wf.task(depends_on=["root"])
        async def right(ctx):
            return ctx.inputs + 20

        @wf.task(depends_on=["left", "right"])
        async def merge(ctx):
            return ctx.inputs["left"] + ctx.inputs["right"]

        result = await wf.build().execute()
        assert result.outputs["merge"] == 32
