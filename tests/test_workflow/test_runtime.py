"""Tests for workflow execution through pydantic-graph runtime."""

import json

import pytest

from molexp.workflow import Task, TaskContext, WorkflowBuilder, workflow
from molexp.workspace import Workspace


@pytest.mark.asyncio
class TestFunctionalExecution:
    async def test_single_task(self):
        wf = workflow(name="single")

        @wf.task
        async def double(ctx: TaskContext) -> int:
            return (ctx.inputs or 5) * 2

        result = await wf.build().execute()
        assert result.status == "completed"
        assert result.outputs["double"] == 10

    async def test_chain(self):
        wf = workflow(name="chain")

        @wf.task
        async def step_a(ctx: TaskContext) -> int:
            return 10

        @wf.task(depends_on=["step_a"])
        async def step_b(ctx: TaskContext) -> int:
            return ctx.inputs + 5

        result = await wf.build().execute()
        assert result.outputs == {"step_a": 10, "step_b": 15}

    async def test_failure_propagates(self):
        wf = workflow(name="fail")

        @wf.task
        async def boom(ctx):
            raise RuntimeError("oops")

        result = await wf.build().execute()
        assert result.status == "failed"

    async def test_run_context_is_forwarded_to_tasks(self, tmp_path):
        wf = workflow(name="with-run-context")

        class RunContextStub:
            """Minimal stub — dry_run fixed at construction, no late-bind."""

            def __init__(self, *, dry_run: bool = False):
                self.run = None
                self.work_dir = tmp_path / "run"
                self.dry_run = dry_run
                self.context = type("ContextStub", (), {"status": {}})()

            def _save_context(self) -> None:
                return None

        run_ctx = RunContextStub(dry_run=True)

        @wf.task
        async def inspect(ctx: TaskContext) -> bool:
            assert ctx.run_context is run_ctx
            assert ctx.dry_run is True
            return True

        result = await wf.build().execute(run_context=run_ctx)
        assert result.status == "completed"
        assert result.outputs["inspect"] is True
        assert run_ctx.dry_run is True

    async def test_run_context_with_dry_run_raises(self, tmp_path):
        """Passing run_context and dry_run=True is a hard error — no late-bind."""
        wf = workflow(name="no-late-bind")

        class RunContextStub:
            def __init__(self):
                self.run = None
                self.work_dir = tmp_path / "run"
                self.dry_run = False
                self.context = type("ContextStub", (), {"status": {}})()

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(ValueError, match="Cannot combine run_context"):
            await wf.build().execute(run_context=RunContextStub(), dry_run=True)

    async def test_run_is_managed_by_runtime(self, tmp_path):
        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        workspace.materialize()
        project = workspace.create_project(name="demo")
        experiment = project.create_experiment(name="runtime")
        run = experiment.create_run(parameters={"seed": 42})

        wf = workflow(name="managed-run")

        @wf.task
        async def inspect(ctx: TaskContext) -> str:
            assert ctx.run_context is not None
            assert ctx.run_context.run is run
            assert ctx.run_context.dry_run is True
            assert ctx.dry_run is True
            ctx.set_result("mode", "dry")
            return "ok"

        result = await wf.build().execute(run=run, dry_run=True)

        assert result.status == "completed"
        assert result.outputs["inspect"] == "ok"
        assert run.status == "dry_run"
        assert run.metadata.dry_run is True
        assert run.metadata.labels["mode"] == "dry-run"

        run_json = json.loads((run.run_dir / "run.json").read_text())
        assert run_json["context"]["results"]["mode"] == "dry"

    async def test_normal_run_clears_stale_dry_run_badge(self, tmp_path):
        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        workspace.materialize()
        project = workspace.create_project(name="demo")
        experiment = project.create_experiment(name="runtime")
        run = experiment.create_run(parameters={"seed": 42})

        wf = workflow(name="badge-clear")

        @wf.task
        async def inspect(ctx: TaskContext) -> str:
            return "dry" if ctx.dry_run else "wet"

        spec = wf.build()
        first = await spec.execute(run=run, dry_run=True)
        second = await spec.execute(run=run, dry_run=False)

        assert first.status == "completed"
        assert second.status == "completed"
        assert run.metadata.dry_run is False
        assert "mode" not in run.metadata.labels

    async def test_run_and_run_context_are_mutually_exclusive(self, tmp_path):
        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        workspace.materialize()
        project = workspace.create_project(name="demo")
        experiment = project.create_experiment(name="runtime")
        run = experiment.create_run(parameters={})

        wf = workflow(name="exclusive")

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with run.start() as ctx:
            with pytest.raises(ValueError, match="either run or run_context"):
                await wf.build().execute(run=run, run_context=ctx)


@pytest.mark.asyncio
class TestOOPExecution:
    async def test_task_subclass(self):
        class AddTen(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return (ctx.inputs or 0) + 10

        spec = WorkflowBuilder(name="oop").add(AddTen()).build()
        result = await spec.execute()
        assert result.outputs["add_ten"] == 10


@pytest.mark.asyncio
class TestProtocolExecution:
    async def test_external_runnable(self):
        class External:
            async def execute(self, ctx) -> int:
                return 99

        spec = WorkflowBuilder(name="ext").add(External(), name="ext").build()
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
            WorkflowBuilder(name="mixed")
            .add(OOP(), name="oop")
            .add(External(), name="ext", depends_on=["oop"])
            .build()
        )
        result = await spec.execute()
        assert result.outputs == {"oop": 10, "ext": 100}


@pytest.mark.asyncio
class TestParallelExecution:
    async def test_independent_tasks_parallel(self):
        wf = workflow(name="parallel")

        @wf.task
        async def a(ctx):
            return "a"

        @wf.task
        async def b(ctx):
            return "b"

        result = await wf.build().execute()
        assert result.outputs == {"a": "a", "b": "b"}

    async def test_diamond_dependency(self):
        wf = workflow(name="diamond")

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
