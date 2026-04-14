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
        from molexp.config import ProfileConfig

        wf = workflow(name="with-run-context")

        class RunContextStub:
            """Minimal stub — profile fixed at construction, no late-bind."""

            def __init__(self, *, config: ProfileConfig | None = None):
                self.run = None
                self.work_dir = tmp_path / "run"
                self.config = config or ProfileConfig({}, name=None)
                self.context = type("ContextStub", (), {"status": {}})()

            def _save_context(self) -> None:
                return None

        run_ctx = RunContextStub(config=ProfileConfig({"epochs": 1}, name="smoke"))

        @wf.task
        async def inspect(ctx: TaskContext) -> bool:
            assert ctx.run_context is run_ctx
            assert ctx.config.name == "smoke"
            assert ctx.config["epochs"] == 1
            return True

        result = await wf.build().execute(run_context=run_ctx)
        assert result.status == "completed"
        assert result.outputs["inspect"] is True

    async def test_run_context_with_profile_config_raises(self, tmp_path):
        """Passing run_context and profile_config is a hard error — no late-bind."""
        from molexp.config import ProfileConfig

        wf = workflow(name="no-late-bind")

        class RunContextStub:
            def __init__(self):
                self.run = None
                self.work_dir = tmp_path / "run"
                self.config = ProfileConfig({}, name=None)
                self.context = type("ContextStub", (), {"status": {}})()

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(ValueError, match="Cannot combine run_context"):
            await wf.build().execute(
                run_context=RunContextStub(),
                profile_config=ProfileConfig({"x": 1}, name="smoke"),
            )

    async def test_run_is_managed_by_runtime(self, tmp_path):
        from molexp.config import ProfileConfig

        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        project = workspace.project("demo")
        experiment = project.experiment("runtime")
        run = experiment.run(parameters={"seed": 42})

        wf = workflow(name="managed-run")

        @wf.task
        async def inspect(ctx: TaskContext) -> str:
            assert ctx.run_context is not None
            assert ctx.run_context.run is run
            assert ctx.run_context.config.name == "smoke"
            assert ctx.config["epochs"] == 1
            ctx.set_result("epochs", ctx.config["epochs"])
            return "ok"

        profile_cfg = ProfileConfig({"epochs": 1}, name="smoke")
        result = await wf.build().execute(run=run, profile_config=profile_cfg)

        assert result.status == "completed"
        assert result.outputs["inspect"] == "ok"
        # Profile is orthogonal to status — run succeeded.
        assert run.status == "succeeded"
        assert run.metadata.profile == "smoke"
        assert run.metadata.config["epochs"] == 1

        run_json = json.loads((run.run_dir / "run.json").read_text())
        assert run_json["context"]["results"]["epochs"] == 1

    async def test_default_profile_is_empty_config(self, tmp_path):
        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        project = workspace.project("demo")
        experiment = project.experiment("runtime")
        run = experiment.run(parameters={"seed": 42})

        wf = workflow(name="no-profile")

        @wf.task
        async def inspect(ctx: TaskContext) -> str:
            assert ctx.config.name is None
            assert len(ctx.config) == 0
            return "ok"

        result = await wf.build().execute(run=run)
        assert result.status == "completed"
        assert run.metadata.profile is None

    async def test_run_and_run_context_are_mutually_exclusive(self, tmp_path):
        workspace = Workspace(root=tmp_path / "lab", name="Test Lab")
        project = workspace.project("demo")
        experiment = project.experiment("runtime")
        run = experiment.run(parameters={})

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
