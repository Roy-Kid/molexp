"""Tests for workflow execution through the molexp scheduler.

After the rectification, the runtime takes an opaque duck-typed
``run_context`` payload and a ``Mapping[str, Any]`` config — never a
``Workspace.Run`` or ``ProfileConfig``. The legacy ``run=`` kwarg is
gone; ``run_dir=`` accepts a path directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workflow import Task, TaskContext, WorkflowCompiler, WorkflowRuntime


class _RunContextStub:
    """Minimal duck-typed ``run_context`` — what the runtime now requires.

    Exposes ``.work_dir`` / ``.config`` / ``.run`` so the runtime can
    extract a run dir and forward the value to ``TaskContext.run_context``.
    No ``Workspace`` import.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        config: dict | None = None,
        run_id: str | None = None,
        params: dict | None = None,
    ):
        self.work_dir = work_dir
        self.config = config or {}
        # Root-task params reach the body by name (engine reads run_context.params).
        self.params = params or {}
        self.run = type(
            "RunStub",
            (),
            {"id": run_id or "stub-run", "run_dir": work_dir},
        )()


@pytest.mark.asyncio
class TestFunctionalExecution:
    async def test_single_task(self):
        wf = WorkflowCompiler(name="single")

        @wf.task
        async def double(x: int = 5) -> int:
            return x * 2

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "completed"
        assert result.outputs["double"] == 10

    async def test_chain(self):
        wf = WorkflowCompiler(name="chain")

        @wf.task
        async def step_a(ctx: TaskContext) -> int:
            return 10

        @wf.task(depends_on=["step_a"])
        async def step_b(step_a: int) -> int:
            return step_a + 5

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs == {"step_a": 10, "step_b": 15}

    async def test_failure_propagates(self):
        wf = WorkflowCompiler(name="fail")

        @wf.task
        async def boom(ctx):
            raise RuntimeError("oops")

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "failed"

    async def test_run_context_is_not_exposed_on_task_ctx(self, tmp_path):
        # Pure-task-context contract: run_context is NOT forwarded to the public
        # TaskContext. A task accessing ctx.run_context raises AttributeError; the
        # engine still drives the run via its private channel.
        wf = WorkflowCompiler(name="no-run-context")

        run_ctx = _RunContextStub(
            work_dir=tmp_path / "run",
            config={"epochs": 1, "dataset": "md17"},
        )

        @wf.task
        async def inspect(ctx: TaskContext) -> bool:
            assert not hasattr(ctx, "run_context")
            return True

        result = await WorkflowRuntime().execute(wf.compile(), run_context=run_ctx)
        assert result.status == "completed"
        assert result.outputs["inspect"] is True

    async def test_runtime_runs_with_duck_typed_run_context_no_workspace(self, tmp_path):
        """Critical: runtime drives a workflow with a stub run_context that
        has no Workspace ancestry whatsoever, and writes workflow.json under
        run_dir/executions/<execution_id>/."""
        wf = WorkflowCompiler(name="duck")

        run_ctx = _RunContextStub(work_dir=tmp_path / "stub-run")

        @wf.task
        async def step(ctx: TaskContext) -> str:
            return "ok"

        result = await WorkflowRuntime().execute(wf.compile(), run_context=run_ctx)
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
        wf = WorkflowCompiler(name="no-run-kwarg")

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(TypeError):
            await WorkflowRuntime().execute(wf.compile(), run=object())

    async def test_legacy_profile_config_kwarg_is_rejected(self, tmp_path):
        """The runtime no longer accepts ``profile_config=``. Use ``config=``."""
        from molexp.profile import ProfileConfig

        wf = WorkflowCompiler(name="no-profile-config-kwarg")

        @wf.task
        async def noop(ctx: TaskContext) -> None:
            return None

        with pytest.raises(TypeError):
            await WorkflowRuntime().execute(
                wf.compile(), profile_config=ProfileConfig({}, name=None)
            )

    async def test_run_dir_kwarg_writes_workflow_json(self, tmp_path):
        wf = WorkflowCompiler(name="run-dir-only")

        @wf.task
        async def step(ctx: TaskContext) -> int:
            return 7

        result = await WorkflowRuntime().execute(wf.compile(), run_dir=tmp_path / "rd")
        assert result.status == "completed"
        wf_jsons = list((tmp_path / "rd" / "executions").rglob("workflow.json"))
        assert wf_jsons

    async def test_config_is_plain_mapping(self, tmp_path):
        wf = WorkflowCompiler(name="plain-config")

        # Build-time/runtime config now reaches a root task as a by-name run
        # param: the engine binds ``epochs`` from the run's params dict.
        @wf.task
        async def inspect(epochs: int) -> int:
            return epochs

        run_ctx = _RunContextStub(work_dir=tmp_path / "run", params={"epochs": 42})
        result = await WorkflowRuntime().execute(wf.compile(), run_context=run_ctx)
        assert result.outputs["inspect"] == 42


@pytest.mark.asyncio
class TestOOPExecution:
    async def test_task_subclass(self):
        class AddTen(Task):
            async def execute(self, ctx: TaskContext, value: int = 0) -> int:
                return value + 10

        spec = WorkflowCompiler(name="oop").add(AddTen()).compile()
        result = await WorkflowRuntime().execute(spec)
        assert result.outputs["add_ten"] == 10


@pytest.mark.asyncio
class TestProtocolExecution:
    async def test_external_runnable(self):
        class External:
            async def execute(self, ctx) -> int:
                return 99

        spec = WorkflowCompiler(name="ext").add(External(), name="ext").compile()
        result = await WorkflowRuntime().execute(spec)
        assert result.outputs["ext"] == 99

    async def test_mixed_task_types(self):
        class OOP(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 10

        class External:
            async def execute(self, ctx, oop) -> int:
                return oop + 90

        spec = (
            WorkflowCompiler(name="mixed")
            .add(OOP(), name="oop")
            .add(External(), name="ext", depends_on=["oop"])
            .compile()
        )
        result = await WorkflowRuntime().execute(spec)
        assert result.outputs == {"oop": 10, "ext": 100}


@pytest.mark.asyncio
class TestParallelExecution:
    async def test_independent_tasks_parallel(self):
        wf = WorkflowCompiler(name="parallel")

        @wf.task
        async def a(ctx):
            return "a"

        @wf.task
        async def b(ctx):
            return "b"

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs == {"a": "a", "b": "b"}

    async def test_diamond_dependency(self):
        wf = WorkflowCompiler(name="diamond")

        @wf.task
        async def root():
            return 1

        # Single scalar upstream binds positionally to the sole free parameter.
        @wf.task(depends_on=["root"])
        async def left(root) -> dict:
            return {"left": root + 10}

        @wf.task(depends_on=["root"])
        async def right(root) -> dict:
            return {"right": root + 20}

        # Two upstreams: their dict outputs MERGE into one flat name→value map,
        # and the body binds each by name.
        @wf.task(depends_on=["left", "right"])
        async def merge(left: int, right: int):
            return left + right

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["merge"] == 32
