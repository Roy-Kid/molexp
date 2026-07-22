"""Tests for ``SubWorkflow`` — the sanctioned sub-workflow composition node.

Contract:

* ``SubWorkflow`` is an exported ``Task`` subclass that runs an inner spec
  end-to-end through the engine and returns the inner terminal output (default
  dependency-leaf, or ``output=`` when set).
* It runs the inner via the engine-injected ``sub_runner`` capability — never a
  hand-built ``TaskContext`` and never a ``run_context`` on the task ctx.
* It slots into ``builder.parallel(body="sub", ...)`` as the per-element body
  with no per-element node growth, forwarding each element into the inner entry.
"""

from __future__ import annotations

import pytest

from molexp.workflow import (
    SubWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)


def _build_multi_step_inner() -> WorkflowCompiler:
    """A 3-task inner chain: load → normalize → scale (terminal = 1.75)."""
    wf = WorkflowCompiler(name="inner-multi")

    @wf.task
    async def load() -> list[float]:
        return [2.0, 4.0, 8.0]

    @wf.task(depends_on=["load"])
    async def normalize(values: list[float]) -> list[float]:
        top = max(values)
        return [x / top for x in values]

    @wf.task(depends_on=["normalize"])
    async def scale(values: list[float]) -> float:
        return sum(values)

    return wf


def _build_input_consuming_inner() -> WorkflowCompiler:
    """An inner chain whose ENTRY reads the forwarded value: seed(x)→x → scale→x*10."""
    wf = WorkflowCompiler(name="inner-consume")

    @wf.task
    async def seed(x: int) -> int:
        return x

    @wf.task(depends_on=["seed"])
    async def scale(x: int) -> int:
        return x * 10

    return wf


class TestSubWorkflow:
    @pytest.mark.asyncio
    async def test_multi_step_inner_returns_terminal_leaf_output(self) -> None:
        outer = (
            WorkflowCompiler(name="outer-multi")
            .add(SubWorkflow(_build_multi_step_inner()), name="sub")
            .compile()
        )
        result = await WorkflowRuntime().execute(outer)
        assert result.status == "succeeded"
        # load=[2,4,8] → normalize=[0.25,0.5,1.0] → scale=1.75
        assert result.outputs["sub"] == pytest.approx(1.75)

    @pytest.mark.asyncio
    async def test_accepts_precompiled_inner_workflow(self) -> None:
        inner_compiled = _build_multi_step_inner().compile()
        outer = (
            WorkflowCompiler(name="outer-compiled")
            .add(SubWorkflow(inner_compiled), name="sub")
            .compile()
        )
        result = await WorkflowRuntime().execute(outer)
        assert result.status == "succeeded"
        assert result.outputs["sub"] == pytest.approx(1.75)

    @pytest.mark.asyncio
    async def test_output_arg_selects_inner_task_output(self) -> None:
        outer = (
            WorkflowCompiler(name="outer-explicit")
            .add(SubWorkflow(_build_multi_step_inner(), output="normalize"), name="sub")
            .compile()
        )
        result = await WorkflowRuntime().execute(outer)
        assert result.status == "succeeded"
        assert result.outputs["sub"] == pytest.approx([0.25, 0.5, 1.0])

    def test_ambiguous_leaf_without_output_raises(self) -> None:
        inner = WorkflowCompiler(name="inner-two-leaves")

        @inner.task
        async def seed() -> int:
            return 1

        @inner.task(depends_on=["seed"])
        async def leaf_a(value: int) -> int:
            return value + 1

        @inner.task(depends_on=["seed"])
        async def leaf_b(value: int) -> int:
            return value + 2

        with pytest.raises(ValueError, match="leaf"):
            SubWorkflow(inner)._resolve_output_name()

    @pytest.mark.asyncio
    async def test_inner_runs_via_injected_sub_runner_without_run_context(self) -> None:
        """The inner task does NOT see ``run_context`` on its ctx; the engine
        injects a ``sub_runner`` capability bound to the outer run instead."""
        ran: list[bool] = []

        inner = WorkflowCompiler(name="inner-rc")

        @inner.task
        async def observe(ctx: TaskContext) -> str:
            assert not hasattr(ctx, "run_context")
            ran.append(True)
            return "ok"

        outer = WorkflowCompiler(name="outer-rc").add(SubWorkflow(inner), name="sub").compile()
        result = await WorkflowRuntime().execute(outer, run_context=object())
        assert result.status == "succeeded"
        assert ran == [True]
        assert result.outputs["sub"] == "ok"

    @pytest.mark.asyncio
    async def test_as_parallel_body_forwards_element_without_node_growth(self) -> None:
        """SubWorkflow is the per-element body of ``parallel``: the compiled task
        set is exactly the declared outer tasks (no per-element growth), and each
        element is forwarded into the inner entry → distinct per-element outputs."""
        wf = WorkflowCompiler(name="outer-parallel", entry="emit")

        @wf.task
        async def emit() -> list[int]:
            return [1, 2, 3]

        wf.add(SubWorkflow(_build_input_consuming_inner()), name="sub")

        @wf.task
        async def collect(values: list[int]) -> list[int]:
            return list(values)

        wf.parallel(map_over="emit", body="sub", join="collect", max_concurrency=2)
        compiled = wf.compile()

        # No per-element node growth — exactly the declared outer task set.
        assert {t.name for t in compiled._tasks} == {"emit", "sub", "collect"}

        result = await WorkflowRuntime().execute(compiled)
        assert result.status == "succeeded"
        # seed(x)=x → scale=x*10 — one DISTINCT output per element.
        assert result.outputs["collect"] == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_parallel_inner_failure_surfaces_via_parallel_execution_error(self) -> None:
        from molexp.workflow import ParallelExecutionError

        state = {"n": 0}

        inner = WorkflowCompiler(name="inner-maybe-fail")

        @inner.task
        async def step() -> int:
            idx = state["n"]
            state["n"] += 1
            if idx == 1:
                raise ValueError("boom")
            return idx

        wf = WorkflowCompiler(name="outer-parallel-fail", entry="emit")

        @wf.task
        async def emit() -> list[int]:
            return [0, 1, 2]

        wf.add(SubWorkflow(inner), name="sub")

        @wf.task
        async def collect(values: list[int]) -> int:
            return len(values)

        wf.parallel(map_over="emit", body="sub", join="collect", max_concurrency=1)

        with pytest.raises(ParallelExecutionError) as exc_info:
            await WorkflowRuntime().execute(wf.compile())
        assert exc_info.value.body == "sub"
        assert set(exc_info.value.failures.keys()) == {1}

    @pytest.mark.asyncio
    async def test_non_parallel_node_forwards_upstream_output_into_inner_entry(self) -> None:
        class Source(Task):
            async def execute(self, ctx: TaskContext) -> int:
                return 7

        outer = (
            WorkflowCompiler(name="outer-chain-forward")
            .add(Source(), name="src")
            .add(SubWorkflow(_build_input_consuming_inner()), name="sub", depends_on=["src"])
            .compile()
        )
        result = await WorkflowRuntime().execute(outer)
        assert result.status == "succeeded"
        assert result.outputs["sub"] == 70  # seed(7) → scale 70

    @pytest.mark.asyncio
    async def test_bare_root_multi_root_inner_runs_without_forwarding(self) -> None:
        """A bare-root SubWorkflow forwards nothing, so a multi-root inner spec is
        NOT forced to declare a single entry (input-less inner runs unchanged)."""
        inner = WorkflowCompiler(name="inner-two-roots")

        @inner.task
        async def root_a() -> dict:
            return {"root_a": 1}

        @inner.task
        async def root_b() -> dict:
            return {"root_b": 2}

        @inner.task(depends_on=["root_a", "root_b"])
        async def merge(root_a: int, root_b: int) -> int:
            return root_a + root_b

        outer = (
            WorkflowCompiler(name="outer-two-roots").add(SubWorkflow(inner), name="sub").compile()
        )
        result = await WorkflowRuntime().execute(outer)
        assert result.status == "succeeded"
        assert result.outputs["sub"] == 3


class TestResolveSingleRoot:
    def test_multi_root_forwarding_target_raises(self) -> None:
        from molexp.workflow._engine.runtime import _resolve_single_root

        inner = WorkflowCompiler(name="inner-ambiguous-roots")

        @inner.task
        async def root_a() -> int:
            return 1

        @inner.task
        async def root_b() -> int:
            return 2

        with pytest.raises(ValueError, match="single entry"):
            _resolve_single_root(inner.compile())

    def test_honors_explicit_entry(self) -> None:
        from molexp.workflow._engine.runtime import _resolve_single_root

        inner = WorkflowCompiler(name="inner-explicit-entry", entry="head")

        @inner.task
        async def head(x: int) -> int:
            return x

        @inner.task(depends_on=["head"])
        async def tail(x: int) -> int:
            return x + 1

        assert _resolve_single_root(inner.compile()) == "head"
