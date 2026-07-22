"""Tests for ``wf.parallel`` — the fan-out / map-reduce primitive.

The primitive's contract:

* ``wf.parallel(map_over, body, join, max_concurrency)`` declares one body task
  invoked once per element of the upstream task's collection output, bounded by
  ``Semaphore(max_concurrency)``.
* When all elements complete, ``join`` runs with ``ctx.inputs == [out_0, out_1,
  …]`` in element-iteration order.
* The static graph stays static — exactly one task entry per registered task (no
  per-element growth); the runtime multiplexes call coroutines.
* Per-element failures are captured into
  :class:`molexp.workflow.ParallelExecutionError`; siblings still complete and
  their outputs are recorded.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.workflow import (
    EdgeShapeError,
    ParallelExecutionError,
    UnknownTaskError,
    WorkflowCompiler,
    WorkflowRuntime,
)
from molexp.workflow.types import Next


class TestParallel:
    def test_no_per_element_node_growth(self) -> None:
        """The compiled task set is exactly the declared tasks, independent of
        how many elements the parallel section fans out over.

        The fan-out width is decided at run time from ``state.results[map_over]``
        and never grows the compile-time task set.
        """
        wf = WorkflowCompiler(name="no-per-elem-growth", entry="enumerate")

        @wf.task
        async def enumerate(ctx) -> list[int]:
            return [10, 20, 30, 40]  # 4 elements

        @wf.task
        async def square(value: int) -> int:
            return value * value

        @wf.task
        async def sum_results(values: list[int]) -> int:
            return sum(values)

        wf.parallel(map_over="enumerate", body="square", join="sum_results", max_concurrency=2)

        compiled = wf.compile()

        assert {t.name for t in compiled._tasks} == {"enumerate", "square", "sum_results"}, (
            f"Expected exactly 3 task entries, got {sorted(t.name for t in compiled._tasks)}"
        )

    @pytest.mark.asyncio
    async def test_happy_path_maps_then_reduces(self) -> None:
        """Squarer + summer pipeline; ordered per-element list and reduced sum."""
        wf = WorkflowCompiler(name="parallel-happy", entry="enumerate")

        @wf.task
        async def enumerate(ctx) -> list[int]:
            return [1, 2, 3]

        @wf.task
        async def body(value: int) -> int:
            return value * value

        @wf.task
        async def join(values: list[int]) -> int:
            return sum(values)

        wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=3)

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["body"] == [1, 4, 9]
        assert result.outputs["join"] == 14

    @pytest.mark.asyncio
    async def test_records_iteration_order_not_completion_order(self) -> None:
        """Body completion order does not affect the recorded list order.

        Element 0 sleeps the longest; the recorded list MUST still be in
        iteration order, not completion order.
        """
        wf = WorkflowCompiler(name="parallel-order", entry="enumerate")

        @wf.task
        async def enumerate(ctx) -> list[float]:
            # Sleeps in *reverse* — element 0 is slowest, element 4 fastest.
            return [0.05, 0.04, 0.03, 0.02, 0.01]

        @wf.task
        async def body(delay: float) -> float:
            await asyncio.sleep(delay)
            return delay

        @wf.task
        async def join(values: list[float]) -> list[float]:
            return list(values)

        wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=5)

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["body"] == [0.05, 0.04, 0.03, 0.02, 0.01]
        assert result.outputs["join"] == [0.05, 0.04, 0.03, 0.02, 0.01]

    @pytest.mark.asyncio
    async def test_max_concurrency_caps_in_flight_bodies(self) -> None:
        """Five sleeping body invocations run with ``max_concurrency=2``; the
        instrumented in-flight counter never exceeds 2.
        """
        inflight = 0
        observed_max = 0
        lock = asyncio.Lock()

        wf = WorkflowCompiler(name="parallel-throttle", entry="enumerate")

        @wf.task
        async def enumerate(ctx) -> list[int]:
            return [0, 1, 2, 3, 4]

        @wf.task
        async def body(value: int) -> int:
            nonlocal inflight, observed_max
            async with lock:
                inflight += 1
                observed_max = max(observed_max, inflight)
            try:
                await asyncio.sleep(0.05)
                return value
            finally:
                async with lock:
                    inflight -= 1

        @wf.task
        async def join(values: list[int]) -> int:
            return len(values)

        wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=2)

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.status == "succeeded"
        assert result.outputs["join"] == 5
        assert observed_max <= 2, f"Expected in-flight cap of 2, observed {observed_max}"

    @pytest.mark.asyncio
    async def test_per_element_failure_captured_siblings_complete(self) -> None:
        """Element 1 raises; siblings finish; ``ParallelExecutionError`` surfaces
        with ``failures[1]`` populated (capture-don't-cancel).
        """
        sibling_completions: list[int] = []

        wf = WorkflowCompiler(name="parallel-failure", entry="enumerate")

        @wf.task
        async def enumerate(ctx) -> list[int]:
            return [0, 1, 2]

        @wf.task
        async def body(value: int) -> int:
            if value == 1:
                raise ValueError("boom")
            sibling_completions.append(value)
            return value * 10

        @wf.task
        async def join(values: list[int]) -> int:
            return sum(x for x in values if x is not None)

        wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=3)

        with pytest.raises(ParallelExecutionError) as exc_info:
            await WorkflowRuntime().execute(wf.compile())

        assert exc_info.value.body == "body"
        assert set(exc_info.value.failures.keys()) == {1}
        assert isinstance(exc_info.value.failures[1], ValueError)
        assert "boom" in str(exc_info.value.failures[1])
        # Siblings 0 and 2 still ran (capture-don't-cancel).
        assert set(sibling_completions) == {0, 2}

    def test_rejects_unregistered_map_over(self) -> None:
        """An unregistered ``map_over`` task name → ``UnknownTaskError``."""
        wf = WorkflowCompiler(name="bad-map-over", entry="body")

        @wf.task
        async def body(ctx) -> int:
            return ctx.inputs

        @wf.task
        async def join(ctx) -> int:
            return sum(ctx.inputs)

        wf.parallel(map_over="missing", body="body", join="join", max_concurrency=2)

        with pytest.raises(UnknownTaskError) as exc_info:
            wf.compile()
        assert "missing" in str(exc_info.value)

    def test_rejects_body_shared_across_two_parallels(self) -> None:
        """Same ``body`` shared across two parallel decls → ``EdgeShapeError``."""
        wf = WorkflowCompiler(name="parallel-of-parallel", entry="seed")

        @wf.task
        async def seed(ctx) -> list[int]:
            return [1, 2]

        @wf.task
        async def other_seed(ctx) -> list[int]:
            return [3, 4]

        @wf.task
        async def shared_body(ctx) -> int:
            return ctx.inputs

        @wf.task
        async def join(ctx) -> int:
            return sum(ctx.inputs)

        wf.parallel(map_over="seed", body="shared_body", join="join", max_concurrency=2)
        wf.parallel(map_over="other_seed", body="shared_body", join="join", max_concurrency=2)

        with pytest.raises(EdgeShapeError) as exc_info:
            wf.compile()
        assert "shared_body" in str(exc_info.value)

    def test_rejects_loop_until_reused_as_parallel_body(self) -> None:
        """A loop ``until`` task reused as a parallel ``body`` → ``EdgeShapeError``
        (a parallel-join and a loop-until cannot be fused onto the same task).
        """
        wf = WorkflowCompiler(name="loop-parallel-collision", entry="seed")

        @wf.task
        async def seed(ctx) -> list[int]:
            return [1, 2, 3]

        @wf.task
        async def body(ctx) -> int:
            return ctx.inputs

        @wf.task
        async def join(ctx) -> Next:
            return Next("exit")

        # join is the until of a loop AND the body of a parallel — illegal.
        wf.loop(body=["body"], until="join", max_iters=10)
        wf.parallel(map_over="seed", body="join", join="seed", max_concurrency=2)

        with pytest.raises(EdgeShapeError):
            wf.compile()

    def test_rejects_body_with_extra_depends_on(self) -> None:
        """A body declared with extra ``depends_on`` beyond ``[map_over]`` →
        ``EdgeShapeError`` (parallel owns the wiring per D6).
        """
        wf = WorkflowCompiler(name="body-extra-deps", entry="seed")

        @wf.task
        async def seed(ctx) -> list[int]:
            return [1, 2]

        @wf.task
        async def other(ctx) -> int:
            return 99

        @wf.task(depends_on=["other"])
        async def body(ctx) -> int:
            return ctx.inputs

        @wf.task
        async def join(ctx) -> int:
            return sum(ctx.inputs)

        wf.parallel(map_over="seed", body="body", join="join", max_concurrency=2)

        with pytest.raises(EdgeShapeError):
            wf.compile()
