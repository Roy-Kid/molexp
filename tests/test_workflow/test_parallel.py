"""Tests for wf.parallel — fan-out / map-reduce primitive (spec 05).

The primitive's contract:

* ``wf.parallel(map_over, body, join, max_concurrency)`` declares one
  body task that is invoked once per element of the upstream task's
  collection output, bounded by ``Semaphore(max_concurrency)``.
* When all elements complete, ``join`` runs with ``ctx.inputs ==
  [out_0, out_1, …]`` in element-iteration order.
* The static graph stays static — exactly one BaseNode subclass per
  registered task (no per-element growth); the runtime multiplexes
  call coroutines.
* Per-element failures are captured into
  :class:`molexp.workflow.ParallelExecutionError`; siblings still
  complete and their outputs are recorded.
"""

from __future__ import annotations

import asyncio

import pytest

from molexp.workflow import WorkflowBuilder
from molexp.workflow.types import Next

# ── T1 / ac-002, ac-003: encapsulation invariants ───────────────────────────


def test_parallel_execution_error_importable() -> None:
    """ac-002 — ``ParallelExecutionError`` is exported from ``molexp.workflow``."""
    from molexp.workflow import ParallelExecutionError

    assert issubclass(ParallelExecutionError, Exception)


def test_no_per_element_node_growth() -> None:
    """ac-003 — compiled spec carries one task entry per registered task,
    independent of how many elements the parallel section will fan out over.

    After the single-track rectification, the compiler no longer emits per-task
    pg ``BaseNode`` subclasses; it stores the user's registered Task /
    Actor / callable directly under ``compiled.task_by_name``. The fan-out
    width is decided at run time from ``state.results[map_over]`` and never
    grows the compile-time entry set.
    """
    from molexp.workflow._pydantic_graph.compiler import WorkflowGraphCompiler

    wf = WorkflowBuilder(name="no-per-elem-growth", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[int]:
        return [10, 20, 30, 40]  # 4 elements

    @wf.task
    async def square(ctx) -> int:
        return ctx.inputs * ctx.inputs

    @wf.task
    async def sum_results(ctx) -> int:
        return sum(ctx.inputs)

    wf.parallel(map_over="enumerate", body="square", join="sum_results", max_concurrency=2)

    compiled = WorkflowGraphCompiler().compile(wf.build())

    assert set(compiled.task_by_name.keys()) == {"enumerate", "square", "sum_results"}, (
        f"Expected exactly 3 entries (enumerate + square + sum_results), "
        f"got {sorted(compiled.task_by_name.keys())}"
    )


# ── T2 / ac-004: happy path ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_happy_path() -> None:
    """ac-004 — squarer + summer pipeline; ordered list and reduced sum."""
    wf = WorkflowBuilder(name="parallel-happy", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[int]:
        return [1, 2, 3]

    @wf.task
    async def body(ctx) -> int:
        return ctx.inputs * ctx.inputs

    @wf.task
    async def join(ctx) -> int:
        return sum(ctx.inputs)

    wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=3)

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["body"] == [1, 4, 9]
    assert result.outputs["join"] == 14


@pytest.mark.asyncio
async def test_parallel_preserves_iteration_order() -> None:
    """ac-004 — body completion order does not affect the recorded list order.

    Element 0 sleeps the longest; the recorded list MUST still be in
    iteration order, not completion order.
    """
    wf = WorkflowBuilder(name="parallel-order", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[float]:
        # Sleeps in *reverse* — element 0 is slowest, element 4 fastest.
        return [0.05, 0.04, 0.03, 0.02, 0.01]

    @wf.task
    async def body(ctx) -> float:
        await asyncio.sleep(ctx.inputs)
        return ctx.inputs

    @wf.task
    async def join(ctx) -> list[float]:
        return list(ctx.inputs)

    wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=5)

    result = await wf.build().execute()
    assert result.status == "completed"
    # Iteration order, not completion order.
    assert result.outputs["body"] == [0.05, 0.04, 0.03, 0.02, 0.01]
    assert result.outputs["join"] == [0.05, 0.04, 0.03, 0.02, 0.01]


# ── T6 / ac-005, ac-006: throttling + failure semantics ────────────────────


@pytest.mark.asyncio
async def test_max_concurrency_throttle() -> None:
    """ac-005 — instrumented in-flight counter caps at ``max_concurrency=2``.

    Five sleeping body invocations run with ``max_concurrency=2``; the
    instrumented counter (incremented on entry, decremented on exit)
    must never observe more than 2 in-flight at once.
    """
    inflight = 0
    observed_max = 0
    lock = asyncio.Lock()

    wf = WorkflowBuilder(name="parallel-throttle", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[int]:
        return [0, 1, 2, 3, 4]

    @wf.task
    async def body(ctx) -> int:
        nonlocal inflight, observed_max
        async with lock:
            inflight += 1
            observed_max = max(observed_max, inflight)
        try:
            await asyncio.sleep(0.05)
            return ctx.inputs
        finally:
            async with lock:
                inflight -= 1

    @wf.task
    async def join(ctx) -> int:
        return len(ctx.inputs)

    wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=2)

    result = await wf.build().execute()
    assert result.status == "completed"
    assert result.outputs["join"] == 5
    assert observed_max <= 2, f"Expected in-flight cap of 2, observed {observed_max}"


@pytest.mark.asyncio
async def test_parallel_failure_capture() -> None:
    """ac-006 — element 1 raises; siblings finish; ``ParallelExecutionError``
    surfaces with ``failures[1]`` populated.
    """
    from molexp.workflow import ParallelExecutionError, WorkflowBuilder

    sibling_completions: list[int] = []

    wf = WorkflowBuilder(name="parallel-failure", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[int]:
        return [0, 1, 2]

    @wf.task
    async def body(ctx) -> int:
        if ctx.inputs == 1:
            raise ValueError("boom")
        sibling_completions.append(ctx.inputs)
        return ctx.inputs * 10

    @wf.task
    async def join(ctx) -> int:
        return sum(x for x in ctx.inputs if x is not None)

    wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=3)

    with pytest.raises(ParallelExecutionError) as exc_info:
        await wf.build().execute()

    assert exc_info.value.body == "body"
    assert set(exc_info.value.failures.keys()) == {1}
    assert isinstance(exc_info.value.failures[1], ValueError)
    assert "boom" in str(exc_info.value.failures[1])
    # Siblings 0 and 2 still ran (capture-don't-cancel).
    assert set(sibling_completions) == {0, 2}


@pytest.mark.asyncio
async def test_parallel_records_run_count_for_observability() -> None:
    """ac-004 — ``state.parallel_runs[body] == N`` after fan-out completes.

    Verified indirectly via the workflow's persisted state. We can't
    inspect ``state`` directly through ``WorkflowResult``, so we use a
    workspace-backed run; in lieu of that, we re-execute and assert the
    outputs[body] length, which is the observable consequence.
    """
    wf = WorkflowBuilder(name="parallel-count", entry="enumerate")

    @wf.task
    async def enumerate(ctx) -> list[int]:
        return [10, 20, 30, 40]

    @wf.task
    async def body(ctx) -> int:
        return ctx.inputs

    @wf.task
    async def join(ctx) -> int:
        return len(ctx.inputs)

    wf.parallel(map_over="enumerate", body="body", join="join", max_concurrency=4)

    result = await wf.build().execute()
    assert result.status == "completed"
    assert len(result.outputs["body"]) == 4
    assert result.outputs["join"] == 4


# ── T8 / ac-007: compile-time validation ────────────────────────────────────


def test_validation_max_concurrency_zero_rejected() -> None:
    """ac-007 — ``max_concurrency=0`` is a programming error, eager-rejected."""
    wf = WorkflowBuilder(name="bad-concurrency", entry="seed")

    @wf.task
    async def seed(ctx) -> list[int]:
        return [1, 2]

    @wf.task
    async def body(ctx) -> int:
        return ctx.inputs

    @wf.task
    async def join(ctx) -> int:
        return sum(ctx.inputs)

    with pytest.raises(ValueError):
        wf.parallel(map_over="seed", body="body", join="join", max_concurrency=0)


def test_validation_unregistered_map_over_rejected() -> None:
    """ac-007 — unregistered ``map_over`` task name → ``UnknownTaskError``."""
    from molexp.workflow import UnknownTaskError, WorkflowBuilder

    wf = WorkflowBuilder(name="bad-map-over", entry="body")

    @wf.task
    async def body(ctx) -> int:
        return ctx.inputs

    @wf.task
    async def join(ctx) -> int:
        return sum(ctx.inputs)

    wf.parallel(map_over="missing", body="body", join="join", max_concurrency=2)

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.build()
    assert "missing" in str(exc_info.value)


def test_validation_unregistered_body_rejected() -> None:
    """ac-007 — unregistered ``body`` task name → ``UnknownTaskError``."""
    from molexp.workflow import UnknownTaskError, WorkflowBuilder

    wf = WorkflowBuilder(name="bad-body", entry="seed")

    @wf.task
    async def seed(ctx) -> list[int]:
        return [1, 2]

    @wf.task
    async def join(ctx) -> int:
        return sum(ctx.inputs)

    wf.parallel(map_over="seed", body="missing", join="join", max_concurrency=2)

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.build()
    assert "missing" in str(exc_info.value)


def test_validation_unregistered_join_rejected() -> None:
    """ac-007 — unregistered ``join`` task name → ``UnknownTaskError``."""
    from molexp.workflow import UnknownTaskError, WorkflowBuilder

    wf = WorkflowBuilder(name="bad-join", entry="seed")

    @wf.task
    async def seed(ctx) -> list[int]:
        return [1, 2]

    @wf.task
    async def body(ctx) -> int:
        return ctx.inputs

    wf.parallel(map_over="seed", body="body", join="missing", max_concurrency=2)

    with pytest.raises(UnknownTaskError) as exc_info:
        wf.build()
    assert "missing" in str(exc_info.value)


def test_validation_parallel_of_parallel_rejected() -> None:
    """ac-007 — same ``body`` shared across two parallel decls → ``EdgeShapeError``."""
    from molexp.workflow import EdgeShapeError, WorkflowBuilder

    wf = WorkflowBuilder(name="parallel-of-parallel", entry="seed")

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
        wf.build()
    assert "shared_body" in str(exc_info.value)


def test_validation_loop_until_eq_parallel_body_rejected() -> None:
    """ac-007 — using a loop ``until`` task as a parallel ``body`` → ``EdgeShapeError``."""
    from molexp.workflow import EdgeShapeError, WorkflowBuilder

    wf = WorkflowBuilder(name="loop-parallel-collision", entry="seed")

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
        wf.build()


def test_validation_body_extra_depends_on_rejected() -> None:
    """ac-007 — body declared with extra ``depends_on`` than ``[map_over]`` →
    ``EdgeShapeError`` (parallel owns the wiring per D6).
    """
    from molexp.workflow import EdgeShapeError, WorkflowBuilder

    wf = WorkflowBuilder(name="body-extra-deps", entry="seed")

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
        wf.build()
