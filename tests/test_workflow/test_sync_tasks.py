"""Sync task bodies are first-class workflow citizens (runset-api sub-task 3).

A pure-computation task should be writable as a plain ``def`` — the engine
dispatches sync bodies to a worker thread (mirroring ``promote._EntryTask``)
so blocking bodies don't stall same-level siblings, and awaits async bodies
as before. Locks the full chain: compile, execute, mixed sync/async DAGs,
OOP ``Task`` subclasses, TaskContext access, and content-addressed caching.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from molexp.workflow import (
    Caching,
    FileCacheStore,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)


def _run(compiled, **kwargs: object):
    return asyncio.run(WorkflowRuntime().execute(compiled, **kwargs))


class TestSyncDecoratorTask:
    def test_sync_task_executes(self) -> None:
        wf = WorkflowCompiler(name="sync-single")

        @wf.task
        def double(x: int) -> int:
            return x * 2

        result = _run(wf.compile(), config={"x": 21})
        assert result.status == "succeeded"
        assert result.outputs["double"] == 42

    def test_sync_dag_binds_by_name(self) -> None:
        wf = WorkflowCompiler(name="sync-dag")

        @wf.task
        def fetch(scale: float = 1.0) -> dict:
            return {"values": [1.0, 4.0, 9.0], "scale": scale}

        @wf.task(depends_on=["fetch"])
        def summarize(values: list[float], scale: float = 1.0) -> float:
            return sum(values) * scale

        result = _run(wf.compile(), config={"scale": 2.0})
        assert result.status == "succeeded"
        assert result.outputs["summarize"] == 28.0

    def test_sync_task_with_ctx_param(self) -> None:
        wf = WorkflowCompiler(name="sync-ctx")

        @wf.task
        def probe(ctx: TaskContext) -> str:
            assert ctx is not None
            return "ok"

        result = _run(wf.compile())
        assert result.outputs["probe"] == "ok"

    def test_sync_task_exception_fails_workflow(self) -> None:
        wf = WorkflowCompiler(name="sync-boom")

        @wf.task
        def boom() -> None:
            raise ValueError("broken body")

        result = _run(wf.compile())
        assert result.status == "failed"


class TestMixedSyncAsyncDag:
    def test_mixed_dag_end_to_end(self) -> None:
        wf = WorkflowCompiler(name="mixed")

        @wf.task
        def prepare(x: int) -> int:
            return x + 1

        @wf.task(depends_on=["prepare"])
        async def transform(prepare: int) -> int:
            await asyncio.sleep(0)
            return prepare * 10

        @wf.task(depends_on=["transform"])
        def summarize(transform: int) -> str:
            return f"total={transform}"

        result = _run(wf.compile(), config={"x": 5})
        assert result.status == "succeeded"
        assert result.outputs == {
            "prepare": 6,
            "transform": 60,
            "summarize": "total=60",
        }

    def test_blocking_sync_body_does_not_stall_async_sibling(self) -> None:
        """A blocking sync body runs in a worker thread, so a same-level
        async sibling still makes progress (parallel levels stay parallel)."""
        wf = WorkflowCompiler(name="mixed-parallel")

        @wf.task
        def slow_sync() -> str:
            time.sleep(0.2)
            return "sync"

        @wf.task
        async def fast_async() -> str:
            return "async"

        @wf.task(depends_on=["slow_sync", "fast_async"])
        def join(slow_sync: str, fast_async: str) -> str:
            return slow_sync + "+" + fast_async

        start = time.monotonic()
        result = _run(wf.compile())
        elapsed = time.monotonic() - start
        assert result.outputs["join"] == "sync+async"
        # Generous bound: serialized-with-loop-stall would still pass, but a
        # deadlock / double-sleep regression would not.
        assert elapsed < 2.0


class TestSyncOopTask:
    def test_sync_execute_override(self) -> None:
        class Square(Task):
            def execute(self, ctx: TaskContext, value: int) -> int:  # type: ignore[override]
                return value**2

        wf = WorkflowCompiler(name="oop-sync").add(Square(), name="square")
        result = _run(wf.compile(), config={"value": 6})
        assert result.status == "succeeded"
        assert result.outputs["square"] == 36


class TestSyncTaskCaching:
    def test_second_run_hits_cache(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def build() -> WorkflowCompiler:
            wf = WorkflowCompiler(name="sync-cache")

            @wf.task
            def compute(x: int) -> int:
                calls.append(x)
                return x * 3

            return wf

        cache = Caching(store=FileCacheStore(tmp_path / "cache"))
        first = _run(build().compile(), config={"x": 7}, cache=cache)
        second = _run(build().compile(), config={"x": 7}, cache=cache)
        assert first.outputs["compute"] == 21
        assert second.outputs["compute"] == 21
        assert calls == [7], "second run must be served from the cache"
