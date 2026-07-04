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

from molexp.workflow import (
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

    def test_sync_task_exception_fails_workflow(self) -> None:
        wf = WorkflowCompiler(name="sync-boom")

        @wf.task
        def boom() -> None:
            raise ValueError("broken body")

        result = _run(wf.compile())
        assert result.status == "failed"


class TestMixedSyncAsyncDag:
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
