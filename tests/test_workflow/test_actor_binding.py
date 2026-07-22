"""A streaming actor binds its non-``ctx`` parameters by name, exactly like a task.

Actors were once invoked as ``body.run(ctx)`` — only the context, no by-name
binding — so a streaming body could not read an upstream output or a build-time
config field. That was an engine asymmetry: batch tasks bind named parameters
through ``_bind_call_args`` while streaming bodies got only ``ctx``. These tests
pin the unified contract in ``_engine.node._drain_stream``: an actor's non-``ctx``
parameters bind from {build-time config} | {upstream outputs | run params} just
like a task's, and the async-generator drain (last yield → output) is unchanged.
"""

from __future__ import annotations

from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime


class TestActorBinding:
    async def test_reads_upstream_output_by_name_without_ctx(self) -> None:
        wf = WorkflowCompiler(name="actor-upstream")

        @wf.task
        async def source() -> list[int]:
            return [1, 2, 3]

        @wf.actor(depends_on=["source"])
        async def stream(source: list[int]):
            # ``source`` binds the upstream task's output by name — no ``ctx``.
            for x in source:
                yield x * 10

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["stream"] == 30  # last yield: 3 * 10

    async def test_reads_build_time_config_by_name(self) -> None:
        wf = WorkflowCompiler(name="actor-config", entry="stream")

        @wf.actor
        async def stream(factor: int = 1):
            # ``factor`` binds from build-time config, falling back to its default.
            for x in [1, 2, 3]:
                yield x * factor

        result = await WorkflowRuntime().execute(wf.compile(), config={"factor": 100})
        assert result.outputs["stream"] == 300  # last yield: 3 * 100

    async def test_ctx_only_actor_reduces_to_ctx_call(self) -> None:
        """Boundary: an actor whose sole param is ``ctx`` still drains unchanged."""
        wf = WorkflowCompiler(name="actor-ctx-only", entry="stream")

        @wf.actor
        async def stream(ctx: TaskContext):
            for item in [1, 2, 3]:
                yield {"seen": item}

        result = await WorkflowRuntime().execute(wf.compile())
        assert result.outputs["stream"] == {"seen": 3}
