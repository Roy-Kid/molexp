"""RED tests: a streaming actor binds its inputs by name, exactly like a task.

Actors used to be invoked as ``body.run(ctx)`` — only the context, no by-name
binding — so an actor could not read an upstream task's output, a build-time
config field, or a run param. That was an engine asymmetry: batch tasks bind
named parameters through ``_bind_call_args`` while streaming bodies were handed
only ``ctx``. These tests pin the unified contract: an actor's non-``ctx``
parameters bind from {build-time config} | {upstream outputs | run params} just
like a task's, and the async-generator drain (last yield → output) is unchanged.
"""

from __future__ import annotations

from molexp.workflow import Actor, TaskContext, WorkflowCompiler, WorkflowRuntime


async def test_decorator_actor_reads_upstream_by_name_without_ctx() -> None:
    wf = WorkflowCompiler(name="actor-upstream")

    @wf.task
    async def source() -> list[int]:
        return [1, 2, 3]

    @wf.actor(depends_on=["source"])
    async def stream(source: list[int]):
        # ``source`` binds the upstream task's output by name — no ``ctx`` needed.
        for x in source:
            yield x * 10

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.outputs["stream"] == 30  # last yield: 3 * 10


async def test_decorator_actor_reads_upstream_with_ctx_first() -> None:
    wf = WorkflowCompiler(name="actor-ctx-upstream")

    @wf.task
    async def source() -> list[int]:
        return [4, 5, 6]

    @wf.actor(depends_on=["source"])
    async def stream(ctx: TaskContext, source: list[int]):
        # ``ctx`` is the leading param (workdir); ``source`` still binds by name.
        assert ctx.workdir is None  # no workspace run attached
        for x in source:
            yield x + 1

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.outputs["stream"] == 7  # last yield: 6 + 1


async def test_oop_actor_subclass_reads_upstream_by_name() -> None:
    class Streamer(Actor):
        async def run(self, ctx: TaskContext, data: list[int]):
            for x in data:
                yield x - 1

    wf = WorkflowCompiler(name="oop-actor")

    @wf.task
    async def data() -> list[int]:
        return [10, 20, 30]

    compiled = wf.add(Streamer(), name="stream", depends_on=["data"]).compile()
    result = await WorkflowRuntime().execute(compiled)
    assert result.outputs["stream"] == 29  # last yield: 30 - 1


async def test_actor_reads_build_time_config_by_name() -> None:
    wf = WorkflowCompiler(name="actor-config", entry="stream")

    @wf.actor
    async def stream(factor: int = 1):
        # ``factor`` binds from the build-time config, falling back to its default.
        for x in [1, 2, 3]:
            yield x * factor

    result = await WorkflowRuntime().execute(wf.compile(), config={"factor": 100})
    assert result.outputs["stream"] == 300  # last yield: 3 * 100


async def test_actor_with_only_ctx_still_works() -> None:
    """Backward compatibility: an actor whose sole param is ``ctx`` is unchanged."""
    wf = WorkflowCompiler(name="actor-ctx-only", entry="stream")

    @wf.actor
    async def stream(ctx: TaskContext):
        for item in [1, 2, 3]:
            yield {"seen": item}

    result = await WorkflowRuntime().execute(wf.compile())
    assert result.outputs["stream"] == {"seen": 3}
