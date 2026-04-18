# Actor Model

Actors are the streaming variant of a workflow task. Where a `Task` produces a single value per execution, an `Actor` is an async generator that yields a series of values — useful for continuous processing, stream transformation, and components that run at different rates than their peers.

The current actor support in MolExp is **minimal on purpose**: a base class, a structural protocol, and `ctx.receive()` / `ctx.send()` primitives that route through the enclosing `RunContext`. Richer features (hot reconfig, automatic backpressure, replay) are not implemented — use this page as a ground-truth reference for what works today.

## Defining an Actor

Two equivalent styles:

```python
# Functional DSL
from molexp.workflow import workflow, ActorContext

wf = workflow(name="stream")

@wf.actor
async def monitor(ctx: ActorContext):
    while True:
        msg = await ctx.receive()
        yield {"seen": msg}
```

```python
# Subclass the Actor base
from molexp.workflow import Actor, ActorContext

class Monitor(Actor):
    async def run(self, ctx: ActorContext):
        while True:
            msg = await ctx.receive()
            yield {"seen": msg}
```

Any object with `async def run(self, ctx)` returning an async iterator satisfies the `Streamable` protocol and can be added via `WorkflowBuilder.add(obj)`. The `Runnable` vs `Streamable` choice is made at runtime through `isinstance(obj, Streamable)`.

## ActorContext

`ActorContext` extends `TaskContext` (so you still have `ctx.state`, `ctx.deps`, `ctx.inputs`, `ctx.config`, and all workspace helpers) and adds:

```python
await ctx.receive()          # wait for the next message on the default "input" channel
await ctx.send(output)       # emit to the default "output" channel
```

These call into `RunContext.receive("input")` / `RunContext.emit("output", output)` under the hood. If you invoke them without a connected `RunContext`, they raise `NotImplementedError` — actors only make sense when a `Run` is driving them.

## When to Use an Actor

- Continuous sampling / monitoring where one upstream produces at a variable rate.
- Stream transformation where buffering should be handled by the channel, not by the task body.
- Loops that need to interleave with other tasks via `await` without blocking a whole level.

If your computation is a single function call that returns one value, use `Task` — actors are more machinery than you need.

## What the Runtime Currently Does

- Detects streaming tasks via `Streamable` and marks them as actors in the compiled graph.
- Provides `ctx.receive()` / `ctx.send()` plumbing when a `RunContext` is attached.
- Interleaves actor `run()` coroutines via the normal pydantic-graph scheduler.

## What the Runtime Does *Not* Currently Do

- Automatic channel lifecycle (you register channels on the `RunContext._channels` dict yourself).
- Backpressure policies, buffered-channel sizing, or drop strategies (these are `asyncio.Queue` defaults).
- Replay / resume for an actor mid-stream.
- Hot reconfiguration of an actor's config while it runs.

If you need any of those, open an issue describing the workload — they are non-trivial extensions to the runtime, not just docs.

## Pointer to the Code

- `molexp.workflow.task.Actor` — base class
- `molexp.workflow.context.ActorContext` — the context passed to `run()`
- `molexp.workflow.protocols.Streamable` — structural protocol for zero-import integration
- `molexp.workspace.run.RunContext.receive` / `emit` — channel plumbing
