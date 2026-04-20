# Task and Actor

`Task` and `Actor` are the atomic units of a MolExp workflow. `Task` runs once and returns a value; `Actor` runs continuously and yields a stream of values. Both have a **protocol** form (zero-import third-party integration) and a **convenience base class** form (`molexp.workflow.Task` / `Actor`).

## Task Semantics

A `Task` is a single async function (or class with `async def execute(self, ctx)`) that consumes the typed output of its upstream task and returns a typed output of its own. The compiler groups tasks into **levels** by the dependency graph — every task on the same level runs in parallel.

Three equivalent ways to define a task:

```python
# 1. Function decorated with @wf.task
from molexp.workflow import workflow, TaskContext

wf = workflow(name="pipeline")

@wf.task
async def fetch(ctx: TaskContext) -> dict:
    return {"n": 42}
```

```python
# 2. Subclass the convenience base class
from molexp.workflow import Task, TaskContext

class Fetch(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        return {"n": 42}
```

```python
# 3. Any object whose method signature matches Runnable
class ExternalFetch:
    async def execute(self, ctx) -> dict:
        return {"n": 42}
```

All three are interchangeable — the last form **requires no `molexp` import**, which is what lets you drop in third-party components unchanged.

## Declaring Dependencies

Dependencies are declared **by task name** (not output name). The return value of an upstream task becomes `ctx.inputs` of its single direct downstream:

```python
@wf.task
async def square(ctx: TaskContext) -> float:
    return ctx.inputs ** 2

@wf.task(depends_on=["square"])
async def add_bias(ctx: TaskContext) -> float:
    return ctx.inputs + 10.0
```

For the OOP builder:

```python
from molexp.workflow import WorkflowBuilder

spec = (
    WorkflowBuilder(name="pipeline")
    .add(Fetch())                                   # auto-named "fetch"
    .add(Process(), depends_on=["fetch"])           # name inferred from class
    .add(Report(), depends_on=["process"], name="report")
    .build()
)
```

## Generic Parameters

`Task` is generic over `StateT, DepsT, InputT, OutputT`. Fill them in only when you want full static typing:

```python
from dataclasses import dataclass

@dataclass
class PipeState:
    cursor: int = 0

@dataclass
class PipeDeps:
    storage: Any
    metrics: Any

class Fetch(Task[PipeState, PipeDeps, None, dict]):
    async def execute(
        self, ctx: TaskContext[PipeState, PipeDeps, None]
    ) -> dict:
        ctx.state.cursor += 1
        return ctx.deps.storage.read("batch-0")
```

Plain `Task` (no generics) defaults to `Any` everywhere.

## Actor — Streaming Tasks

`Actor` is the streaming variant. Its `run()` returns an async iterator; the compiler detects it structurally via the `Streamable` protocol. Use actors for continuous sampling, stream transformation, and components that run at a different rate than their peers. If your computation is a single call returning one value, use `Task` — actors are more machinery than you need.

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
# Subclass Actor
from molexp.workflow import Actor, ActorContext

class Monitor(Actor):
    async def run(self, ctx: ActorContext):
        while True:
            msg = await ctx.receive()
            yield {"seen": msg}
```

Any object with `async def run(self, ctx)` returning an async iterator satisfies the `Streamable` protocol and can be added via `WorkflowBuilder.add(obj)` — no molexp import required.

### ActorContext

`ActorContext` extends `TaskContext` (so `ctx.state`, `ctx.deps`, `ctx.inputs`, `ctx.config`, and workspace helpers are all still there) and adds two primitives:

```python
await ctx.receive()       # wait for the next message on the default "input" channel
await ctx.send(output)    # emit to the default "output" channel
```

These route through `RunContext.receive("input")` / `RunContext.emit("output", output)`. Without a connected `RunContext` they raise `NotImplementedError` — actors only make sense under a `Run`.

### Runtime Boundaries

Supported today:

- Structural detection of streaming tasks via `Streamable`.
- `ctx.receive()` / `ctx.send()` plumbing when a `RunContext` is attached.
- Interleaving actor coroutines through the normal pydantic-graph scheduler.

Not implemented (open an issue if you need it — these are runtime extensions, not doc gaps):

- Automatic channel lifecycle (you register channels on `RunContext._channels` yourself).
- Backpressure / buffered-channel sizing / drop strategies beyond `asyncio.Queue` defaults.
- Replay or resume for an actor mid-stream.
- Hot reconfiguration of an actor's config while it runs.

Relevant code: `molexp.workflow.task.Actor`, `molexp.workflow.context.ActorContext`, `molexp.workflow.protocols.Streamable`, `molexp.workspace.run.RunContext.receive` / `emit`.

## Task Name Resolution

| Form | Name source |
|------|-------------|
| `@wf.task def fetch(...)` | Function name (`fetch`) |
| `@wf.task(name="X")` | Explicit `name=` |
| `WorkflowBuilder.add(FetchTask())` | Class name converted to snake_case, minus trailing `_task` / `_actor` → `fetch` |
| `WorkflowBuilder.add(Fetch(), name="X")` | Explicit `name=` |

`depends_on` values must match one of these resolved names exactly.

## Parallel-Map and Join Helpers

Fan-out / fan-in is supported through the `parallel_map` and `join` decorators:

```python
from molexp.workflow import workflow, parallel_map, join, TaskContext

wf = workflow(name="sweep")

@wf.task
async def scatter(ctx: TaskContext) -> list[int]:
    return [1, 2, 3, 4]

@parallel_map(wf, fan_out_over="scatter")
async def compute(ctx: TaskContext) -> int:
    return ctx.inputs ** 2

@join(wf, depends_on=["compute"], reducer="sum")
async def reduce(ctx: TaskContext) -> int:
    return ctx.inputs
```

These are additive shortcuts over the normal `@wf.task` registration — they set per-task metadata (`fan_out_over`, `reducer`) that the runtime interprets during scheduling.

## Selection Guide

| Pattern | Use |
|---------|-----|
| One-shot computation, single value returned | `Task` |
| Streaming / continuous / event-driven | `Actor` |
| Third-party class you don't want to subclass | Structural `Runnable` / `Streamable` (just implement the method) |
| Fan-out over a list produced by an upstream | `@parallel_map(wf, fan_out_over=...)` |
| Fan-in reduction | `@join(wf, reducer=...)` |

For a broader walk-through, see the [Quick Start](../getting-started/quick-start.md). For the exact TaskContext API see [task-context.md](task-context.md).

## Runnable Example

`examples/workflow/task_and_actor.py` executes functional DSL, OOP builder, protocol form, and a streaming actor in one script.
