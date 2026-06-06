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
from molexp.workflow import workflow, TaskContext

wf = workflow(name="stream")

@wf.actor
async def monitor(ctx: TaskContext):
    for item in ctx.inputs:
        yield {"seen": item}
```

```python
# Subclass Actor
from molexp.workflow import Actor, TaskContext

class Monitor(Actor):
    async def run(self, ctx: TaskContext):
        for item in ctx.inputs:
            yield {"seen": item}
```

Any object with `async def run(self, ctx)` returning an async iterator satisfies the `Streamable` protocol and can be added via `WorkflowCompiler.add(obj)` — no molexp import required.

### Context and output

A streaming body receives the same `TaskContext` as a batch task (`ctx.state`, `ctx.deps`, `ctx.inputs`, `ctx.config`, `ctx.run_context`). The engine drives the async generator to exhaustion and records **the last yielded value** as the task's output; downstream tasks read that value from the shared results like any other. Streaming bodies are never cached (they are marked `is_actor` at compile time).

### Runtime Boundaries

Supported today:

- Structural detection of streaming tasks via `Streamable` (sets `is_actor`, which disables result caching for that task).
- Driving the async generator to completion and recording the last yielded value as the output.

Not implemented:

- Inter-task message-passing channels (`receive` / `send` / `emit`). An earlier, never-wired channel surface was removed — every path raised `NotImplementedError`. If you need streaming *between* concurrently-running tasks, open an issue; today an actor consumes `ctx.inputs` and yields outputs, it does not exchange messages mid-run with peers.

Relevant code: `molexp.workflow.task.Actor`, `molexp.workflow.context.TaskContext`, `molexp.workflow.protocols.Streamable`, and the drain loop in `molexp.workflow._pydantic_graph.node`.

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

wf = workflow(name="fan-out")

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
