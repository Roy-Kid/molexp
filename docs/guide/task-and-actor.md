# Task and Actor

`Task` and `Actor` are the atomic units of a MolExp workflow. `Task` runs once and returns a value; `Actor` runs continuously and yields a stream of values. Both have a **protocol** form (zero-import third-party integration) and a **convenience base class** form (`molexp.workflow.Task` / `Actor`).

## Task Semantics

A `Task` is a single async function (or class with `async def execute(self, ctx)`) that consumes the typed output of its upstream task and returns a typed output of its own. The compiled graph runs tasks as soon as their dependencies are satisfied — tasks with no unresolved dependency between them run in parallel.

Three equivalent ways to define a task, all on the same `WorkflowCompiler`:

```python
# 1. Function decorated with @wf.task
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="pipeline")

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

For instance registration, chain `.add(...)` calls and finish with `.compile()`:

```python
from molexp.workflow import WorkflowCompiler

compiled = (
    WorkflowCompiler(name="pipeline")
    .add(Fetch())                                   # auto-named "fetch"
    .add(Process(), depends_on=["fetch"])           # name inferred from class
    .add(Report(), depends_on=["process"], name="report")
    .compile()
)
```

## Generic Parameters

`Task` is generic over `StateT, InputT, OutputT`. Fill them in only when you want full static typing:

```python
class Fetch(Task[None, str, dict]):
    async def execute(self, ctx: TaskContext[None, str]) -> dict:
        # the source path arrives as an input, not via ambient deps
        return read_records(ctx.inputs)
```

Plain `Task` (no generics) defaults to `Any` everywhere. Build-time configuration is the task instance's own `__init__` arguments — captured automatically as the task's config identity (the cache and the IR both key on it), and read inside the body as plain `self.*` attributes.

## Actor — Streaming Tasks

`Actor` is the streaming variant. Its `run()` returns an async iterator; the compiler detects it structurally via the `Streamable` protocol. Use actors for continuous sampling, stream transformation, and components that run at a different rate than their peers. If your computation is a single call returning one value, use `Task` — actors are more machinery than you need.

```python
# Decorator style
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="stream")

@wf.actor(depends_on=["load"])
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

A streaming body receives the same `TaskContext` as a batch task (`ctx.inputs`, `ctx.config`, `ctx.workdir`). The engine drives the async generator to exhaustion and records **the last yielded value** as the task's output; downstream tasks read that value from the shared results like any other. Streaming bodies are never cached (they are marked `is_actor` at compile time).

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
| `WorkflowCompiler.add(FetchTask())` | Class name converted to snake_case, minus trailing `_task` / `_actor` → `fetch` |
| `WorkflowCompiler.add(Fetch(), name="X")` | Explicit `name=` |

`depends_on` values must match one of these resolved names exactly.

## Runtime Fan-Out and Fan-In

Fan-out over a runtime-produced list is declared with `wf.parallel`:

```python
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="fan-out", entry="scatter")

@wf.task
async def scatter(ctx: TaskContext) -> list[int]:
    return [1, 2, 3, 4]

@wf.task
async def compute(ctx: TaskContext) -> int:
    return ctx.inputs ** 2          # ctx.inputs is one element

@wf.task
async def reduce(ctx: TaskContext) -> int:
    return sum(ctx.inputs)          # one output per element, in order

wf.parallel(map_over="scatter", body="compute", join="reduce", max_concurrency=2)
```

The runtime runs `compute` once per element of `scatter`'s output and delivers the collected results to `reduce`. See [Control Flow](control-flow.md) for branches and loops.

## Selection Guide

| Pattern | Use |
|---------|-----|
| One-shot computation, single value returned | `Task` |
| Streaming / continuous / event-driven | `Actor` |
| Third-party class you don't want to subclass | Structural `Runnable` / `Streamable` (just implement the method) |
| Fan-out over a list produced by an upstream | `wf.parallel(map_over=..., body=..., join=...)` |

For a broader walk-through, see the [Quick Start](../getting-started/quick-start.md). For the exact TaskContext API see [task-context.md](task-context.md).

## Runnable Example

`examples/workflow/task_and_actor.py` executes the decorator style, OOP registration, protocol form, and a streaming actor in one script.
