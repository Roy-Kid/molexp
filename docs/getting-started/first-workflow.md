# Your First Workflow

Before MolExp becomes a workspace or a CLI tool, it is a workflow system. The first thing to understand is that a workflow is just a compiled graph of computation steps. You can build and run that graph without creating a workspace at all.

## Writing the First Task

A task is an ordinary function — a plain synchronous `def` for pure computation, or an `async def` when the body genuinely awaits something. It declares the runtime values it consumes as named parameters, and the engine binds them by name: an upstream task's output and any build-time configuration are matched to parameters that share their name. The leading `ctx` parameter is optional — a task keeps it only when it needs a scratch directory (`ctx.workdir`), which is the one data surface left on the `TaskContext`. A task that just transforms data, like the two below, omits `ctx` entirely.

```python
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="pipeline")


@wf.task
def fetch() -> dict:
    return {"value": 42}


@wf.task(depends_on=["fetch"])
def scale(value: int, factor: int = 2) -> int:
    return value * factor
```

The important thing here is not the arithmetic. It is the dependency line. `scale` depends on `fetch`, so `fetch`'s output — the `{"value": 42}` dictionary — is bound into `scale`'s parameters by name: the `value` key fills the `value` parameter. The `factor` parameter has no upstream match, so it falls back to its declared default of `2`; build-time configuration could supply a different `factor`, and that dynamic value would win over the default.

Sync bodies run in a worker thread, so a blocking computation never stalls tasks scheduled alongside it. Reach for `async def` only when the body itself awaits — the two styles mix freely in one graph.

## Compiling the Graph

Once the tasks have been declared, `wf.compile()` turns the definition into a frozen `CompiledWorkflow`:

```python
compiled = wf.compile()
```

That compiled artifact is what MolExp executes. It has a deterministic `workflow_id`, a validated dependency structure, and per-task content snapshots that let the runtime and the cache reason about the graph consistently. (The one-step entry points compile for you when handed a `WorkflowCompiler`, but compiling explicitly is how you validate the graph early.)

## Running Without a Workspace

The compiled workflow can run purely in memory through the async `WorkflowRuntime` — no projects, experiments, or run directories involved:

```python
import asyncio

from molexp.workflow import WorkflowRuntime

result = asyncio.run(WorkflowRuntime().execute(compiled))
print(result.status, result.outputs)
```

This mode is useful during early iteration because it lets you work on task boundaries and data flow without also thinking about persistence. As soon as you want a durable record, hand the same object to a tracked run instead — `run.execute(compiled)` on the next page folds the runtime, the run lifecycle, and the asyncio plumbing into one call.

## Mixing Sync and Async Tasks

An `async def` task drops into the same graph unchanged:

```python
@wf.task(depends_on=["scale"])
async def publish(scale: int) -> str:
    return f"published {scale}"


compiled = wf.compile()
result = asyncio.run(WorkflowRuntime().execute(compiled))
print(result.outputs["publish"])
```

## Using Task Classes Instead of Inline Functions

The decorator style is the shortest path for small workflows, but it is not the only authoring style. If you want reusable task classes, the same `WorkflowCompiler` accepts instances through `.add(...)`. The `execute` body may be sync or async, exactly like the decorator form:

```python
from molexp.workflow import Task, TaskContext


class Fetch(Task):
    def execute(self, ctx: TaskContext) -> dict:
        return {"value": 42}


class Scale(Task):
    def execute(self, ctx: TaskContext, value: int, factor: int = 2) -> int:
        return value * factor


compiled_oop = (
    WorkflowCompiler(name="pipeline-oop")
    .add(Fetch())
    .add(Scale(), depends_on=["fetch"])
    .compile()
)
```

Both styles produce the same kind of compiled workflow. The class form keeps the `ctx` parameter on `execute` by convention; the input parameters (`value`, `factor`) follow it and bind by name exactly as they do for the decorator form. The choice is mostly about how you want to organize code.

## What to Learn Next

Once the workflow model itself is clear, the next question is usually what happens when you want that execution to leave a durable record behind. That is the point where the workspace layer becomes relevant, so the next page to read is [Track a Run](tracked-runs.md).

## Runnable Example

`examples/getting_started/02_first_workflow.py` compiles and runs this kind of workspace-less workflow as a single script.
