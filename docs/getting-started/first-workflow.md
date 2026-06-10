# Your First Workflow

Before MolExp becomes a workspace or a CLI tool, it is a workflow system. The first thing to understand is that a workflow is just a compiled graph of typed computation steps. You can build and run that graph without creating a workspace at all.

## Writing the First Task

A task is an async computation that receives a `TaskContext`. The context gives the task its upstream input (`ctx.inputs`) and the active configuration mapping (`ctx.config`).

```python
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="pipeline")


@wf.task
async def fetch(ctx: TaskContext) -> dict:
    return {"value": 42}


@wf.task(depends_on=["fetch"])
async def scale(ctx: TaskContext) -> int:
    factor = ctx.config.get("factor", 2)
    return ctx.inputs["value"] * factor
```

The important thing here is not the arithmetic. It is the dependency line. `scale` depends on `fetch`, so its `ctx.inputs` is the output of that upstream task.

## Compiling the Graph

Once the tasks have been declared, `wf.compile()` turns the definition into a frozen `CompiledWorkflow`:

```python
compiled = wf.compile()
```

That compiled artifact is what MolExp executes. It has a deterministic `workflow_id`, a validated dependency structure, and per-task content snapshots that let the runtime and the cache reason about the graph consistently.

## Running Without a Workspace

Execution lives on `WorkflowRuntime`, not on the artifact. The compiled workflow can run purely in memory:

```python
import asyncio

from molexp.workflow import WorkflowRuntime


async def main() -> None:
    result = await WorkflowRuntime().execute(compiled)
    print(result.status, result.outputs)


asyncio.run(main())
```

This mode is useful during early iteration because it lets you work on task boundaries and data flow without also thinking about projects, experiments, or run directories.

## Using Task Classes Instead of Inline Functions

The decorator style is the shortest path for small workflows, but it is not the only authoring style. If you want reusable task classes, the same `WorkflowCompiler` accepts instances through `.add(...)`:

```python
from molexp.workflow import Task, TaskContext, WorkflowCompiler


class Fetch(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        return {"value": 42}


class Scale(Task):
    async def execute(self, ctx: TaskContext) -> int:
        return ctx.inputs["value"] * 2


compiled = (
    WorkflowCompiler(name="pipeline")
    .add(Fetch())
    .add(Scale(), depends_on=["fetch"])
    .compile()
)
```

Both styles produce the same kind of compiled workflow. The choice is mostly about how you want to organize code.

## What to Learn Next

Once the workflow model itself is clear, the next question is usually what happens when you want that execution to leave a durable record behind. That is the point where the workspace layer becomes relevant, so the next page to read is [Track a Run](tracked-runs.md).

## Runnable Example

`examples/getting_started/02_first_workflow.py` compiles and runs this kind of workspace-less workflow as a single script.
