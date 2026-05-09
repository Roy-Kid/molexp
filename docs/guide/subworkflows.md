# Workflow Composition

There is **no dedicated `SubWorkflow` type** in MolExp. Composition is done by calling a `Workflow.execute(...)` from inside another task, or by sharing task classes / DSL fragments between multiple `WorkflowBuilder` instances. This page shows the three composition patterns the library currently supports.

## Pattern 1: Call a sub-spec from a task

Wrap the inner workflow in a task and let the outer runtime treat it as an opaque node:

```python
from molexp.workflow import workflow, Task, TaskContext, WorkflowBuilder

# Inner pipeline
inner_wf = workflow(name="preprocess")

@inner_wf.task
async def normalize(ctx: TaskContext) -> list[float]:
    return [x / max(ctx.inputs) for x in ctx.inputs]

inner_spec = inner_wf.build()

# Outer pipeline that delegates to the inner one
class Preprocess(Task):
    def __init__(self, spec):
        self._spec = spec
    async def execute(self, ctx: TaskContext) -> list[float]:
        result = await self._spec.execute(run=ctx.run_context.run if ctx.run_context else None)
        return result.outputs["normalize"]

outer = (
    WorkflowBuilder(name="train")
    .add(Preprocess(inner_spec), name="preprocess")
    .add(Train(), depends_on=["preprocess"])
    .build()
)
```

The outer workflow sees `preprocess` as a single task; the inner workflow gets its own topology and its own compile-time validation. Caching, parallelism, and cancellation apply independently at each level.

## Pattern 2: Share task classes across builders

Since tasks are plain classes that satisfy the `Runnable` protocol, the same class can appear in multiple workflows:

```python
class Fetch(Task):
    async def execute(self, ctx: TaskContext) -> dict: ...

class Clean(Task):
    async def execute(self, ctx: TaskContext) -> dict: ...

baseline = (
    WorkflowBuilder(name="baseline")
    .add(Fetch())
    .add(Clean(), depends_on=["fetch"])
    .build()
)

augmented = (
    WorkflowBuilder(name="augmented")
    .add(Fetch())
    .add(Clean(), depends_on=["fetch"])
    .add(Augment(), depends_on=["clean"])
    .build()
)
```

Each builder produces a distinct `Workflow` with its own `workflow_id`; the shared classes keep the code base DRY without introducing runtime coupling.

## Pattern 3: Factor a builder helper

For larger shared fragments, factor out a helper that mutates a builder:

```python
def with_preprocessing(builder: WorkflowBuilder) -> WorkflowBuilder:
    return (
        builder
        .add(Fetch())
        .add(Clean(), depends_on=["fetch"])
        .add(Normalize(), depends_on=["clean"])
    )

train = with_preprocessing(WorkflowBuilder(name="train")).add(Train(), depends_on=["normalize"]).build()
eval_ = with_preprocessing(WorkflowBuilder(name="eval")).add(Evaluate(), depends_on=["normalize"]).build()
```

This is the most lightweight way to reuse a DAG fragment; you stay inside the plain DSL and don't pay any wrapper overhead.

## Cases That May Require a True Subworkflow Type

If you find yourself:

- wanting a single **shared** cache entry for a reusable fragment across workflows, or
- needing to replay the inner graph independently from the outer one,

open an issue describing the workload. The current runtime handles every practical case we have encountered through the three patterns above; a formal `SubWorkflow` type would be added only if there's a concrete caching / observability need it solves.

## Runnable Example

`examples/workflow/subworkflows.py` wraps an inner preprocess spec inside an outer training workflow — the "Pattern 1" shape above.
