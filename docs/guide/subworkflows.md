# Workflow Composition

`SubWorkflow` is the **sanctioned composition node** in MolExp. It wraps a
reusable inner workflow and runs it end-to-end through the engine as a single
node of an outer workflow — including as the per-element `body` of
`builder.parallel`.

Do **not** hand-build a child `TaskContext` and call an inner task's `execute`
directly. Cloning `TaskContext` by hand is unsupported: it bypasses the engine,
duplicates step definitions, and can drift from the engine's context contract.
`SubWorkflow` routes the nested call through a real `WorkflowRuntime` execution
and forwards the outer `run_context` **by identity**, so inner-task workspace
helpers keep working against the same workspace / run.

## Pattern 1: `SubWorkflow` as one node of a chain

```python
from molexp.workflow import (
    SubWorkflow,
    Task,
    TaskContext,
    WorkflowCompiler,
    WorkflowRuntime,
)

# Inner pipeline (a WorkflowCompiler — compiled eagerly when wrapped).
inner = WorkflowCompiler(name="preprocess")

@inner.task
async def load(ctx: TaskContext) -> list[float]:
    return [3.0, 1.0, 4.0, 1.0, 5.0]

@inner.task(depends_on=["load"])
async def normalize(ctx: TaskContext) -> list[float]:
    return [x / max(ctx.inputs) for x in ctx.inputs]

class Train(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return sum(ctx.inputs) / len(ctx.inputs)

outer = (
    WorkflowCompiler(name="train")
    .add(SubWorkflow(inner), name="preprocess")
    .add(Train(), depends_on=["preprocess"])
    .compile()
)

result = await WorkflowRuntime().execute(outer)
```

The outer workflow sees `preprocess` as a single task; the inner workflow keeps
its own topology, `workflow_id`, and compile-time validation. By default
`SubWorkflow` returns the inner spec's single **dependency leaf** (the task no
other task depends on — `normalize` above). Pass `output="<task_name>"` to
select a different inner output. If the inner spec has more than one leaf and no
`output=` is given, `execute` raises a `ValueError` naming the candidates.

`SubWorkflow(inner)` accepts either a `WorkflowCompiler` (compiled on
construction) or an already-compiled `CompiledWorkflow`.

## Pattern 2: `SubWorkflow` as a `parallel` body

Because a `SubWorkflow` is a single registered task from the outer graph's
perspective, it slots straight into `builder.parallel(body=...)` — no change to
`ParallelDecl`, the compiler, or the plan lowering. The outer engine fans out the
single `SubWorkflow` node per element, and the node runs the full inner chain
for each element. The compiled task set stays exactly the declared outer tasks
(no per-element node growth).

```python
wf = WorkflowCompiler(name="fanout", entry="enumerate")

@wf.task
async def enumerate(ctx: TaskContext) -> list[int]:
    return [0, 1, 2]

wf.add(SubWorkflow(inner), name="preprocess")

@wf.task
async def collect(ctx: TaskContext) -> list[list[float]]:
    return list(ctx.inputs)

wf.parallel(map_over="enumerate", body="preprocess", join="collect", max_concurrency=2)

compiled = wf.compile()
result = await WorkflowRuntime().execute(compiled)
# `collect` receives one inner output per element, in iteration order.
```

A per-element inner failure surfaces through the engine's existing
`ParallelExecutionError` aggregation while sibling elements still complete.

## Pattern 3: Share task classes across compilers

Since tasks are plain classes that satisfy the `Runnable` protocol, the same
class can appear in multiple workflows — the lightest form of reuse when you do
not need a whole sub-pipeline as one node:

```python
baseline = (
    WorkflowCompiler(name="baseline")
    .add(Fetch())
    .add(Clean(), depends_on=["fetch"])
    .compile()
)

augmented = (
    WorkflowCompiler(name="augmented")
    .add(Fetch())
    .add(Clean(), depends_on=["fetch"])
    .add(Augment(), depends_on=["clean"])
    .compile()
)
```

## Runnable Example

`examples/workflow/subworkflows.py` shows both `SubWorkflow` shapes: as one node
of an outer chain, and as the per-element body of `builder.parallel`.
