# Control Flow

MolExp's workflow engine handles control flow through the **shape of the DAG**, not through special task types. There is no `IfTask`, `ForLoopTask`, or `MapTask` — parallelism, fan-out, and fan-in are expressed by how you wire `depends_on` edges, with two opt-in decorators (`parallel_map`, `join`) when you need to fan out over a runtime-produced list.

## Automatic Parallelism

Tasks at the same topological level run in parallel automatically. You don't mark anything as "parallel"; you just make sure they share the same set of upstream dependencies.

```python
from molexp.workflow import workflow, TaskContext

wf = workflow(name="pipeline")

@wf.task
async def fetch(ctx: TaskContext) -> dict:
    return {"data": load()}

@wf.task(depends_on=["fetch"])
async def parse(ctx: TaskContext) -> dict: ...

@wf.task(depends_on=["fetch"])
async def validate(ctx: TaskContext) -> dict: ...

@wf.task(depends_on=["parse", "validate"])
async def merge(ctx: TaskContext) -> dict: ...
```

`parse` and `validate` run concurrently once `fetch` finishes. `merge` waits for both. This is the idiomatic way to express "run these in parallel, then reduce".

## Conditional Execution

Use a plain `if` inside a task. If a branch has no work, return a sentinel / `None`:

```python
@wf.task(depends_on=["fetch"])
async def maybe_clean(ctx: TaskContext) -> dict:
    if ctx.config.get("skip_cleaning", False):
        return ctx.inputs
    return clean(ctx.inputs)
```

For larger branch-specific pipelines, keep each branch as its own task and wire `depends_on` accordingly; skipped branches still schedule but can short-circuit cheaply from `ctx.inputs`.

## Loops

Sequential repetition belongs **inside** a task (Python `for` / `while` is fine):

```python
@wf.task
async def iterate(ctx: TaskContext) -> list[float]:
    xs = ctx.inputs
    for _ in range(ctx.config.get("iters", 10)):
        xs = [x * 1.01 for x in xs]
    return xs
```

Loops that should themselves parallelize across runs live one level up: at the sweep layer. See `molexp.sweep.run_sweep` for fan-out over `(experiment, Run)` pairs with a bounded `jobs` semaphore.

## Fan-Out Over a Runtime List

Use the `parallel_map` decorator when you need to fan out over a list produced by an upstream task:

```python
from molexp.workflow import workflow, parallel_map, join, TaskContext

wf = workflow(name="sweep")

@wf.task
async def scatter(ctx: TaskContext) -> list[int]:
    return [1, 2, 3, 4]

@parallel_map(wf, fan_out_over="scatter")
async def process(ctx: TaskContext) -> int:
    return ctx.inputs ** 2

@join(wf, depends_on=["process"], reducer="sum")
async def reduce(ctx: TaskContext) -> int:
    return ctx.inputs
```

The decorators set per-task metadata (`_parallel_map_config`, `_join_config`) the runtime reads during scheduling. Use them when the fan-out count is only known at runtime; prefer plain `depends_on` when it's known at authoring time.

## What About …?

| Want | Use |
|------|-----|
| Same-time concurrent tasks | same-level `depends_on` — no extra config |
| Conditional branch | plain Python `if` inside the task |
| Fixed-size fan-out | `N` tasks authored at build time with identical `depends_on` |
| Runtime-sized fan-out | `@parallel_map(wf, fan_out_over=...)` + `@join(wf, reducer=...)` |
| Sweep across `(experiment, Run)` pairs | `molexp.sweep.run_sweep` |
| Long-running streaming processing | `Actor` (see [actors.md](actors.md)) |

Explicit IR-level control-flow tasks (`IfTask`, `ForTask`, etc.) are **not part of the current API** and are not planned in the short term — the DAG shape + decorators cover the cases we've actually needed.
