# Control Flow

MolExp's workflow engine handles control flow through the **shape of the DAG**, not through special task types. There is no `IfTask`, `ForLoopTask`, or `MapTask` — parallelism, fan-out, and fan-in are expressed by how you wire `depends_on` edges, plus three compiler declarations: `wf.parallel` (fan out over a runtime-produced list), `wf.branch` (label-routed edges) and `wf.loop` (repeat a body until a condition task exits).

## Automatic Parallelism

Tasks whose dependencies are all satisfied run in parallel automatically. You don't mark anything as "parallel"; you just make sure they share the same set of upstream dependencies.

```python
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="pipeline")

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

For larger branch-specific pipelines, route between whole tasks with `wf.branch` (next section).

## Routed Branches (`wf.branch`)

When a decision selects between **whole downstream tasks**, declare label-routed edges with `wf.branch` and return `Next` from the deciding task:

```python
from molexp.workflow import Next, TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="triage", entry="classify")

@wf.task
async def classify(ctx: TaskContext) -> tuple[dict, Next]:
    score = run_model()
    return {"score": score}, Next("accept" if score > 0.5 else "reject")

@wf.task
async def accepted(ctx: TaskContext) -> dict:
    return ctx.inputs               # {"score": ...} — the routed value

@wf.task
async def rejected(ctx: TaskContext) -> None: ...

wf.branch("classify", routes={"accept": "accepted", "reject": "rejected"})

result = await WorkflowRuntime().execute(wf.compile())
```

Two declaration forms are equivalent: `wf.branch("src", routes={"l1": "t1", "l2": "t2"})` and the single-edge `wf.branch("src", "label", "target")`. The same routes can also be declared inline on the task — `@wf.task(routes={"accept": "accepted", ...})`. Route to the reserved name `"_end"` to terminate the workflow on that label.

The branch task returns a bare `Next("label")` or a `(value, Next("label"))` tuple. **Values ride the edges**: the routed target receives `value` as its `ctx.inputs` (a declared `depends_on` interface always wins — a target with data deps keeps its collected upstream shape). Returning an undeclared label raises `UnknownRouteError`; a task with declared routes that returns no `Next` raises `MissingRouteError`.

Un-routed targets never run. If a downstream task `depends_on` a task the branch routed *away* from, the engine detects this **structurally** and raises `WorkflowDeadlockError` (naming the unsatisfiable dependencies) the moment the consumer becomes control-ready — no timeouts; a genuinely slow upstream is never mistaken for a dead one. Design joins so every `depends_on` edge is on a live path for every route, or split the join per route.

## Loops

Sequential repetition that fits in one task body belongs **inside** the task (Python `for` / `while` is fine):

```python
@wf.task
async def iterate(ctx: TaskContext) -> list[float]:
    xs = ctx.inputs
    for _ in range(ctx.config.get("iters", 10)):
        xs = [x * 1.01 for x in xs]
    return xs
```

When each iteration is itself a (multi-task) piece of the graph, declare a workflow-level loop with `wf.loop`:

```python
from molexp.workflow import Next, TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="refine", entry="step")

@wf.task
async def step(ctx: TaskContext) -> int:
    prev = ctx.inputs if isinstance(ctx.inputs, int) else 0
    return prev + 1                 # ctx.inputs = previous iteration's value

@wf.task(depends_on=["step"])
async def check(ctx: TaskContext) -> tuple[int, Next]:
    n = ctx.inputs
    return n, Next("exit" if n >= 3 else "continue")

@wf.task
async def report(ctx: TaskContext) -> str:
    return f"final:{ctx.inputs}"    # ctx.inputs = the value check routed out

wf.loop(body=["step"], until="check", max_iters=10, on_exit="report")
```

`wf.loop(body=..., until=..., max_iters=..., on_exit=...)` semantics:

- `body` (list of task names) runs, then the `until` task decides: return `Next("continue")` to run the body again, or `Next("exit")` to proceed to `on_exit` (default `"_end"` — terminate).
- Loop-back values ride the edges: when `until` returns `(value, Next("continue"))`, the next iteration's body head receives `value` as its `ctx.inputs` (the first iteration sees `None`, or the entry inputs). The same `(value, Next("exit"))` value reaches the `on_exit` task. No shared mutable state is involved — accumulate by passing values forward.
- `max_iters` is a mandatory runaway guard: after the `until` task has dispatched `Next("continue")` `max_iters` times, the engine forces `Next("exit")` and emits a `LoopMaxItersExceeded` *warning* — the workflow completes rather than failing.

!!! warning "Loops and parallel joins don't fuse"
    A parallel-`join` and a loop-`until` cannot be fused onto the same task — use separate tasks (have the `wf.parallel` join feed a distinct `until` task).

## Fan-Out Over a Runtime List

Use `wf.parallel` when you need to fan out over a list produced by an upstream task:

```python
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="fan-out", entry="scatter")

@wf.task
async def scatter(ctx: TaskContext) -> list[int]:
    return [1, 2, 3, 4]

@wf.task
async def process(ctx: TaskContext) -> int:
    return ctx.inputs ** 2          # ctx.inputs is one fan-out element

@wf.task
async def reduce(ctx: TaskContext) -> int:
    return sum(ctx.inputs)          # collected outputs, one per element, in order

wf.parallel(map_over="scatter", body="process", join="reduce", max_concurrency=2)
```

The engine runs the `body` task once per element of `map_over`'s output (bounded by `max_concurrency`) and delivers the collected results to the `join` task. The fan-out is runtime-sized — the compiled task set stays exactly `{scatter, process, reduce}` with no per-element node growth. A `SubWorkflow` also slots in as the `body`, which is how you fan an entire inner pipeline out per element (see [Sub-workflows](subworkflows.md)). Per-element failures aggregate into a `ParallelExecutionError` while sibling elements still complete.

Use `wf.parallel` when the fan-out count is only known at runtime; prefer plain `depends_on` when it's known at authoring time.

## Pattern Selection

| Want | Use |
|------|-----|
| Same-time concurrent tasks | same-level `depends_on` — no extra config |
| Conditional logic inside one step | plain Python `if` inside the task |
| Selecting between downstream tasks | `wf.branch` + `(value, Next("label"))` |
| Iteration inside one step | plain Python `for` / `while` inside the task |
| Repeat a graph section until a condition | `wf.loop(body=..., until=..., max_iters=...)` |
| Fixed-size fan-out | `N` tasks authored at build time with identical `depends_on` |
| Runtime-sized fan-out | `wf.parallel(map_over=..., body=..., join=...)` |
| Long-running streaming processing | `Actor` (see [task-and-actor.md](task-and-actor.md#actor-streaming-tasks)) |

Explicit IR-level control-flow tasks (`IfTask`, `ForTask`, etc.) are **not part of the current API** and are not planned in the short term — the DAG shape plus the `wf.parallel` / `wf.branch` / `wf.loop` declarations cover the cases we've actually needed.

## Exporting the graph (UI / observability)

To export this control-flow topology — including the parallel fan-out edges —
for a UI canvas or observability tool, use `CompiledWorkflow.to_graph_ir()`, not
`to_ir()`. See [ir-export.md](ir-export.md) for which export to use when.

## Runnable Examples

- `examples/workflow/control_flow.py` runs a diamond, a conditional branch driven by `ctx.config`, a build-time fan-out, and a `wf.parallel` runtime fan-out.
- `examples/workflow/branch_and_loop.py` runs a `wf.branch` routed decision and a `wf.loop` refinement loop, with routed / loop-back values arriving via `ctx.inputs`.
