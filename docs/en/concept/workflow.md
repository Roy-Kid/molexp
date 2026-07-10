# Workflow Model

The workflow layer is the computation graph. It answers *what should happen* — task boundaries, dependency order, and data flow. It does **not** know about project grouping, run directories, or scheduler transport.

## Author, compile, execute

MolExp separates a workflow's lifecycle into three stages:

| Stage | Tool | What happens |
|---|---|---|
| **Author** | `WorkflowCompiler` | Declare tasks and dependencies |
| **Compile** | `.compile()` | Freeze into a validated `CompiledWorkflow` |
| **Execute** | `WorkflowRuntime` or `Run.execute()` | Drive the graph |

You can author with decorators:

```python
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="demo")

@wf.task
def fetch() -> list[float]:
    return [1.0, 4.0, 9.0]

@wf.task(depends_on=["fetch"])
def summarize(data: list[float]) -> float:
    return sum(data)

compiled = wf.compile()
```

Or with reusable task classes — both produce the same kind of `CompiledWorkflow`.

## Execution is independent of persistence

A compiled workflow can run purely in memory:

```python
import asyncio
from molexp.workflow import WorkflowRuntime

result = asyncio.run(WorkflowRuntime().execute(compiled))
```

Or under a tracked run with full persistence:

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("baseline").add_run(params={})
result = run.execute(wf)
```

The graph is the same. Only the lifecycle around it changes.

## What stays outside

The workflow layer deliberately does **not** know about:

- Project/experiment grouping (that's workspace)
- Run directories and execution history (workspace)
- Scheduler transport (plugins)
- Shared datasets and derived resources (assets)

That narrow boundary keeps workflows reusable — the same compiled graph runs locally, under `molexp run`, or from a remote worker.

## Data flow: name binding

Values move between tasks by **name**. An upstream task's return-value keys bind to the downstream task's parameters that share the same name. Parameters with no upstream match fall back to defaults. Build-time configuration (`params`) can override defaults.

```python
@wf.task
def source() -> dict:
    return {"x": 10, "y": 20}

@wf.task(depends_on=["source"])
def consumer(x: int, y: int, z: int = 0) -> int:
    return x + y + z  # x=10 (from source), y=20 (from source), z=0 (default)
```

## Next

For the persistent side of this story, read [Workspace](workspace.md). For reusable data and provenance, read [Assets and Reproducibility](assets-and-reproducibility.md).
