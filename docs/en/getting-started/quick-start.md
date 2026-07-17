# Quick Start

This page gets you from zero to a tracked run in under a minute. You will define a two-task workflow, create a workspace on disk, execute the workflow, and read back the result.

## The Script

Copy this into a file named `demo.py`:

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

# 1. Define the workflow
wf = WorkflowCompiler(name="sum")

@wf.task
def fetch(scale: float = 1.0) -> dict:
    return {"values": [1.0, 4.0, 9.0], "scale": scale}

@wf.task(depends_on=["fetch"])
def summarize(values: list[float], scale: float = 1.0) -> float:
    return sum(values) * scale

# 2. Create the workspace hierarchy
ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("sum").add_run(params={"scale": 2.0})

# 3. Execute and read the result
result = run.execute(wf)
print(run.status, result.outputs["summarize"])
```

Run it:

```bash
python demo.py
```

The output is `succeeded 28.0`.

## What Just Happened

**Step 1 — Define.** `WorkflowCompiler` holds task definitions. `@wf.task` turns a plain function into a workflow node. `depends_on=["fetch"]` tells the engine that `summarize` runs after `fetch` and receives its output.

**Step 2 — Create.** `Workspace("./lab")` creates a directory on disk. The fluent chain `.project("demo").experiment("sum").add_run(params={"scale": 2.0})` builds the persistent hierarchy: a project groups related work, an experiment names one repeatable definition, and a run records one concrete execution with its parameters.

**Step 3 — Execute.** `run.execute(wf)` does everything: compiles the workflow, opens the run's tracked lifecycle, executes the graph with the run's params bound to the root task, persists every task's output under the run directory, and returns the result.

## How Data Flows

The secret is **name binding**. The run's `params={"scale": 2.0}` binds to `fetch`'s `scale` parameter because they share the same name. `fetch` returns a dict with keys `values` and `scale` — those keys bind to `summarize`'s `values` and `scale` parameters. A parameter with no upstream match falls back to its declared default.

```
params={"scale": 2.0}
        │
        ▼
     fetch(scale=2.0)  →  {"values": [1,4,9], "scale": 2.0}
        │                      │            │
        │   binds by name ─────┘            │
        ▼                                   │
  summarize(values=[1,4,9], scale=2.0)  ←───┘
        │
        ▼
      28.0
```

## The Result Stays on Disk

After the script exits, the run is still there. Open a new Python session and read it back:

```python
# The run persists on disk — open the same workspace and read it back.
# run.id was printed above; use it here.
same_run = ws.project("demo").experiment("sum").get_run(run.id)
print(same_run.status)                     # succeeded
print(same_run.get_result("summarize"))    # 28.0
```

`get_run(params=...)` rediscovers the run by its content-addressed identity — the same params always resolve to the same run.

## Next Steps

- If the workflow definition was the unfamiliar part, continue with [Your First Workflow](first-workflow.md).
- If the workspace hierarchy is new to you, read [Track a Run](tracked-runs.md).
- If you want `molexp run` to drive this instead of calling `run.execute` yourself, go to [CLI and Profiles](cli-and-profiles.md).
- If you prefer clicking to scripting, try [Start from the UI](start-from-ui.md).

## Runnable Example

`examples/getting_started/01_quick_start.py` is a stand-alone version of this script.
