# Workspace API

This reference covers the public surface of `molexp.workspace`: the four hierarchy classes (`Workspace`, `Project`, `Experiment`, `Run`), their factory methods, and the execution `RunContext`.

## Entry Points

All four entity types are re-exported from the top-level `molexp` package:

```python
import molexp as me

me.Workspace
me.Project
me.Experiment
me.Run
me.RunContext
me.GridSpace
me.UniformSpace
```

## `Workspace`

Top-level container. The constructor materializes `workspace.json` on demand the first time a child factory is called.

```python
ws = me.Workspace("./lab", name="Lab")           # or Workspace.from_path("./lab")
ws = me.Workspace.load("./lab")                  # raises if workspace.json missing
```

| Attribute / method | Description |
|--------------------|-------------|
| `ws.id`, `ws.name` | Stable slug + display name from `workspace.json` |
| `ws.root` | Resolved root path |
| `ws.assets` | Workspace-scoped `AssetLibrary` |
| `ws.project(name)` | Idempotent get-or-create of a `Project` |
| `ws.get_project(name_or_id)` | Returns existing project or `None` |
| `ws.list_projects()` | Disk scan merged with in-memory cache |
| `ws.delete_project(project_id)` | `shutil.rmtree` the project directory |
| `ws.materialize()` | Explicitly write `workspace.json` |
| `ws.save()` | Persist current metadata (rarely needed) |

## `Project`

Research-area container. Created exclusively through `ws.project(...)`.

| Attribute / method | Description |
|--------------------|-------------|
| `project.id`, `project.name`, `project.description`, `project.owner`, `project.tags`, `project.config` | From `project.json` |
| `project.project_dir` | `<workspace>/projects/<id>/` |
| `project.assets` | Project-scoped `AssetLibrary` |
| `project.experiment(name, *, id=None, params={}, n_replicas=1, seeds=None, workflow_source=None, workflow_type=None, git_commit=None)` | Idempotent get-or-create `Experiment` |
| `project.get_experiment(experiment_id)` | Returns existing experiment or `None` |
| `project.list_experiments()` | Disk scan + cache |
| `project.import_asset(name, src, action="copy", meta=None)` | Shortcut for `project.assets.import_asset(...)` |

## `Experiment`

Workflow definition bound to a concrete parameter set + replica configuration.

| Attribute / method | Description |
|--------------------|-------------|
| `exp.id`, `exp.name`, `exp.description`, `exp.tags` | Standard metadata |
| `exp.parameter_space` / `exp.params` | Frozen parameter dict |
| `exp.n_replicas`, `exp.seeds` | Replica configuration |
| `exp.get_seeds()` | Returns seeds list of length `n_replicas` (auto-generated from defaults if not provided) |
| `exp.workflow_source`, `exp.workflow` | Source file (string) and bound `WorkflowSpec` |
| `exp.experiment_dir` | `<project>/experiments/<id>/` |
| `exp.assets` | Experiment-scoped `AssetLibrary` |
| `exp.set_workflow(spec_or_fn)` | **Write-once** bind; accepts a `WorkflowSpec` or bare `fn(RunContext)` (auto-promoted to a single-Task spec) |
| `exp.run(parameters=None, *, id=None)` | Idempotent get-or-create `Run` |
| `exp.get_run(run_id)` | Returns existing run or `None` |
| `exp.list_runs()` | Disk scan |

## `Run`

Single execution instance. Created exclusively through `exp.run(...)`.

Key fields (on `run.metadata` — a pydantic `RunMetadata`):

| Field | Type | Meaning |
|-------|------|---------|
| `id` | `str` | Run identifier (used in the directory name) |
| `status` | `str` | `"pending"` / `"running"` / `"succeeded"` / `"failed"` / `"cancelled"` |
| `parameters` | `dict` | Concrete parameter dict for this run |
| `workflow_snapshot` | `WorkflowSnapshotRef \| None` | Frozen reference to the workflow source + git commit at creation time |
| `profile` | `str \| None` | Activated molcfg profile name |
| `config` | `dict` | Frozen merged profile data |
| `config_hash` | `str \| None` | sha256 of `config`, pre-computed for queryability |
| `execution_history` | `list[ExecutionRecord]` | One entry per attempt (re-run → append) |
| `error` | `ErrorInfo \| None` | Structured error summary on failure |

Direct `run.parameters`, `run.status`, `run.run_dir`, etc., are proxied from `run.metadata`.

## `RunContext`

The execution-time context manager. Normally you don't construct one directly — the workflow runtime opens it for you when you pass `run=` to `spec.execute(...)`. You can also open one explicitly for manual control:

```python
with run.start(profile_config=cfg) as ctx:
    # Inside this block: run.status == "running"
    ...
# On exit: status = "succeeded" or "failed", execution history updated
```

Inside a task, `ctx: TaskContext` exposes the `RunContext` through `ctx.run_context`, plus convenience helpers that no-op when no run is attached:

```python
ctx.save_artifact("output.pt", data)      # → <run_dir>/artifacts/output.pt
ctx.get_artifact_path("output.pt")
ctx.find_asset("dataset")                 # walks experiment → project → workspace
ctx.checkpoint("pre-eval")                # writes a JSON snapshot under .ckpt/
ctx.set_result("best_loss", 0.02)
ctx.get_result("best_loss")
```

## Assets

`AssetLibrary` is the scoped store at every level. Public surface:

```python
lib = project.assets
lib.import_asset(name, src, action="copy", meta=None)  # copy | move | link
lib.get_asset(name)
lib.list_assets()
```

## Parameter Spaces

```python
from molexp import GridSpace, UniformSpace

grid = GridSpace({"lr": [1e-3, 1e-4], "batch": [32, 64]})
len(grid)  # 4
list(grid)

uniform = UniformSpace({"lr": [1e-3, 5e-4, 1e-4]}, n_samples=10, seed=42)
```

Parameter expansion is user-driven: iterate the space at script level and materialize one `Experiment` per combination. The library never auto-creates experiments for you.

## Putting It Together

```python
import molexp as me
from molexp.workflow import workflow, TaskContext

wf = workflow(name="demo")

@wf.task
async def compute(ctx: TaskContext) -> float:
    return ctx.config.get("lr", 1e-3) * 2

spec = wf.build()

ws = me.Workspace("./lab")
project = ws.project("demo")
exp = project.experiment(
    "baseline",
    params={"lr": 1e-3},
    n_replicas=1,
    workflow_source="demo.py",
)
exp.set_workflow(spec)

run = exp.run(parameters={"lr": 1e-3})
result = await spec.execute(run=run)
print(run.status, result.outputs)
```
