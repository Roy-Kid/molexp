# Workspace Architecture

MolExp organizes experiments in a four-tier hierarchy:

```
Workspace
└── Project
    └── Experiment      (bound to a WorkflowSpec + parameters + replica config)
        └── Run          (one execution; may be re-executed → ExecutionRecord)
```

Each level owns an `AssetLibrary` for scoped, content-addressed artifact storage. Every metadata write is atomic (temp-file + `os.rename`), so a crash never leaves a half-written JSON file.

## What Each Level Is

| Level | Directory | Metadata file | Purpose |
|-------|-----------|---------------|---------|
| `Workspace` | `<root>/` | `workspace.json` | Top-level container. Materialized by the constructor. |
| `Project` | `<root>/projects/<project_id>/` | `project.json` | Research-area container (MD, ML training, data pipeline, …). |
| `Experiment` | `<project>/experiments/<exp_id>/` | `experiment.json` | Repeatable workflow bound to a concrete parameter set + replica count. |
| `Run` | `<exp>/runs/run-<run_id>/` | `run.json` | Single execution instance; re-runs append `ExecutionRecord` entries. |

Children are **not** stored as lists in the parent metadata — parents discover children by scanning the filesystem. This keeps writes local and avoids lock contention.

## Why This Design

Scientific workflows tend to run the same pipeline many times with different parameters, then compare outcomes. The `Experiment → Run` split reflects that:

- **Experiment** is the *definition* — workflow source, parameters, replica count, seeds.
- **Run** is a *realization* — one execution, one set of concrete parameter values, one outcome.

Each `Run` captures reproducibility metadata: the bound workflow source (`WorkflowSnapshotRef`), the resolved molcfg profile, a `config_hash`, execution history, error info, and produced artifacts.

## Creating a Hierarchy

All child factories are **idempotent** (get-or-create by slug/ID) and **materialize immediately**. There is no separate "save" step for structure creation.

```python
import molexp as me

ws = me.Workspace("./lab")                                # materializes workspace.json
project = ws.project("MD Simulations")                    # materializes project.json
exp = project.experiment(
    "temperature-sweep",
    params={"T": 300, "pressure": 1.0},
    n_replicas=3,
    seeds=[42, 43, 44],
    workflow_source="workflows/md.py",
    git_commit="abc123",
)
run = exp.run(parameters={"T": 300, "pressure": 1.0})     # materializes run.json
```

Re-calling any factory with the same name / ID returns the **same** in-memory instance (preserving bound workflows and cached handles) and loads from disk if the directory already exists.

## Parameter Sweeps

`GridSpace` and `UniformSpace` generate parameter combinations. One combination = one `Experiment`:

```python
from molexp import GridSpace

grid = GridSpace({"T": [300, 310, 320], "force_field": ["amber", "charmm"]})

for params in grid:
    slug = f"T{params['T']}-{params['force_field']}"
    exp = project.experiment(slug, params=params, n_replicas=3, workflow_source="md.py")
    exp.set_workflow(spec)
    for seed in exp.get_seeds():
        run = exp.run(parameters={**params, "seed": seed})
```

`UniformSpace(param_values, n_samples, seed=None)` samples `n_samples` combinations uniformly at random — handy for broader search spaces.

## Binding a Workflow

```python
exp = project.experiment("baseline", params={"lr": 1e-3}, workflow_source="train.py")
exp.set_workflow(spec)          # WorkflowSpec
# or
exp.set_workflow(train_fn)      # bare fn(RunContext) — auto-promoted to a single-Task spec
```

`set_workflow` is write-once; calling it twice on the same Experiment raises `ValueError`.

## Executing a Run

```python
result = await exp.workflow.execute(run=run)
```

Under the hood, the runtime opens a `RunContext` which:

1. Ensures `artifacts/` and `logs/` subdirectories exist.
2. Claims exclusive ownership of the run directory.
3. Appends a new `ExecutionRecord` to `run.execution_history`.
4. Sets `run.status = "running"` and runs the workflow.
5. On success, writes `status="succeeded"` plus the final timestamp.
6. On failure, writes `status="failed"`, an `ErrorInfo`, and a traceback file under `execution/<exec_id>/`.

Every attempt appears in `run.metadata.execution_history`, newest last — a run that was retried twice will have three records.

## Assets

Each level owns an `AssetLibrary` (content-addressed, deduplicated). Lookup walks up the hierarchy automatically:

```python
ws.assets.import_asset("bert_model", "/models/bert.pt")
project.assets.import_asset("dataset", "/data/qm9.tar.bz2")
exp.assets.import_asset("features", "/data/features.h5")

# Inside a task
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        dataset = ctx.find_asset("dataset")       # searches experiment → project → workspace
        features = ctx.find_asset("features")
        ...
```

`import_asset(name, src, action="copy", meta=None)` supports `"copy"`, `"move"`, and `"link"` for ingestion.

## CLI Surface

The same hierarchy is exposed through the CLI:

```bash
molexp project   create|list|info
molexp experiment create|list
molexp runs      create|list|info|cancel|prune
molexp asset     list
molexp info      # show workspace summary
```

`molexp runs prune` interactively walks the project → experiment → run → execution tree and lets you delete per-execution records (removes `execution/<exec_id>/` and rewrites `run.execution_history`).

## Directory Layout

```
./lab/
├── workspace.json
├── assets/                         # workspace-level AssetLibrary
└── projects/
    └── qm9/
        ├── project.json
        ├── assets/                 # project-level AssetLibrary
        └── experiments/
            └── baseline/
                ├── experiment.json
                ├── assets/         # experiment-level AssetLibrary
                └── runs/
                    └── run-<id>/
                        ├── run.json
                        ├── artifacts/
                        ├── logs/
                        └── execution/<exec_id>/   # per-attempt error traces etc.
```
