# Workspace Architecture

MolExp organizes experiments in a four-tier hierarchy:

```
Workspace
└── Project
    └── Experiment      (bound to a WorkflowSpec + parameters + replica config)
        └── Run          (one execution; may be re-executed → ExecutionRecord)
```

Every persistent byproduct — imported data, task artifacts, logs, checkpoints, error traces, workflow execution state — is a typed `Asset` subclass recorded in a per-scope `assets.json` manifest and indexed by the workspace catalog at `.catalog/index.json`. Every metadata write is atomic (temp-file + `os.rename`), so a crash never leaves a half-written JSON file.

## Hierarchy Levels

| Level | Directory | Metadata file | Purpose |
|-------|-----------|---------------|---------|
| `Workspace` | `<root>/` | `workspace.json` | Top-level container. Materialized explicitly or when the first child is created. |
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

The hierarchy is created from the top down, but not every step has identical identity rules. `ws.project(...)` and `project.experiment(...)` are get-or-create operations keyed by slug or explicit id, so repeated calls can load existing objects from disk. `exp.run(...)` is different: it creates a fresh run unless you provide an explicit `id`, in which case it becomes a get-or-load operation for that concrete run directory.

```python
import molexp as me

ws = me.Workspace("./lab")                                # lightweight object; no files yet
project = ws.project("MD Simulations")                    # materializes workspace.json and project.json
exp = project.experiment(
    "temperature-sweep",
    params={"T": 300, "pressure": 1.0},
    n_replicas=3,
    seeds=[42, 43, 44],
    workflow_source="workflows/md.py",
    git_commit="abc123",
)
run = exp.run(
    parameters={"T": 300, "pressure": 1.0},
    id="temperature-sweep-seed-42",
)                                                         # materializes run.json
```

Re-calling the project or experiment factory with the same name or id returns the same in-memory object within the current process and loads from disk when needed. Runs only behave that way when the run id is stable.

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

1. Ensures `artifacts/`, `logs/`, and `.ckpt/` subdirectories exist.
2. Records temporary ownership metadata for the active execution.
3. Appends a new `ExecutionRecord` to `run.execution_history`.
4. Sets `run.status = "running"` and runs the workflow.
5. On success, writes `status="succeeded"` plus the final timestamp.
6. On failure, writes `status="failed"`, an `ErrorInfo`, and registers an `ErrorTraceAsset` pointing at `execution/<exec_id>/error.txt`.

Every attempt appears in `run.metadata.execution_history`, newest last — a run that was retried twice will have three records.

## Assets

Every scope exposes a typed `assets` view into the workspace catalog and a `data_assets` library for importing external data. Run-time writes (artifacts, logs, checkpoints) go through the `RunContext` accessors, which keep the on-disk manifest and the catalog in sync.

```python
ws.data_assets.import_asset("bert_model", "/models/bert.pt")
project.data_assets.import_asset("dataset", "/data/qm9.tar.bz2")
exp.data_assets.import_asset("features", "/data/features.h5")

# Inside a task
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        dataset = ctx.find_asset("dataset")       # walks run → experiment → project → workspace
        features = ctx.find_asset("features")
        ctx.artifact.save("metrics.json", {"val_loss": 0.1})
        ctx.log("train").append("epoch 1")
        ...
```

`import_asset(name, src, action="copy", meta=None)` supports `"copy"`, `"move"`, `"symlink"`, and `"hardlink"` for ingestion. See [Unified Asset Model](assets.md) for the full list of asset kinds (artifact, log, checkpoint, error trace, execution state, output, data) and how the catalog can be rebuilt from disk when needed.

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
├── .catalog/index.json             # derived workspace-wide catalog
├── assets.json                     # workspace-scoped asset manifest
├── data_assets/<asset_id>/payload/ # imported DataAssets
└── projects/
    └── qm9/
        ├── project.json
        ├── assets.json             # project-scoped asset manifest
        ├── data_assets/            # project-scoped DataAssets
        └── experiments/
            └── baseline/
                ├── experiment.json
                ├── assets.json     # experiment-scoped asset manifest
                ├── data_assets/    # experiment-scoped DataAssets
                └── runs/
                    └── run-<id>/
                        ├── run.json
                        ├── assets.json     # run-scoped asset manifest
                        ├── artifacts/
                        ├── logs/
                        ├── .ckpt/
                        └── execution/<exec_id>/   # per-attempt workflow.json + error.txt
```

## Runnable Example

`examples/workspace/workspace_architecture.py` seeds a tracked run and then prints the full on-disk tree with file sizes so you can see each layer for yourself.
