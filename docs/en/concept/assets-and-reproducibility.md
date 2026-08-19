# Assets and Reproducibility

Reproducibility in MolExp is not only about rerunning code. It's about recovering the workflow definition, the configuration, the execution record, and the data the workflow depended on. The asset layer gives those resources first-class names and scopes.

## Assets as named resources

Every persistent byproduct — imported data, task artifacts, logs, checkpoints — is a typed `Asset` scoped to the workspace, a project, an experiment, or a single run:

| Scope | Example |
|---|---|
| Workspace | Shared training datasets |
| Project | Force-field parameter files |
| Experiment | Derived feature caches |
| Run | Task artifacts, logs, checkpoints |

Assets are recorded in each scope's `assets.json` manifest. Queries scan these manifests — no filesystem walking, no undocumented paths:

```python
# docs: skip — conceptual illustration; requires a workspace and run context
# Find all error traces in an experiment
traces = exp.assets.query(kind="error_trace", recursive=True)

# A task finds a named asset without hard-coding paths
data = ctx.find_asset("training_data")
```

## Reproducibility records

Beyond outputs, MolExp persists the metadata that makes a run interpretable later:

| Record | Where | What it captures |
|---|---|---|
| Workflow snapshot | `run.json` | The exact graph identity (`workflow_id`) |
| Config hash | `run.json` | The resolved profile's `config_hash` |
| Execution history | `ops/run.json` | Status transitions, timestamps, attempt count |
| Per-task outputs | `executions/<exec_id>/workflow.json` | Each task's return value |

Together, these turn a run directory into a scientific record — not just a pile of output files.

## FAIR boundaries

MolExp supports FAIR-oriented practice inside a managed workspace:

- **Findable** — stable names, scoped manifests, queryable
- **Accessible** — Python API, server, and UI all read the same records
- **Interoperable** — teams can adopt consistent metadata conventions
- **Reusable** — workflows recover assets by name and scope, not ephemeral paths

What MolExp does **not** do automatically: publish external identifiers, enforce a community schema, or turn local records into a repository-grade FAIR archive. It gives you a strong internal record. That is a substantial improvement over ad hoc folders.

## Why this matters

Most workflow systems fail gradually. The script still exists, but nobody remembers which dataset directory was the real one, which profile was used for the paper figure, or whether the checkpoint was generated before or after the last code change. MolExp keeps the workflow, the run metadata, and the reusable assets in one persistent structure — so those questions have answers.
