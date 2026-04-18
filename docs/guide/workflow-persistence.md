# Workflow Persistence

MolExp **does not serialize workflow topology** to JSON. Workflows are authored in Python and re-imported on every execution. This page documents what *is* persisted — the reproducibility data needed to recreate a run — and how to use it.

## What Is Persisted

Three pieces of data, all written atomically (temp file + `os.rename`):

### 1. `Experiment.workflow_source`

A string pointing to the Python file that defines the workflow (typically the same file where you call `me.entry(ws)`). Stored in `experiment.json`.

```json
{
  "id": "baseline",
  "name": "Baseline",
  "workflow_source": "train.py",
  "workflow_type": "taskgraph_v1",
  "git_commit": "abc123",
  "parameter_space": {"lr": 0.001}
}
```

### 2. `RunMetadata.workflow_snapshot`

A frozen `WorkflowSnapshotRef` captured at run-creation time:

```json
{
  "workflow_snapshot": {
    "source": "train.py",
    "git_commit": "abc123",
    "code_hash": null,
    "config_hash": null
  }
}
```

`source` + `git_commit` let you retrieve the exact code that produced the run.

### 3. `RunMetadata.config` / `config_hash`

The fully merged molcfg profile data the run executed against, plus a `sha256` digest for fast querying. Profiles are opaque to molexp — it stores them verbatim.

```json
{
  "profile": "smoke",
  "config": {"lr": 0.001, "epochs": 3},
  "config_hash": "f8d9..."
}
```

## What Is *Not* Persisted

- The workflow topology (DAG shape) — recomputed from `workflow_source` on replay.
- Task code — implicit in `workflow_source` + `git_commit`.
- Per-task configuration — implicit in the workflow definition.

This is deliberate: a serialized DAG can drift from the live code base. Re-importing the script guarantees the on-disk `Run` always lines up with the current Python definition. If the definition has changed, the `workflow_id` (topology hash) or `TaskSnapshot.code_hash` will too.

## Replaying a Run

```bash
# Re-execute from the CLI
molexp run train.py --profile smoke

# Or execute a worker from an existing run directory
molexp execute path/to/run-<id>/
```

`molexp execute` is the worker entry point used by cluster backends. It reads `run.json` for the `script` field, re-imports the script, matches the project + experiment IDs via `find_workflow_for_run(...)`, and drives the bound `WorkflowSpec` against the existing run directory — appending a new `ExecutionRecord` to `execution_history`.

## Identity and Correlation

| Field | Where | Meaning |
|-------|-------|---------|
| `WorkflowSpec.workflow_id` | derived | sha256 over `name + task topology`; stable across machines |
| `TaskSnapshot.code_hash` | derived | sha256 over AST-normalized `execute()` source |
| `TaskSnapshot.config_hash` | derived | sha256 over serialized task config |
| `RunMetadata.workflow_snapshot.source` | `run.json` | path to the defining script |
| `RunMetadata.workflow_snapshot.git_commit` | `run.json` | commit SHA at experiment-creation time |
| `RunMetadata.config_hash` | `run.json` | sha256 over the merged profile dict |

Use these to group, compare, and replay runs.

## Workspace-Level Files

```
./lab/
├── workspace.json
└── projects/<proj_id>/
    ├── project.json
    └── experiments/<exp_id>/
        ├── experiment.json
        └── runs/run-<id>/
            ├── run.json                      ← RunMetadata (pydantic frozen)
            ├── artifacts/
            ├── logs/
            └── execution/<exec_id>/
                ├── traceback.txt             ← on failure
                └── ...
```

All JSON files use `json.dumps(..., default=str, indent=2)` with atomic writes; structure is discovered by scanning directories, so you can move, inspect, or archive experiments independently without rewriting parent metadata.
