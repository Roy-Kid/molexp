# Workflow Persistence

MolExp **does not serialize workflow topology** to JSON. Workflows are authored in Python and re-imported on every execution. This page documents what *is* persisted — the reproducibility data needed to recreate a run — and how to use it.

## Persistent Metadata

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

An opaque JSON dict captured at run-creation time. The canonical shape is `molexp.workflow.snapshot_ref.WorkflowSnapshotRef` — but workspace stores the value as a plain dict to keep the dependency direction one-way (workspace ← workflow). Workflow-layer code dumps the model into JSON before handing it to workspace; workspace just round-trips it:

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

## Deliberate Omissions

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

`molexp execute` is the worker entry point used by cluster backends. It reads `run.json` for the `script` field, re-imports the script, matches the project + experiment IDs via `find_workflow_for_run(...)`, and drives the bound `Workflow` against the existing run directory — appending a new `ExecutionRecord` to `execution_history`.

## Identity and Correlation

| Field | Where | Meaning |
|-------|-------|---------|
| `Workflow.workflow_id` | derived | sha256 over `name + task topology`; stable across machines |
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
        └── runs/run-<run_id>/
            ├── run.json                  ← RunMetadata: identity/provenance only
            │                                (params, config_hash, profile, target —
            │                                 NO status / execution history)
            ├── ops/run.json             ← hot state: status, ownership,
            │                                heartbeat, execution records
            ├── assets.json               ← run-scoped asset manifest
            ├── artifacts/                ← final products
            └── executions/<exec_id>/     ← one dir per attempt
                ├── execution.json
                ├── workflow.json         ← per-node status + outputs (resume seed source)
                ├── logs/<name>.log
                └── error.txt             ← on failure: why, with traceback
```

A run's `run.json` entity file is pure identity and provenance; the hot operational state — status, ownership stamps, heartbeat, the `ExecutionRecord` list — lives in the `ops/run.json` sidecar, which is what `run.status` and `run.execution_history` read. Per-attempt files live under `executions/<exec_id>/` (the exec id is `exec-<run_id>` plus an optional `-N` suffix for reruns); when an attempt fails, `executions/<exec_id>/error.txt` records what went wrong. Run-lifecycle milestones (created / started / completed / failed) are also appended to a workspace-level timeline at `<workspace_root>/workspace.events.sqlite` — `molexp runs info` shows a run's recent events, and reading the timeline never creates the file.

All JSON files are written atomically (temp file + `os.rename`); structure is discovered by scanning directories, so you can move, inspect, or archive experiments independently without rewriting parent metadata.

## Rerun, Resume, and the Cache

A run that ended `failed` or `cancelled` can be re-executed on the **same** `run_id` in exactly two ways — there is no "clone into a new run" operation:

- **`--resume`** reopens the *existing* execution: completed task outputs are seeded from that execution's persisted `workflow.json`, and only unfinished/failed nodes (and their downstream) recompute. Same `exec_id`, continued in place.
- **`--rerun`** opens a *fresh* attempt (`exec-<run_id>-N`): a new `ExecutionRecord`, executed from the top of the graph.

`--rerun` interacts with the content-addressed cache. A rerun does not seed anything, but every task whose cache identity (code + config + upstream outputs + sweep params) is unchanged **may hit the cache** — a deterministic task that already succeeded can be served its previous output instead of recomputing. That is usually what you want; when it is not (say the task reads mutable external state the cache key cannot see), pass `--rerun --fresh` to bypass cache *reads* for that execution, forcing every node to genuinely re-execute while still writing fresh cache entries.

Neither verb touches `pending` or `succeeded` runs, and a live `running` run must be cancelled first — retrying is always an explicit verb, never implicit.

## Runnable Example

`examples/workspace/workflow_persistence.py` runs a deliberately flaky task twice and prints the `execution_history`, `profile`, `config`, and `config_hash` fields from `run.json`.
