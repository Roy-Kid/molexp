# Workspace Model

The workspace layer is the persistent record on disk. It answers *what survives after execution*.

## The four-level hierarchy

```
Workspace          ← root directory (e.g. ./lab)
└── Project        ← groups related work (e.g. qm9)
    └── Experiment ← repeatable definition (workflow + params)
        └── Run    ← one concrete execution attempt
```

| Level | What it is | On disk |
|---|---|---|
| **Workspace** | Root of a body of work | `workspace.json` |
| **Project** | Groups related experiments | `projects/<slug>/project.json` |
| **Experiment** | One workflow + parameter space | `projects/<slug>/experiments/<slug>/experiment.json` |
| **Run** | One execution with status and outputs | `projects/<slug>/experiments/<slug>/runs/run-<id>/run.json` |

## Definition vs. outcome

The critical distinction is between **experiment** (what you intend to repeat) and **run** (what actually happened). An experiment carries the workflow reference, parameter space, and provenance. A run carries the things that vary per attempt: status, timestamps, profile, results, errors, and execution history.

Without that split, retries and comparisons become ambiguous.

## Profiles and metadata

Profiles (`molcfg.yaml`) live at the boundary between workflow execution and workspace persistence. Tasks read profile fields as named parameters. The resolved profile — name, merged config, `config_hash` — is stored on the run record. You can look at `run.json` later and recover the exact configuration a run used.

## What's on disk

```
workspace_root/
├── workspace.json          ← entity metadata
├── projects.json           ← children index (projects; plural)
├── meta.json               ← sole concept identity (type; path = id)
├── index.md                ← knowledge graph narrative
└── projects/<project_id>/
    ├── project.json        ← entity metadata (singular)
    ├── experiments.json    ← children index (experiments; plural)
    └── experiments/<exp_id>/
        ├── experiment.json ← entity metadata (singular)
        ├── runs.json       ← children index (runs; plural)
        └── runs/run-<id>/
            ├── run.json    ← identity and provenance (singular)
            ├── ops/run.json ← hot state (status, ownership)
            ├── assets.json ← run-scoped asset manifest
            └── executions/<exec_id>/
                ├── execution.json
                ├── workflow.json  ← per-task outputs
                ├── stdout.log
                └── stderr.log
```

## Next

- For the concrete Python API, see [Workspace API](../guide/workspace-api.md).
- For reusable data and provenance, see [Assets and Reproducibility](assets-and-reproducibility.md).
- For the CLI that discovers this hierarchy, see [CLI and Profiles](../getting-started/cli-and-profiles.md).
