# MolExp Examples

Each guide in `docs/guide/` (and each onboarding page in `docs/getting-started/`)
has a runnable example here. Read the guide for prose, run the example to see
the same idea working.

Most examples execute the workflow in-process and write into a temporary
directory under the system temp location (printed at the top of every run).
You can delete these freely; none of them touch `~/` or any system path.

## Getting Started

| Guide | Example | What it shows |
|---|---|---|
| [quick-start](../docs/getting-started/quick-start.md) | `getting_started/01_quick_start.py` | End-to-end: workspace + experiment + run + result |
| [first-workflow](../docs/getting-started/first-workflow.md) | `getting_started/02_first_workflow.py` | A `WorkflowSpec` with no workspace attached |
| [tracked-runs](../docs/getting-started/tracked-runs.md) | `getting_started/03_tracked_run.py` | What appears on disk when a run is tracked |
| [cli-and-profiles](../docs/getting-started/cli-and-profiles.md) | `getting_started/04_cli_and_profiles/` | `molexp run` + `molcfg.yaml` + `--profile` |

## Workflow Authoring

| Guide | Example | What it shows |
|---|---|---|
| [task-and-actor](../docs/guide/task-and-actor.md) | `workflow/task_and_actor.py` | Functional DSL, OOP builder, and `Actor` streaming |
| [task-context](../docs/guide/task-context.md) | `workflow/task_context.py` | `ctx.state` / `ctx.deps` / `ctx.inputs` / `ctx.config` |
| [workflow-runtime](../docs/guide/workflow-runtime.md) | `workflow/workflow_runtime.py` | `spec.execute()` vs `spec.start()` |
| [control-flow](../docs/guide/control-flow.md) | `workflow/control_flow.py` | Diamond fan-out, conditional branches, fan-out via `parallel_map` |
| [subworkflows](../docs/guide/subworkflows.md) | `workflow/subworkflows.py` | Calling a sub-spec from inside a task |

## Records and Assets

| Guide | Example | What it shows |
|---|---|---|
| [workspace-api](../docs/guide/workspace-api.md) | `workspace/workspace_api.py` | `Workspace → Project → Experiment → Run` walk |
| [workspace-architecture](../docs/guide/workspace-architecture.md) | `workspace/workspace_architecture.py` | What files actually land on disk |
| [workflow-persistence](../docs/guide/workflow-persistence.md) | `workspace/workflow_persistence.py` | `run.json`, `execution_history`, `config_hash` |
| [assets](../docs/guide/assets.md) | `workspace/assets.py` | Artifact, log, checkpoint, `find_asset` |

## Operations

| Guide | Example | What it shows |
|---|---|---|
| [run-profiles](../docs/guide/run-profiles.md) | `operations/run_profiles/` | `molcfg.yaml`, `--profile`, `--override` |
| [server-lifecycle](../docs/guide/server-lifecycle.md) | `operations/server_lifecycle.py` | Programmatic `ServerManager.start()` / `stop()` |
| [molq](../docs/guide/molq.md) | `operations/molq.py` | How `--scheduler slurm` composes a `SubmitHandler` |

## Running an Example

Every `.py` example runs stand-alone:

```bash
python examples/getting_started/01_quick_start.py
```

Examples under a subdirectory (`04_cli_and_profiles/`, `run_profiles/`) ship
a matching `molcfg.yaml` and run through the `molexp` CLI:

```bash
molexp run examples/getting_started/04_cli_and_profiles/train.py --profile smoke
```
