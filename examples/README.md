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
| [first-workflow](../docs/getting-started/first-workflow.md) | `getting_started/02_first_workflow.py` | A `Workflow` with no workspace attached |
| [tracked-runs](../docs/getting-started/tracked-runs.md) | `getting_started/03_tracked_run.py` | What appears on disk when a run is tracked |
| [cli-and-profiles](../docs/getting-started/cli-and-profiles.md) | `getting_started/04_cli_and_profiles/` | `molexp run` + `molcfg.yaml` + `--profile` |

## Workflow Authoring

| Guide | Example | What it shows |
|---|---|---|
| [task-and-actor](../docs/guide/task-and-actor.md) | `workflow/task_and_actor.py` | Functional DSL, OOP builder, and Protocol-form tasks |
| [task-context](../docs/guide/task-context.md) | `workflow/task_context.py` | `ctx.inputs` / `ctx.deps` / `ctx.config` / `ctx.run_context` |
| [workflow-runtime](../docs/guide/workflow-runtime.md) | `workflow/workflow_runtime.py` | `spec.execute()` vs `spec.start()` |
| [control-flow](../docs/guide/control-flow.md) | `workflow/control_flow.py` | Diamond fan-out, conditional branches, build-time fan-out |
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
| [molq](../docs/guide/molq.md) | `operations/scheduler_molq.py` | How `--scheduler slurm` composes a `SubmitHandler` |

## Agent Layer

| Example | What it shows |
|---|---|
| `agent/chat_mode.py` | Minimum viable agent loop — `ChatMode` + a named runtime `AgentSession` driven through `AgentRunner`, using `pydantic_ai.models.test.TestModel` so it runs offline. Shows the harness-based contract: `runner.run` drains the mode's `AgentEvent` stream into an `AgentRunResult`, and the Jsonl-backed session resumes across runner instances. |
| `agent/plan_mode.py` | The read-only `PlanMode` planner — its seven-stage async pipeline on the `AgentHarness` turning a free-text report into an approved typed `PlanGraph`. Runs offline with a scripted `Router` + stub `CapabilityProbe` (mirroring the test-suite); the docstring shows how to wire a real provider. `runner.run_events` exposes the live `AgentEvent` stream. |

## Driving a Run

The canonical pattern is to bind a `Workflow` to the experiment via
the workflow-layer registry, enter a workspace `RunContext`, and forward
it to `spec.execute()`:

```python
from molexp.workflow import promote_callable, Workflow

spec = promote_callable(train, name="train")  # or wf.build()
spec.bind_to(experiment)

run = experiment.Run(parameters={"seed": 0})
with run.start(profile_config=cfg) as ctx:
    result = await spec.execute(run_context=ctx)
```

`Workflow.bind_to` stores the spec in a class-level process-local
registry keyed by `experiment.id` so the CLI, server, and agent tools
can later retrieve it via `Workflow.for_experiment(experiment)`. The
registry is process-local — cluster workers re-establish it by
re-running the user script on import.

`run_context=` is the only way to expose workspace helpers
(`ctx.artifact` / `ctx.log` / `ctx.set_result`) inside task bodies.
Tasks unable to find a `RunContext` fall through to in-memory execution
and the workspace-side helpers are no-ops.

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
