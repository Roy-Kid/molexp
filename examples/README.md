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
| [task-and-actor](../docs/guide/task-and-actor.md) | `workflow/task_and_actor.py` | Decorator, OOP, and Protocol-form tasks, plus a streaming actor |
| [task-context](../docs/guide/task-context.md) | `workflow/task_context.py` | `ctx.inputs` / `ctx.config` / `ctx.workdir` — the pure task context |
| [workflow-runtime](../docs/guide/workflow-runtime.md) | `workflow/workflow_runtime.py` | `WorkflowRuntime.execute()` vs `.start()` |
| [control-flow](../docs/guide/control-flow.md) | `workflow/control_flow.py` | Diamond fan-out, conditionals, build-time and `wf.parallel` fan-out |
| [control-flow](../docs/guide/control-flow.md) | `workflow/branch_and_loop.py` | `wf.branch` routing and `wf.loop` repeat-until — `(value, Next(label))` values arrive via `ctx.inputs` |
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

The agent examples are **offline-first**: each ships an in-file
`ScriptedRouter` implementing the SDK-free `molexp.agent.router.Router`
Protocol and injects it via `AgentRunner(router=...)` — no network, no API
key, deterministic exit 0, so both files run inside the examples smoke gate
(`tests/test_examples_smoke.py`) and break loudly on any API drift. Paste a
key into the `API_KEY` constant to flip the *same* loop to the real model.

| Example | What it shows |
|---|---|
| `agent/chat_loop.py` | Minimum viable agent loop — `ChatLoop` + a named runtime `AgentSession` driven through `AgentRunner`. The offline run proves the persistence contract: two turns on one named session, then `result.messages` carries all four messages rebuilt from `entries.jsonl`. |
| `agent/interactive_loop.py` | The emergent tool loop — `InteractiveLoop` driving `Router.stream_agentic`. The scripted stream yields a thinking delta, one full tool round, then the streamed answer; the demo asserts the chunk→`AgentEvent` translation (`ToolCallStartedEvent` / `ToolCallCompletedEvent`). The loop behind the `molexp agent` CLI REPL. |

> Note: "**Loop**" is the agent-layer LLM-conversation concept (`AgentLoop` → `ChatLoop` / `InteractiveLoop`). "**Mode**" is reserved for the harness orchestration concept below (`harness.Mode` → `PlanMode` / `RunMode`).
>
> **API keys** — live mode registers the LLM key *in code* via `molexp.config["deepseek_api_key"] = ...` (paste into the `API_KEY` constant at the top of each file). `molexp.config` is a live `molcfg.Config`; molexp reads the key from it, **never from environment variables**.

## Harness Layer

`PlanMode` is the harness `Mode` that turns a short natural-language experiment
draft into generated, validated, runnable `molexp.workflow` source — running its
stage pipeline (ExperimentReport → WorkflowIR → BoundWorkflow → workflow source)
on a `workspace.Run` with full provenance + audit. Its back half, `RunMode`
(chained on the same Run via `molexp plan --execute`), generates unit tests,
really runs them with pytest, executes the workflow through an executor
subprocess on the real engine, and writes the final report + audit trail.

| Example | What it shows |
|---|---|
| `harness/experiment_pipeline.py` | **The flagship**: a natural-language experiment goal → `PlanMode` (plan + validated workflow source) → `RunMode` (generated unit tests REALLY run under pytest, the workflow REALLY executes on the `molexp.workflow` engine in an executor subprocess) → extracted `FinalReport` + audit trail. Offline by default via an in-file `CannedGateway` implementing the public `AgentGateway` Protocol — only the LLM is canned; every validator, pytest, and the engine run for real (a seeded 1D random walk whose D = MSD/(2·d·t) ≈ 0.5). Paste a key into `API_KEY` to run the same pipeline against the real DeepSeek API through `RouterBackedAgentGateway`. |

## Driving a Run

The sanctioned surface is the fluent chain: declare which workflow an
experiment runs (this seeds one content-addressed `Run` per parameter cell
and binds the compiled workflow), then either let `molexp run` drive the
runs or execute one in-process through `WorkflowRuntime`:

```python
from molexp.workflow import WorkflowCompiler, WorkflowRuntime

compiled = WorkflowCompiler(name="train").add(Train()).compile()

exp = ws.project("demo").experiment("train").run(compiled, params={"lr": [1e-3]})

run = exp.list_runs()[0]
with run.start(profile_config=cfg) as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    ctx.set_result("final_loss", result.outputs["train"])
```

`Experiment.run(workflow, params=...)` binds the compiled workflow to the
experiment in `molexp.workflow.default_binding_registry` (an explicit,
injectable `{experiment_id → CompiledWorkflow}` store — the old class-level
`bind_to` registry was replaced) and registers the workspace for CLI
discovery, so a separate `me.entry(ws)` call is no longer needed. The
registry is process-local — cluster workers re-establish it by re-running
the user script on import.

Task bodies stay on the pure `{inputs, config}` contract: a root task of a
tracked run receives `{"params": <run params>, "workdir": <Path>}` as
`ctx.inputs`, and `ctx.config` is the resolved profile. Workspace helpers
(`ctx.set_result` / `ctx.artifact` / `ctx.log`) live on the driver-side
`RunContext`; read persisted results back with the public
`run.get_result(key)` instead of parsing `run.json` by hand.

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
