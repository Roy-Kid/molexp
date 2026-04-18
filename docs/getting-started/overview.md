# MolExp Overview

Before diving into specific APIs, it helps to separate the three models that `molexp` keeps distinct on purpose: workflow structure, workspace state, and optional execution plugins.

## Three Models, One System

```
Workflow authoring → WorkflowSpec → Graph runtime → Workspace → FastAPI → React UI
        ↑                                ↑
   Task / Actor API                molcfg profiles
```

That diagram is easy to read as one pipeline, but in practice it is three connected systems:

1. **Workflow** answers "what computation should happen, and in what dependency order?"
2. **Workspace** answers "where does this experiment live on disk, and how is each execution recorded?"
3. **Plugins** answer "what optional capability transports or extends execution beyond the local core?"

Once those boundaries are clear, the rest of the codebase becomes much easier to reason about.

## Workflow Concepts Describe Computation

The workflow layer is the in-memory model of your computation graph. It is centered around a few ideas:

| Concept | Meaning |
|--------|---------|
| `Task` | One-shot async computation that returns a single value. |
| `Actor` | Streaming async computation that yields multiple values. |
| `TaskContext` / `ActorContext` | The execution context passed into each task or actor. |
| `WorkflowDSL` / `WorkflowBuilder` | Authoring surfaces for registering tasks and dependencies. |
| `WorkflowSpec` | The compiled, executable DAG with a deterministic `workflow_id`. |
| Workflow runtime | The executor that runs the spec, opens run state when needed, and schedules same-level tasks in parallel. |

The important distinction is that a workflow is not yet an experiment record. A `WorkflowSpec` is just a compiled graph. It can be executed purely in memory:

```python
result = await spec.execute()
```

Or it can be executed under a tracked `Run`:

```python
result = await spec.execute(run=run)
```

The graph itself stays the same. What changes is whether a workspace-backed lifecycle exists around it.

### Authoring Compiles Into Execution

`molexp` deliberately separates authoring, compilation, and execution:

1. **Authoring** registers tasks and their dependency edges.
2. **Compilation** turns that registration into a `WorkflowSpec`.
3. **Execution** runs the spec with a runtime and an optional `RunContext`.

This separation matters because it avoids hidden side effects during definition time. A script can import, build, inspect, and bind workflows without immediately executing them.

### Workflow Boundaries Stay Narrow

The workflow layer does not decide where run metadata is stored, how projects are grouped, or which scheduler transports a worker process to a cluster. Those concerns live in the workspace and plugin layers. This is why a `WorkflowSpec` remains valid whether you are running locally, under `molexp run`, or from a cluster worker launched via `molexp execute`.

## Workspace Concepts Describe Experiment State

The workspace layer is the persistent model on disk:

```
Workspace
└── Project
    └── Experiment
        └── Run
```

Each level has a different semantic role.

### `Workspace`

The workspace is the root container. It corresponds to one directory on disk and owns the top-level `workspace.json`. It is the thing you point `molexp serve` and `molexp watch` at.

Use a workspace when you want one stable place to hold projects, shared assets, and all run history for a body of work.

### `Project`

A project is a grouping boundary. It usually corresponds to a research area, a paper thread, a model family, or a dataset campaign. Projects organize experiments; they are not themselves executable.

Projects are where you usually attach longer-lived shared assets and descriptive metadata such as tags or ownership.

### `Experiment`

An experiment is the repeatable definition. It binds together:

- a workflow source,
- a workflow object,
- a parameter set,
- a replica count and seed policy,
- and reproducibility metadata such as the captured git commit.

The key idea is that an experiment describes the run family, not one outcome. If you rerun the same experiment ten times, you still have one experiment and multiple runs.

### `Run`

A run is one realized execution instance of an experiment. This is where status, artifacts, logs, errors, profile metadata, and per-attempt execution history live.

Runs matter because they are the unit of operational truth. If an experiment is the recipe, a run is one actual cooking attempt with a result attached.

### Definition and Outcome Stay Separate

Scientific workflows rarely fail because there is no graph abstraction. They fail because definition, configuration, and outcomes are mixed together. The workspace hierarchy keeps those concerns separate:

- `Experiment` stores the repeatable setup.
- `Run` stores one outcome of that setup.
- `ExecutionRecord` stores one attempt inside that run if retries happen.

That is why `molexp` can support retries, resume behavior, artifact tracking, and later inspection without confusing "this workflow definition exists" with "this execution succeeded".

## `molcfg` Connects Workflow and Workspace

Profiles sit between the workflow layer and the workspace layer.

The workflow consumes profile data through `ctx.config`. The workspace persists the selected profile name, merged config payload, and `config_hash` into the run metadata. This means configuration is both executable and inspectable.

That design keeps the framework neutral about semantics. `molexp` knows how to resolve a profile; it does not know whether `epochs`, `dataset`, or `skip_heavy_compute` should change model behavior. User code decides that.

## Plugin Concepts Describe Optional Capabilities

The plugin layer is how `molexp` adds heavyweight or environment-specific behavior without making the core package depend on it all the time.

Two examples in this repository are:

- `agent_pydanticai` for the optional agent capability.
- `submit_molq` for optional scheduler submission through `molq`.

The core principle is that `import molexp` should still work in a lightweight environment. Optional packages are loaded only when the user asks for the relevant capability.

## The `submit_molq` Plugin in Context

The `submit_molq` plugin is the scheduler bridge behind:

- `molexp run --slurm`
- `molexp run --pbs`
- `molexp run --lsf`
- `molexp run --scheduler <name>`

Its role is narrower than it may first appear:

- It does **not** define workflow semantics.
- It does **not** replace `RunContext` or `WorkflowSpec`.
- It **does** translate CLI resource and scheduling flags into `molq` job submissions.
- It **does** launch `python -m molexp.cli execute <run_dir>` on the target scheduler.
- It **does** persist normalized executor metadata onto the run so the monitor and UI can understand where the job lives.

This is why local and cluster execution remain conceptually aligned. They use the same workflow model and the same run metadata model. Only the transport layer changes.

## A Useful Reading Order

If you are new to the project, this order usually works best:

1. Read the [Quick Start](../tutorial/quick-start.md) to see a full script end to end.
2. Read [Run Profiles and Reproducible CLI Execution](../guide/run-profiles.md) to understand `ctx.config`, profiles, and resume behavior.
3. Read [Workspace Architecture](../guide/workspace-architecture.md) once you need the persistence model.
4. Read [Task and Actor](../guide/task-and-actor.md), [TaskContext](../guide/task-context.md), and [Workflow Runtime](../guide/workflow-runtime.md) when implementing or debugging workflow behavior.
5. Read [Molq Plugin and Cluster Submission](../guide/molq.md) when you need scheduler-backed execution.
