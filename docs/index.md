# MolExp Documentation

Welcome to MolExp — a workflow-and-agent platform for research experiment management. MolExp provides a typed task-graph framework, a three-tier workspace hierarchy (Project → Experiment → Run), content-addressed asset storage, and a FastAPI server with React UI.

## Why MolExp?

MolExp's design philosophy is "minimal yet complete". Every layer has explicit responsibilities with no hidden magic:

- **Workflow layer**: DAG-based task graphs compiled from Python with automatic parallelization
- **Agent layer**: Goal-driven autonomous execution on top of PydanticAI
- **Workspace layer**: File-system-backed experiment management with full reproducibility
- **Server + UI**: FastAPI backend and React frontend for browsing experiments and results

MolExp is particularly suited for scientific computing scenarios that require reproducibility, traceability, and extensibility — molecular dynamics, ML training, data analysis pipelines, and more.

## Architecture

```
WorkflowSpec → Runtime → Workspace → FastAPI → React UI
                             ↑
                      AgentService (PydanticAI)
```

<div class="grid cards" markdown>

-   :material-graph: **Workflow Layer**

    ---

    **DAG-based task graphs**

    Define tasks with the functional DSL (`@wf.task`) or OOP builder. The runtime automatically parallelizes independent tasks.

    [:octicons-arrow-right-24: Quick Start](get-started/quick-start.md)

-   :material-robot: **Agent Layer**

    ---

    **Goal-driven autonomous execution**

    `AgentService` wraps PydanticAI to provide tool approval, streaming events, and workspace-aware session management.

-   :material-folder-multiple: **Workspace Layer**

    ---

    **Project → Experiment → Run hierarchy**

    Each level owns an `AssetLibrary` for scoped, content-addressed artifact storage. All metadata writes are atomic.

    [:octicons-arrow-right-24: Workspace Docs](workspace/)

-   :material-server: **Server + UI**

    ---

    **FastAPI + React**

    All routes under `/api`. The React UI provides a three-panel experiment browser. TypeScript client is auto-generated from `openapi.json`.

</div>

## Core Concepts

### Tasks and Actors

| Type | Base class | Method | Use for |
|------|-----------|--------|---------|
| Batch | `Task` | `async execute(ctx)` | single-pass computation |
| Streaming | `Actor` | `async run(ctx)` (generator) | continuous / event-driven |

### Workflow Definition

Two equivalent styles:

```python
# Functional DSL
wf = workflow(name="pipeline")

@wf.task
async def fetch(ctx: TaskContext) -> list[float]: ...

@wf.task(depends_on=["fetch"])
async def process(ctx: TaskContext) -> float: ...

spec = wf.build()
```

```python
# OOP builder
spec = (
    WorkflowBuilder(name="pipeline")
    .add(FetchTask())
    .add(ProcessTask(), depends_on=["fetch"])
    .build()
)
```

### Workspace Hierarchy

```python
workspace = Workspace.from_path("./lab")
project    = workspace.create_project(name="MD Simulations")
experiment = project.create_experiment(name="Temperature Sweep")
run        = experiment.create_run(parameters={"T": 300})

result = await spec.execute(run=run)
```

## Quick Start

New to MolExp? Start here:

[:octicons-arrow-right-24: Quick Start Guide](get-started/quick-start.md)

## License

BSD 3-Clause License — see [LICENSE](https://github.com/molcrafts/molexp/blob/main/LICENSE) for details.
