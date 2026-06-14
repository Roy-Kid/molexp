# Workflow

The workflow layer is the in-memory model of a computation graph. It is concerned with dependency structure, execution order, and typed task boundaries. It is not the place where projects are grouped, run metadata is stored, or scheduler transport is decided. Those concerns belong to other layers.

## A Workflow Is a Definition

In MolExp, a workflow is something you author, compile, and execute. You can write it with decorators or with reusable task classes, but both styles live on the same `WorkflowCompiler` and produce the same kind of object: a frozen `CompiledWorkflow`.

```python
from molexp.workflow import TaskContext, WorkflowCompiler

wf = WorkflowCompiler(name="demo")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def summarize(ctx: TaskContext) -> float:
    return sum(ctx.inputs)


compiled = wf.compile()
```

That `compiled` is already a complete workflow definition. Execution lives on `WorkflowRuntime`. It can run without a workspace:

```python
from molexp.workflow import WorkflowRuntime

result = await WorkflowRuntime().execute(compiled)
```

Or it can run under a tracked `Run` by forwarding the run's context:

```python
with run.start() as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
```

The graph does not become a different graph when you attach it to persistent state. The same workflow definition is simply being executed under a richer lifecycle.

## Authoring, Compilation, and Execution

MolExp keeps authoring, compilation, and execution separate on purpose. Authoring is the stage where tasks and dependencies are declared on a mutable `WorkflowCompiler`. Compilation is the stage where that declaration becomes a validated, frozen `CompiledWorkflow` with a deterministic `workflow_id`. Execution is the stage where the compiled graph is driven by a `WorkflowRuntime`, optionally under a `RunContext`.

That separation prevents import-time side effects from becoming part of the workflow model. A script can define a graph, inspect it, bind it to an experiment, and expose it to the CLI without accidentally running computation just because the module was imported.

## What Stays Outside the Workflow Layer

The workflow layer does not know how a research program is grouped into projects. It does not decide where a run directory should live. It does not decide whether a job runs in-process or through a scheduler bridge. It also does not own shared datasets or derived feature stores. Those are all deliberate exclusions.

This narrow boundary is what lets `Workflow` remain reusable. The same compiled graph can run during local iteration, under `molexp run`, or from a remote worker process launched later by `molexp execute`. The workflow layer stays stable because persistence and transport are external to it.

## Where It Connects to the Rest of MolExp

The workflow becomes operationally meaningful when it is declared on an experiment in the workspace layer via `experiment.run(compiled, params=...)`. That declaration associates one graph with one persistent research definition and one parameter sweep. At execution time the engine delivers profile data through `ctx.config` and the run's sweep parameters plus a content-addressed working directory through the root task's `ctx.inputs`, which is why authoring code can remain mostly unchanged while the surrounding execution environment becomes richer.

If you need the persistent side of this story, continue with [Workspace](workspace.md). If the missing piece is reusable data and provenance, continue with [Assets and Reproducibility](assets-and-reproducibility.md).
