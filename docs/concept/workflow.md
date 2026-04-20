# Workflow

The workflow layer is the in-memory model of a computation graph. It is concerned with dependency structure, execution order, and typed task boundaries. It is not the place where projects are grouped, run metadata is stored, or scheduler transport is decided. Those concerns belong to other layers.

## A Workflow Is a Definition

In MolExp, a workflow is something you author, compile, and execute. You can write it with the functional DSL, or with reusable task classes and `WorkflowBuilder`, but in both cases the result is the same kind of object: a compiled `WorkflowSpec`.

```python
wf = workflow(name="demo")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def summarize(ctx: TaskContext) -> float:
    return sum(ctx.inputs)


spec = wf.build()
```

That `spec` is already a complete workflow definition. It can run without a workspace:

```python
result = await spec.execute()
```

Or it can run under a tracked `Run`:

```python
result = await spec.execute(run=run)
```

The graph does not become a different graph when you attach it to persistent state. The same workflow definition is simply being executed under a richer lifecycle.

## Authoring, Compilation, and Execution

MolExp keeps authoring, compilation, and execution separate on purpose. Authoring is the stage where tasks and dependencies are declared. Compilation is the stage where that declaration becomes a validated `WorkflowSpec` with a deterministic `workflow_id`. Execution is the stage where the compiled graph is driven by a runtime, optionally under a `RunContext`.

That separation prevents import-time side effects from becoming part of the workflow model. A script can define a graph, inspect it, bind it to an experiment, and expose it to the CLI without accidentally running computation just because the module was imported.

## What Stays Outside the Workflow Layer

The workflow layer does not know how a research program is grouped into projects. It does not decide where a run directory should live. It does not decide whether a job runs in-process or through a scheduler bridge. It also does not own shared datasets or derived feature stores. Those are all deliberate exclusions.

This narrow boundary is what lets `WorkflowSpec` remain reusable. The same compiled graph can run during local iteration, under `molexp run`, or from a remote worker process launched later by `molexp execute`. The workflow layer stays stable because persistence and transport are external to it.

## Where It Connects to the Rest of MolExp

The workflow becomes operationally meaningful when it is bound to an experiment in the workspace layer. That binding associates one graph with one persistent research definition, one parameter set, and one replica policy. The workflow also receives profile data and workspace helpers through `TaskContext`, which is why authoring code can remain mostly unchanged while the surrounding execution environment becomes richer.

If you need the persistent side of this story, continue with [Workspace](workspace.md). If the missing piece is reusable data and provenance, continue with [Assets and Reproducibility](assets-and-reproducibility.md).
