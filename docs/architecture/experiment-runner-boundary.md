# Experiment and Runner Boundary

Plan mode authors workflows. The runner executes materialized workflows.

The boundary is:

```text
PlanSpec -> WorkflowTemplate -> ExperimentSpec -> WorkflowBundle / RunGraph -> Runner
```

## Experiment Boundary

An `ExperimentSpec` combines:

- `WorkflowTemplate`
- `ParamSpace`
- `ExecutionPolicy`

Parameter expansion happens at the experiment/workspace boundary, not inside
plan construction.

The experiment/workspace layer owns:

- persisted projects
- experiments
- runs
- parameter bindings
- run matrix expansion
- workspace artifact storage

## Runner Boundary

The runner receives a materialized `WorkflowBundle` or `RunGraph`.

The runner owns:

- dispatch
- monitoring
- resume
- logging
- failure tracking
- backend-specific execution
- artifact collection

The runner must not construct research plans.

## Workflow Boundary

`molexp.workflow` owns the backend-independent workflow representation and
runtime contracts. It may provide primitives used by a runner, but it should not
own scheduler-specific operations such as Slurm/PBS/LSF submission, remote
transport, job polling, or workspace run directory management.

## Handoff Artifact

Plan mode produces a handoff artifact containing:

- approved `PlanSpec`
- `WorkflowTemplate`
- `ExperimentSpec`
- `WorkflowBundle` / `RunGraph`
- `CompileReport`
- `DryRunReport`
- artifact manifest
- execution policy
- provenance

After handoff approval, execution belongs to the runner.
