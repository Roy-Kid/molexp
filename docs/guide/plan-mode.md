# Plan Mode

Plan mode is a workflow authoring and validation system. It stops at an
approved handoff bundle; it does not execute experiments.

The artifact flow is:

```text
Intent -> PlanSpec -> WorkflowTemplate -> ExperimentSpec -> WorkflowBundle / RunGraph -> Runner
```

## Public Surface

```python
from molexp.agent.planning import (
    PlanModeRunner,
    PlanSpec,
    PlanPatch,
    create_handoff_bundle,
)
```

`PlanModeRunner` composes collaborators for approval, events, artifacts, and
provider invocation. It persists plan artifacts under `plans_root`.

```python
runner = PlanModeRunner(
    invoker=...,
    approver=...,
    event_sink=...,
    artifact_writer=...,
    plans_root=Path("./plans"),
)
result = await runner.run("screen solvent conditions")
assert result.execution_status == "handoff"
```

## Planning Workflow

Plan progress is represented by a `molexp.workflow.WorkflowGraph` with stable
node ids:

- `IntakeNode`
- `GoalDraftNode`
- `ContextCollectionNode`
- `MethodSelectionNode`
- `DecompositionNode`
- `ProtocolDraftNode`
- `PreviewNode`
- `ApprovalNode`
- `RevisionNode`
- `ExecutableWorkflowDraftNode`
- `CompileNode`
- `DryRunNode`
- `HandoffNode`

Rejection creates a `PlanPatch` against one node by default. Agent rewrite is
also node-scoped. Downstream nodes are marked stale; unrelated upstream and
sibling nodes are not regenerated.

## Handoff

`create_handoff_bundle(plan)` compiles the executable draft into a
`WorkflowTemplate`, creates an `ExperimentSpec`, performs a backend-independent
dry run, materializes a `WorkflowBundle` / `RunGraph`, and packages a
runner-facing handoff.

The runner that receives that bundle owns dispatch, monitoring, resume, failure
tracking, backend-specific execution, and artifact collection.
