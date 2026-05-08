# Agent

The agent layer turns natural-language research intent into structured MolExp
artifacts. Provider SDKs and CLI agents stay behind collaborator interfaces;
workflow representation stays in `molexp.workflow`.

## Plan Mode

Plan mode authors workflows. It does not execute experiments.

The lifecycle is:

```text
Intent -> PlanSpec -> WorkflowTemplate -> ExperimentSpec -> WorkflowBundle / RunGraph -> Runner
```

`PlanSpec` is the user-facing artifact. It captures the goal, constraints,
success criteria, evidence, method choices, decomposition, protocol,
assumptions, risks, expected artifacts, open questions, and an executable
workflow draft.

The planning topology itself is a `molexp.workflow.WorkflowGraph` using the
documented node ids:

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

## Revision

Plan rejection creates a structured `PlanPatch` by default. The patch targets
one node or one plan field and supports add, remove, replace, update, rewrite,
disable, and enable operations.

Full replanning is an explicit fallback. The normal path patches or rewrites
one selected node, validates that node, marks downstream nodes stale, and
regenerates the affected preview.

## Runner Boundary

The final plan-mode output is a handoff bundle containing:

- approved `PlanSpec`
- `WorkflowTemplate`
- `ExperimentSpec`
- `WorkflowBundle` / `RunGraph`
- `CompileReport`
- `DryRunReport`
- artifact manifest
- execution policy
- provenance

After handoff approval, execution belongs to the runner layer.

## Capability Gate

Capability levels are an authorization ladder for tool calls:

```text
OBSERVE < DRAFT < PROPOSE < GENERATE < STAGE < EXECUTE
```

They are not a plan progress state machine. Plan progress is represented by the
workflow node ids in `PlanSpec.planning_workflow`.
