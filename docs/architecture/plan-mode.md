# Plan Mode Architecture

Plan mode is a workflow authoring and validation system. It does not execute
experiments.

The planning workflow is represented with `molexp.workflow`. It turns a natural
language request into a structured `PlanSpec`, supports node-level patches and
agent rewrites, compiles the executable portion into workflow artifacts, runs a
dry run, and produces a handoff bundle for the runner.

## Flow

```mermaid
flowchart TD
    A["User Request<br/>natural language goal"] --> B["Planning Workflow<br/>molexp.workflow"]

    B --> C["Intake Node<br/>raw request, constraints, outputs"]
    C --> D["Goal Draft Node<br/>GoalSpec draft"]
    D --> E["Context Collection Node<br/>literature, repo, tools, data, environment"]
    E --> F["Method Selection Node<br/>methods, variables, controls, assumptions"]
    F --> G["Decomposition Node<br/>scientific steps, compute steps, analysis steps"]
    G --> H["Protocol Draft Node<br/>protocol, parameters, artifacts, checks"]

    H --> I["PlanSpec Preview<br/>summary, DAG, parameters, assumptions, risks"]
    I --> J{"Gate A<br/>approve plan?"}

    J -- "patch node" --> K["PlanPatch<br/>node-level structured edit"]
    K --> B

    J -- "agent rewrite node" --> L["Agent Node Rewrite<br/>rewrite selected node only"]
    L --> B

    J -- "approve" --> M["Executable Workflow Draft<br/>molexp.workflow"]

    M --> N["WorkflowTemplate<br/>reusable executable task graph"]
    N --> O["ExperimentSpec<br/>WorkflowTemplate + ParamSpace + ExecutionPolicy"]

    O --> P["Compile<br/>schema, registry, artifacts, dependencies"]
    P --> Q{"Compile OK?"}

    Q -- "no" --> R["Repair Patch<br/>patch affected nodes/subgraph"]
    R --> B

    Q -- "yes" --> S["Dry Run<br/>paths, tools, commands, resources, backend readiness"]
    S --> T{"Dry Run OK?"}

    T -- "no" --> R
    T -- "yes" --> U["WorkflowBundle / RunGraph<br/>materialized executable runs"]

    U --> V{"Gate B<br/>approve execution handoff?"}
    V -- "patch executable node" --> R
    V -- "approve" --> W["Runner<br/>dispatch, monitor, resume, collect"]

    W --> X["Backend<br/>local / slurm / pbs / lsf / remote"]
```

## Planning Nodes

The planning workflow uses these documented node names:

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

Code and documentation must use the same node names.

## Artifacts

Plan mode produces and revises these artifacts:

- `PlanSpec`
- `PlanPreview`
- `PlanPatch`
- `ExecutableWorkflowDraft`
- `WorkflowTemplate`
- `ExperimentSpec`
- `CompileReport`
- `DryRunReport`
- `WorkflowBundle` / `RunGraph`
- handoff bundle

`PlanSpec` is user-facing and editable. It is not the executable workflow.

## Revision

Plan rejection creates a structured `PlanPatch` by default. The patch targets a
specific node or field and can add, remove, replace, update, rewrite, disable,
or enable a node.

Agent rewrite is node-scoped. The agent rewrites the selected node, validates
the node output against the node schema, marks affected downstream nodes stale,
and regenerates only the affected preview or subgraph when possible.

Full replanning is an explicit fallback, not the default revision behavior.

## Handoff

Plan mode ends at an approved handoff bundle. It must not call
`Workflow.execute()` directly.

The runner receives the handoff bundle and owns dispatch, monitoring, resume,
logging, failure tracking, backend execution, and artifact collection.
