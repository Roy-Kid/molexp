# Plan Mode

PlanMode turns a scientific request or experimental report into a
reviewable molexp workspace. It stops at a validated handoff; it does
not execute experiments.

The artifact flow is:

```text
Request -> report digest -> implementation plan -> workflow/task IR
        -> generated Python workflow -> validation -> review
        -> final handoff check -> RunMode handoff
```

## Public Surface

```python
from molexp.agent import AgentSession
from molexp.agent.modes import PlanMode
from molexp.agent.modes.plan import PlanWorkspaceHandle
from molexp.workspace import Workspace

workspace = Workspace("./lab")
handle = PlanWorkspaceHandle.materialize(workspace)
mode = PlanMode(workspace_handle=handle)

session = AgentSession()
result = await mode.run(
    harness=...,  # supplied by AgentRunner in normal use
    session=session,
    user_input="screen solvent conditions",
)

plan_state = result.loop_state["plan"]
assert "ready_for_run" in plan_state
```

`PlanMode` accepts optional provider, model-policy, and gate-policy
collaborators for tests and non-default deployments. Provider calls stay
behind the molexp-owned provider abstraction; PlanMode users do not use
`pydantic-ai` directly.

## Pipeline

Plan progress is represented by a `molexp.workflow.Workflow` with
stable node names:

- `IngestReport`
- `DraftReportDigest`
- `DraftImplementationPlan`
- `CompileWorkflowIR`
- `CompileTaskIR`
- `GenerateWorkflowSkeleton`
- `GenerateTaskTests`
- `GenerateTaskImplementations`
- `ValidateWorkspace`
- `HumanReview`
- `FinalHandoffCheck`

The generated workspace includes report files, workflow IR, per-task IR,
Python source, tests, a manifest, and validation reports.

## Validation

`ValidateWorkspace` does more than compile generated Python syntax. It
checks that the generated `src/` tree can be imported the same way
RunMode will import it:

```text
source_root: src
module: experiment.workflow
symbol: create_workflow
```

The entrypoint is called and must return a `molexp.workflow.Workflow`.
The loaded workflow is then checked with the generic workflow contract
validation API.

`FinalHandoffCheck` repeats the RunMode-facing check after human review.
This keeps the final handoff reliable even if review or edits changed
the workspace.

## Status

Human approval does not mean the workspace is runnable. The result state
and manifest distinguish:

- whether the plan was approved;
- whether machine validation passed;
- whether the workspace is `ready_for_run`.

Failed validation produces a reviewable workspace with validation
reports, not a runnable handoff. `RunMode` should rely on
`manifest.yaml` and the `PlanRunHandoff` entrypoint metadata.
