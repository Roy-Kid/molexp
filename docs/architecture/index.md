# Architecture

Architecture docs describe layer boundaries that must remain true as the code
evolves.

- [Plan Mode](plan-mode.md) — plan mode as workflow authoring and validation,
  ending in runner handoff.
- [Workflow Layer](workflow-layer.md) — `molexp.workflow` as the single workflow
  abstraction and the only place that may hide `pydantic-graph`.
- [Experiment and Runner Boundary](experiment-runner-boundary.md) — how
  `WorkflowTemplate`, `ExperimentSpec`, `WorkflowBundle` / `RunGraph`, workspace,
  and runner responsibilities fit together.
