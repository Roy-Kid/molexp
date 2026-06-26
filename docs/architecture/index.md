# Architecture

Architecture docs describe layer boundaries that must remain true as the code
evolves.

- [Agent Layer](agent.md) — `molexp.agent` as a thin wrapper over pydantic-ai.
  Documents the five-name public surface plus the two loops, the
  `_pydanticai/` firewall, the "don't reinvent pydantic-ai" rule, and why
  pipeline orchestration lives in the harness rather than the agent.
- [Plan Mode](plan-mode.md) — the single harness `PlanMode` pipeline (9 steps +
  an opt-in `--execute` tail; the former separate RunMode is retired): stage
  ledger, executor-subprocess boundary, artifact/lineage layout, and the
  `molexp plan [--execute]` entry point.
- [Workflow Layer](workflow-layer.md) — `molexp.workflow` as the single workflow
  abstraction and the only place that may hide `pydantic-graph`. Also covers
  the boundary between workspace storage primitives and the workflow engine
  that consumes them.
