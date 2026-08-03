# Architecture

Architecture docs describe layer boundaries that must remain true as the code
evolves.

- [Agent Layer](agent.md) — `molexp.agent` as a thin wrapper over pydantic-ai.
  Documents the five-name public surface plus the two loops, the
  `_pydanticai/` firewall, the "don't reinvent pydantic-ai" rule, and why
  pipeline orchestration lives in the harness rather than the agent.
- [Harness Layer](harness.md) — `molexp.harness` as the experiment orchestrator:
  the 22-symbol public surface, artifact lineage / audit / approval machinery,
  the two shipped modes (`PlanOrchestrator`, `ChatMode`), and the boundaries
  that keep the agent edge to one Protocol and the workflow engine out of
  process.
- [Plan Mode](plan-mode.md) — the harness `PlanOrchestrator` pipeline (two-phase planning +
  planning then realization): executor-subprocess boundary, artifact/lineage layout, and the
  `molexp plan [--execute]` entry point.
- [Workflow Layer](workflow-layer.md) — `molexp.workflow` as the single workflow
  abstraction; the engine is fully self-owned under `_engine/` (no `pydantic_graph` dependency). Also covers
  the boundary between workspace storage primitives and the workflow engine
  that consumes them.
