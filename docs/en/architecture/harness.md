# Harness Layer

`molexp.harness` is the experiment orchestrator: the layer that turns an
intent into an audited, reproducible artifact trail. It sits **above** agent,
workflow and workspace and is imported by none of them.

```
       harness          (artifact lineage, audit, stages, executors, gates)
         │
         ├─ uses ─→ agent     (one edge only: agent.router.Router)
         ├─ uses ─→ workflow  (graph engine, content-addressed cache)
         └─ uses ─→ workspace (pure storage)
```

## What the layer owns

| Concern | Where |
|---|---|
| Stage machinery | `Stage`, `StageRunner`, `HarnessRunContext` |
| Artifacts + lineage | `ArtifactStore` / `FileArtifactStore`, `SQLiteArtifactLineageStore` |
| Audit trail | `SQLiteEventLog`, `generate_audit_report`, `replay_metadata` |
| Human gates | `ApprovalGate`, `SQLiteApprovalStore`, `ApprovalPendingError` |
| Execution | `Executor` Protocol, `LocalExecutor`, `DryRunExecutor` |
| LLM dispatch | `AgentGateway` Protocol, `RouterBackedAgentGateway` |
| Capabilities | `CapabilityRegistry`, built-in `curation` + `lifecycle` catalogs |

Run-level provenance (params, config hash, code identity) belongs to
**workspace**, not here. Harness lineage covers agent-pipeline artifacts only,
and stamps every edge with its stage plus the workspace `run_id`.

## Public surface

`molexp.harness.__all__` is deliberately small — **22 symbols**, locked by
`tests/test_harness/test_public_surface.py`. Stages, schemas and validators are
imported from their subpackages (`harness.stages.*`, `harness.schemas.*`),
never from the top level. Plan-owned schema names carry a `Plan` prefix
(`PlanTaskIR`, `PlanWorkflowIR`, `PlanValidationReport`, `PlanArtifactRef`) so
they cannot collide across layers.

## Two modes

Chat is one structured `AgentGateway.call`. Plan is a workflow of
AgentCall/ReAct nodes. They differ in tool surface and in what they are
allowed to write.

### `PlanOrchestrator` — authoritative, two-phase

The production pipeline behind `molexp plan` and `POST /plan-tasks` (both
through the one shared `services.plan_runtime.drive_plan_mode`).

**Phase 1 — plan workflow.** A `draft_board` ReAct mutates a task board through
the harness plan-tool surface (`harness.plan_tools` on `ctx.tools`).
`form_check` is a `wf.loop` edge: a malformed board *denies exit*
and steers the violations back to the planning agent, so a human never sees
one. A read-only `PlanReachabilityProbe` then annotates each task's
feasibility, the `ExperimentPlan` is persisted as the review subject, and a
**hard** review gate runs store-first — no grant means an
`ApprovalPendingError` suspend with a durable pending record, not a failure.
On grant the board is frozen content-addressed.

**Phase 2 — deterministic realization.** `RealizeBoard` maps one codegen
self-repair worker over each frozen task **in parallel** (full coverage by
construction), reduces the greens into a single `workflow_source` /
`test_source`, then compiles. No real science runs here — the compile is
`--compile-only`. A task that never greens returns *blocked*, persists a
durable `intervention_request` and raises before compile; it never silently
reverts to phase 1.

### `Chat` — exploratory, scratch-only

Same loop, different contract: no authoritative project/experiment/run
creation, no `run_land`, code confined to `agent/.scratch/`. Success is an
answer, not a succeeded Run. A policy hook denies workspace mutators so MCP
tools cannot bypass the restriction. Multi-step reviewable work is Plan's job,
and the preamble tells the agent to say so.

## Boundaries worth keeping

- **One agent edge.** Harness imports `agent.router.Router` and nothing else
  from the agent layer, reaching it through `RouterBackedAgentGateway`.
  `pydantic_ai` must never load when `import molexp.harness` runs.
- **The workflow engine is out of process.** It loads only inside executor
  subprocesses, never in the harness process.
- **Suspend/resume is durable and scope-tagged.** `ApprovalScope` separates
  `approval_gate` from `intervention_request` on the run's `harness.sqlite`;
  the shared services resume path rebuilds the scoped session and replays.
  Resume correctness rides on store-first replay, not on a stage ledger —
  there is no `Mode` ABC and no completion ledger.
- **No services/server/cli imports.** Application wiring lives one layer up in
  `molexp.services`, which both the CLI and the server call.
