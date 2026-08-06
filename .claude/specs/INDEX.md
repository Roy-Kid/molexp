# Specs Index

One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.

_No live specs._

## close-loop (chain)

_Closed 01–06 on `feat/close-loop` (2026-08-06) — multi-pillar learning loop: plan knowledge feed, run FailureAnalysis, metrics_ingest product surface, Copilot UI, diagnose wire, entity density locks._

| # | Spec | Commit theme |
|---|---|---|
| 01 | close-loop-01-plan-knowledge | PlanOrchestrator + AssembleKnowledgeContext |
| 02 | close-loop-02-run-failure | services.run_failure + CLI/API |
| 03 | close-loop-03-metrics-land | metrics_ingest plugin + CLI/API + docs |
| 04 | close-loop-04-copilot-ui | RightPanel Copilot over GET /copilot |
| 05 | close-loop-05-diagnose-wire | Analyze button + Copilot diagnose → API |
| 06 | close-loop-06-entity-density | Relations/catalog/lifecycle tests |

### Deferred (not specs yet — open when evidence bites)

| Theme | Why deferred |
|---|---|
| High-risk ChangeProposal policy table (integration P2.1) | lifecycle/curate gates exist; full policy matrix needs production pain |
| `assets.scan` scale index | intentional full-manifest scan; revisit on real large workspaces |
| Multi-agent adversarial plan review (P2.3) | single-auditor StepAuditLoop sufficient for now |
| Workspace-global event timeline UI | `workspace.events` spine exists; consumer is product polish |
| Default-on auto-analyze / auto-ingest | cost/noise; keep opt-in only |

---

## plan-emergent (chain)

_Closed 01–08 (2026-07-11) — two-phase **emergent-planning → deterministic-realization** rewrite of harness PlanMode; cutover landed in `37a3ef0`._

## plan-step-audit (chain)

_Closed 01–05 on feat/plan-step-audit._

_Closed: agent-code-loop 01–05 (2026-07-10); product-gap remediation; agent-record-export 01–08; execution-semantics; pure-task-context-03 (blocked)._
