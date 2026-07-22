# Specs Index
One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.

## plan-emergent (chain)

Two-phase **emergent-planning → deterministic-realization** rewrite of harness PlanMode. Dependency-ordered. Landing gate: merge `feat/plan-step-audit` first (see `plan-emergent-08-cutover`). 05 split into 05a/05b/05c.

- [plan-emergent-01-agent-hooks](plan-emergent-01-agent-hooks.md) — agent-layer neutral tool/hook Protocols (HookOutcome proceed/deny/suspend + before/after-tool + should_stop) threaded into stream_agentic; no behavior change [approved]
- [plan-emergent-02-loop-refactor](plan-emergent-02-loop-refactor.md) — generalize InteractiveLoop into a Pi-style loop (steering, follow-up, durable suspend, should_stop) over the SessionEntry tree; entry-tree canonical [approved]
- [plan-emergent-03-plan-tools](plan-emergent-03-plan-tools.md) — plan tool surface: side-effecting tools via ToolCapability/dispatch_capability + a board-state tool facade + as_loop_tool adapter [approved]
- [plan-emergent-04-task-board-state](plan-emergent-04-task-board-state.md) — mutable task-board current-state file (functional replace) + frozen content-addressed ExperimentPlan (spec+board) [approved]
- [plan-emergent-05a-gateway](plan-emergent-05a-gateway.md) — generalize RouterBackedAgentGateway.call (structured vs agentic) + register create_experiment_plan + plan_report_renderer [approved]
- [plan-emergent-05b-guard](plan-emergent-05b-guard.md) — EmergentPlanFormValidator (should_stop) + read-only PlanReachabilityProbe + build_experiment_plan_review_pack [approved]
- [plan-emergent-05c-orchestrator](plan-emergent-05c-orchestrator.md) — EmergentPlanOrchestrator composing 01–05b into run(...)->ModeResult; coexists with old PlanMode [approved]
- [plan-emergent-06-realization-phase](plan-emergent-06-realization-phase.md) — deterministic map→reduce→compile realizer; per-task self-repair; block→durable intervention request [approved]
- [plan-emergent-07-suspend-resume](plan-emergent-07-suspend-resume.md) — scope-tagged durable suspend/resume (approval-gate + intervention-request) through the shared services path [approved]
- [plan-emergent-08-cutover](plan-emergent-08-cutover.md) — retire Mode/nine-step PlanMode/RepairLoop/SequentialTaskBuild; repoint the shared driver; update public surface + docs [approved]

## plan-step-audit (chain)

_Closed 01–05 on feat/plan-step-audit._


_Closed: agent-code-loop 01–05 (2026-07-10); product-gap remediation; agent-record-export 01–08; execution-semantics; pure-task-context-03 (blocked)._
