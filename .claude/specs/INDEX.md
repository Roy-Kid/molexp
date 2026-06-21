# Specs Index

- [okf-05-00-rewire-plan](okf-05-00-rewire-plan.md) — CHAIN PLAN: rewire workflow/agent/harness/server/cli off workspace onto knowledge. ✅ 05-01 charter, 05-02 run-lifecycle, 05-03 verbs, 05-04 fs-seam, 05-05 fs-append (parity done) DONE. Remaining: 05-06 agent rehome (big) → 05-07 workflow rewire → 05-08 server+cli rewire. [in-progress] — okf-05 (next sub-spec drafted at its impl)
- [okf-06-migration](okf-06-migration.md) — Non-destructive workspace-JSON → OKF bundle converter (`migrate_workspace_to_okf` + `verify_migration` + `MigrationReport`): identity→meta.yaml, runtime→_ops/run.json, dry-run/idempotent/resumable; assets deferred+warned. Depends on okf-05. [draft] — okf-06 (CHECKPOINT — touches real data)

## wsokf chain — workspace OKF-ification (corrected direction: knowledge = registry only; workspace.Folder = OKF-native)

> wsokf-01/02/03 ✅ DONE (committed: index.md graph, meta.yaml + concept-type registry, _ops/ sidecar). Remaining chain below, all [approved].

- [wsokf-04-library](wsokf-04-library.md) — OKF bundle façade `Bundle` (walk/get/put/link/build_index/search → index.json + INDEX.md) over the workspace.Folder concept tree; distinct from the per-scope `Library`. [approved]
- [wsokf-05-note-reference](wsokf-05-note-reference.md) — `Note` / `Reference` as OKF concepts on workspace.Folder + `ReferenceMeta` + read-only Zotero importer; coexist with legacy NoteAsset/record-Reference. [approved]
- [wsokf-06-agent-okf](wsokf-06-agent-okf.md) — agent `Agent`/`AgentSession` adopt OKF workspace.Folder (meta.yaml authority + @concept_type registry); only knowledge edge = `knowledge.types`. [approved]
- [wsokf-07-workflow-server-cli](wsokf-07-workflow-server-cli.md) — workflow resume / server routes / CLI read run status from `_ops/run.json` via `RunOpsState`; identity via `concept_from_dir`. [approved]
- wsokf-08-meta-migration — **DROPPED** (user: 不需要向后兼容的migrate迁移代码). Greenfield rewrite has no legacy on-disk data to backfill; the in-place migrator is dead scope.
- [wsokf-09-remove-knowledge-dup](wsokf-09-remove-knowledge-dup.md) — delete the 11 duplicate storage modules from `molexp.knowledge`, leaving `types.py` (registry) + slim `__init__`; gates on wsokf-06. [approved]


- [execution-semantics](execution-semantics.md) — Workspace↔workflow execution: `ctx.workdir` (first-class, not `inputs["workdir"]`, content-addressed incl. params), persisted binding via `Experiment.run(workflow, params=)` (seam → `workflow.json` + `source/` copy + entrypoint), workflow-layer batch `Runner`; execution model A (re-import). Surface = option C: `ws.project(p).experiment(e).run(wf, params=)`. [draft] **Supersedes the workdir-in-`inputs` parts of 01/03.**
- [pure-task-context-01-cache-contract](pure-task-context-01-cache-contract.md) — Solidify + test + document the cache-identity contract (code+config+inputs hash) [code-complete]
- [pure-task-context-03-build-flow-rewrite](pure-task-context-03-build-flow-rewrite.md) — Rewrite polymer_electrolyte/build_flow.py to the pure {inputs, config} contract [approved; workdir-via-inputs part superseded by execution-semantics]
- [editor-internal-plugin](editor-internal-plugin.md) — Extract Monaco TextEditor from core into an internal `editor` UiPluginModule (peer of molvis); registry stays the sole extension point (higher-priority editor overrides via non-colliding id); lazy-load monaco to code-split it out of the entry chunk. UI-only; not a third-party dynamic bundle. [approved]
- [runinspector-metrics-molplot](runinspector-metrics-molplot.md) — Wire RunInspector's "Metrics view not wired yet" placeholder to a real molplot line-chart view by extracting a coords-driven shared `RunMetricsView` (getRunMetrics polling + MolplotLineChart + controls) reused by RunInspector (via WorkspaceRunRow coords, no snapshot.runs) and RunMetricsTab. UI-only; gap ① only. [approved]
- [multi-run-metrics-aggregation](multi-run-metrics-aggregation.md) — Mode-A multi-run result aggregation in the Experiment view: multi-select runs (toggle + shift/ctrl) → lazy Aggregate tab → Modal picks a metric key + op (overlay / mean / errorbar). Client-side only — parallel getRunMetrics, reuse RunMetricsView's exported builders + a new aggregateSeries.ts, render via MolplotLineChart. Strict same-step (no interpolation), partial-failure tolerant. No backend change; Mode B (param-vs-metric) out of scope. UI-only. [code-complete]

_ui-creation-entries — done & closed 2026-06-10: workspace create-on-open (404 → confirm → create_if_missing; open route materializes new dirs) + WorkflowsPage "New workflow" (seed empty IR → graph editor) + optional workflow field in CreateExperimentDialog; 7/7 criteria verified, committed in the 2026-06-10 batch commit._

_pure-task-context state-elimination — staged removal shipped 2026-06-09: `ctx.state` now emits a DeprecationWarning and returns a read-only snapshot (`MappingProxyType` copy / frozen `ReadOnlyStateView`); all in-repo consumers migrated to the values-on-edges `ctx.inputs` channel; hard removal of `ctx.state` is the remaining step._
