# Specs Index

- [execution-semantics](execution-semantics.md) — Workspace↔workflow execution: `ctx.workdir` (first-class, not `inputs["workdir"]`, content-addressed incl. params), persisted binding via `Experiment.run(workflow, params=)` (seam → `workflow.json` + `source/` copy + entrypoint), workflow-layer batch `Runner`; execution model A (re-import). Surface = option C: `ws.project(p).experiment(e).run(wf, params=)`. [draft] **Supersedes the workdir-in-`inputs` parts of 01/03.**
- [pure-task-context-01-cache-contract](pure-task-context-01-cache-contract.md) — Solidify + test + document the cache-identity contract (code+config+inputs hash) [code-complete]
- [pure-task-context-03-build-flow-rewrite](pure-task-context-03-build-flow-rewrite.md) — Rewrite polymer_electrolyte/build_flow.py to the pure {inputs, config} contract [approved; workdir-via-inputs part superseded by execution-semantics]

_ui-creation-entries — done & closed 2026-06-10: workspace create-on-open (404 → confirm → create_if_missing; open route materializes new dirs) + WorkflowsPage "New workflow" (seed empty IR → graph editor) + optional workflow field in CreateExperimentDialog; 7/7 criteria verified, committed in the 2026-06-10 batch commit._

_pure-task-context state-elimination — staged removal shipped 2026-06-09: `ctx.state` now emits a DeprecationWarning and returns a read-only snapshot (`MappingProxyType` copy / frozen `ReadOnlyStateView`); all in-repo consumers migrated to the values-on-edges `ctx.inputs` channel; hard removal of `ctx.state` is the remaining step._
