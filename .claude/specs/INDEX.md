# Specs Index
One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.
- [execution-semantics](execution-semantics.md) — Workspace↔workflow execution: `ctx.workdir` (first-class, not `inputs["workdir"]`, content-addressed incl. params), persisted binding via `Experiment.run(workflow, params=)` (seam → `workflow.json` + `source/` copy + entrypoint), workflow-layer batch `Runner`; surface `ws.project(p).experiment(e).run(wf, params=)`. [draft] — supersedes the workdir-in-`inputs` parts of 01/03.
- [pure-task-context-01-cache-contract](pure-task-context-01-cache-contract.md) — Solidify + test + document the cache-identity contract (code+config+inputs hash). [code-complete]
- [pure-task-context-03-build-flow-rewrite](pure-task-context-03-build-flow-rewrite.md) — Rewrite polymer_electrolyte/build_flow.py to the pure {inputs, config} contract. [approved] — workdir-via-inputs part superseded by execution-semantics.
- [workspace-event-03-emit](workspace-event-03-emit.md) — emit run.* milestones (opt-in, non-fatal) from run_lifecycle + add_run; asset.added/knowledge.created deferred. P0.3 chain 3/3. [done]
- [knowledge-docs-02-routes](knowledge-docs-02-routes.md) — [server] mutating /knowledge/doc endpoints thin-delegating to 01 Bundle verbs, writable-gated + regen OpenAPI. [approved]
- [knowledge-docs-03-editor](knowledge-docs-03-editor.md) — [ui] Milkdown WYSIWYG note editor (index.md source of truth) + Monaco source mode, saving via 02. [approved]
- [knowledge-docs-04-tree](knowledge-docs-04-tree.md) — [ui] recursive document tree + backlinks panel + markdown export, completing the P0 editable Knowledge surface. [approved]
- [knowledge-docs-05-embed-verb](knowledge-docs-05-embed-verb.md) — [workspace] Bundle.embed typed provenance edge to Run/Asset/Experiment/Reference + entity summary + note tags/status meta. [approved]
- [knowledge-docs-06-embed-routes](knowledge-docs-06-embed-routes.md) — [server] embed endpoint + summary-card enrichment + tag/status filter delegating to 05, regen OpenAPI. [approved]
- [knowledge-docs-07-slash](knowledge-docs-07-slash.md) — [ui] slash-command menu (shadcn popover/command) + clickable entity-reference cards + tag/status controls. [approved]
