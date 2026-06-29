# Specs Index

One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.

- [execution-semantics](execution-semantics.md) — Workspace↔workflow execution: `ctx.workdir` (first-class, not `inputs["workdir"]`, content-addressed incl. params), persisted binding via `Experiment.run(workflow, params=)` (seam → `workflow.json` + `source/` copy + entrypoint), workflow-layer batch `Runner`; surface `ws.project(p).experiment(e).run(wf, params=)`. [draft] — supersedes the workdir-in-`inputs` parts of 01/03.
- [pure-task-context-01-cache-contract](pure-task-context-01-cache-contract.md) — Solidify + test + document the cache-identity contract (code+config+inputs hash). [code-complete]
- [pure-task-context-03-build-flow-rewrite](pure-task-context-03-build-flow-rewrite.md) — Rewrite polymer_electrolyte/build_flow.py to the pure {inputs, config} contract. [approved] — workdir-via-inputs part superseded by execution-semantics.
- [workspace-git-projection-02-objects](workspace-git-projection-02-objects.md) — Content-agnostic git-object framing primitive in molexp/git/objects.py (real blob/tree/commit OIDs via run_git, Layer-0). [approved]
- [workspace-git-projection-03-map](workspace-git-projection-03-map.md) — workspace/git_projection.py maps Folder/Asset/RunMetadata/ExecutionRecord → git objects → refs/molexp/*, with deterministic rebuild. [approved]
- [workspace-git-projection-04-wire](workspace-git-projection-04-wire.md) — Low-frequency checkpoint cadence at Execution-settled + shared CLI/server backend + ApprovalGate-gated push. [approved]
