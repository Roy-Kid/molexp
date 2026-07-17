# Specs Index
One line per **live** spec. `/mol:spec` adds entries; `/mol:impl` ticks the spec's tasks and prunes the entry (with the spec file) on completion.

_No live specs._ agent-code-loop chain 01–05 closed (2026-07-10).

_Closed product-gap remediation (2026-07) and agent-record-export 01–08._
Closed without external work: `execution-semantics` (implemented in-tree: `ctx.workdir`, `Experiment.run`/`sweep`, `execute_run`/`RunSet.execute`, materialization store).
Closed as blocked: `pure-task-context-03-build-flow-rewrite` — target `/Users/roykid/work/molcrafts/polymer_electrolyte/build_flow.py` is absent; workdir contract is `ctx.workdir` (not `ctx.inputs`) per current engine.
