---
title: Workflow refactor 03 — lower to per-task pydantic-graph nodes; delete the hand-rolled scheduler
status: approved
created: 2026-06-02
notes: |
  Third of the 4-spec workflow/ refactor. Internal executor swap only — the
  public surface (WorkflowCompiler.compile → CompiledWorkflow, runtime.execute)
  is unchanged. Replaces molexp's reinvented level-by-level scheduler with genuine
  pydantic-graph node lowering (Fork/Join/Decision/reducers), honoring the
  "patch pydantic-graph, don't reinvent" directive. Observable semantics
  (outputs, concurrency, stall/cycle errors) are identical.
---

# Workflow refactor 03 — lower to per-task pydantic-graph nodes; delete the hand-rolled scheduler

## Summary

`_pydantic_graph/node.py` collapses the entire DAG into a **single self-looping
`WorkflowStep` `BaseNode`** that computes ready-sets level-by-level
(`node.py:167`), `asyncio.gather`s the frontier via `_invoke_one`, runs its own
`_dispatch`, and raises a hand-rolled `WorkflowDeadlockError` on a stall — i.e. it
reimplements scheduling that the `pydantic_graph` library already provides
(`Fork`, `Join`, `Decision`, `StepNode`, `reduce_*`, and stall/cycle detection).
molexp's `_pydantic_graph/` is declared "the sole permitted `import pydantic_graph`
site," and its persistence layer already correctly subclasses pydantic-graph
(`RunStorePersistence(BaseStatePersistence)` using `NodeSnapshot`/`EndSnapshot`) —
that is the model to extend to the scheduler.

This spec makes `WorkflowCompiler.compile()` (from 02) lower **one pydantic-graph
node per task**: data-dependencies become graph edges, `wf.entry` becomes the
graph entries, and the control-flow declarations map onto pydantic-graph
primitives — `wf.parallel` → `Fork`/`Join`, `wf.branch` → `Decision`,
`wf.reduce` → the `reduce_*` reducers. `CompiledWorkflow.graph` becomes a genuine
`pydantic_graph.Graph`; the runtime drives it through `Graph.run` / `Graph.iter`.
The hand-rolled `WorkflowStep`, `_dispatch`, level-index loop, and manual deadlock
detection are **deleted**. The public surface and observable semantics are
unchanged — same `WorkflowResult.outputs`, same concurrency bounds, same
stall/cycle errors (now surfaced by pydantic-graph).

## Design

### Per-task node
- A single parameterized `_TaskNode(BaseNode[WorkflowState, WorkflowDeps])` (one
  instance per task name — avoid per-task code-gen). Its `run(ctx)` collects
  upstream outputs from graph state, builds the `TaskContext`/`ActorContext`
  (preserving the exact `ctx.inputs` shape: `None` / single value / dict-by-name
  from `_collect_upstream_outputs`), dispatches the body via the existing
  `_invoke_body_with_ctx`, records the result into state, and returns the next
  node(s) per the graph edges.
- The remote-execution gate (`node.py:224-233`) and `dependent_params` overlay
  (`node.py:202-210`) are carried over unchanged into the node body.

### Control-flow → pydantic-graph primitives
- `depends_on` edges → `GraphBuilder` node edges; `wf.entry` → graph entries.
- `wf.parallel` (map-over fan-out + join) → `Fork` + `Join`; drop molexp's
  `parallel_decls` / `Semaphore` hand-rolling (use pydantic-graph's fork
  concurrency).
- `wf.branch` → `Decision`.
- `wf.reduce(over=…)` → the matching `reduce_*` reducer (`reduce_list_append`,
  `reduce_sum`, …) via `ReducerContext`.

### Runtime
- `CompiledWorkflow.graph` is the `pydantic_graph.Graph`; the runtime's
  `execute`/`start` delegate to `Graph.run`/`Graph.iter`, dropping the
  `WorkflowStep(level_index=0)` seed and the level loop.
- Persistence (`RunStorePersistence`) is unchanged.

## Files to create or modify

- `src/molexp/workflow/_pydantic_graph/node.py` — rewrite to `_TaskNode` + edge dispatch; delete `WorkflowStep`, `_dispatch`, level loop, manual deadlock raise.
- `src/molexp/workflow/_pydantic_graph/compiler.py` (now inside `WorkflowCompiler` per 02) — emit the per-task pydantic-graph `Graph` via `GraphBuilder`, mapping parallel/branch/reduce to `Fork`/`Join`/`Decision`/reducers.
- `src/molexp/workflow/_pydantic_graph/runtime.py` — drive `Graph.run`/`iter`; remove the `WorkflowStep` seed.
- `src/molexp/workflow/_pydantic_graph/state.py` — keep `WorkflowState` result recording (copy-on-write); adjust to per-node writes if needed.
- `src/molexp/workflow/types.py` — `WorkflowDeadlockError` retained only if pydantic-graph's stall error is wrapped into it; otherwise mapped.

## Tasks

- [ ] Failing golden test: `WorkflowResult.outputs` for every fixture (chain, parallel, branch, loop, reduce) equals the pre-03 golden — run through the new pydantic-graph executor.
- [ ] Failing test: `CompiledWorkflow.graph` is a `pydantic_graph.Graph`; a grep gate confirms `WorkflowStep` / `_dispatch` / `level_index` no longer exist in `_pydantic_graph/`.
- [ ] Failing tests: `wf.parallel` lowers to `Fork`/`Join` (concurrency + ordered results preserved); `wf.branch` to `Decision` (route selection); `wf.reduce` to a `reduce_*` reducer (aggregation); a cyclic / stalled graph raises (via pydantic-graph, mapped to the existing error type).
- [ ] Implement the per-task `_TaskNode` + `GraphBuilder` lowering; carry over remote gate + dependent_params + `ctx.inputs` shape.
- [ ] Delete the hand-rolled scheduler; wire the runtime to `Graph.run`/`iter`.
- [ ] Run the full gate: `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/ && pytest tests/`.

## Testing strategy

Pure behavior-preservation at the public surface: golden equality of
`WorkflowResult.outputs` across all fixtures; concurrency semantics
(`max_concurrency`) preserved; ordered parallel results preserved; stall and
cycle still raise (now from pydantic-graph). Net LOC should drop (the scheduler
deletion outweighs the node lowering).

## Out of scope

- Caching and the `WorkflowRuntime` rename — **spec 04** (the cache hook lands in `_TaskNode.run` introduced here).
- Any change to `CompiledWorkflow`'s public fields/methods from 02.
- IR control-flow serialization extensions.
