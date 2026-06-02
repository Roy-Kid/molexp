---
title: Workflow refactor 02 — merge build+compile into WorkflowCompiler; emit CompiledWorkflow
status: draft
created: 2026-06-02
notes: |
  Second of the 4-spec workflow/ orthogonality refactor. Depends on 01 (the
  "WorkflowCompiler" name must already be free). Collapses build→spec→compile
  into one WorkflowCompiler.compile() that emits a single rich frozen artifact,
  CompiledWorkflow, holding everything the compiler derives (graph + snapshots +
  version + binding + codec). The standalone Workflow spec dissolves; experiment
  binding moves off the process-global onto an explicit registry. Execution-path
  internals are NOT touched here (still the hand-rolled scheduler) — that is 03.
---

# Workflow refactor 02 — merge build+compile into WorkflowCompiler; emit CompiledWorkflow

## Summary

Today authoring a workflow is two steps with a discarded intermediate:
`WorkflowBuilder.add(...).build()` produces a frozen `Workflow` spec, and
`build()` *secretly* runs `_pydantic_graph.WorkflowGraphCompiler` for
side-effect validation then throws the result away (`builder.py:329`); the
runtime recompiles later. Worse, `Workflow` (spec.py, 464 lines) is a god-object
fusing six concerns: compiled-spec data, the execution facade
(`execute`/`start`/`run_on`), the IR codec, experiment binding via a
process-global `_bindings_registry` class dict, versioning, and graph algebra.

This spec merges build and compile into **one `WorkflowCompiler`** (claiming the
name freed in 01) whose `.compile(*, experiment=None, registry=None)` does it all
in a single pass and emits **`CompiledWorkflow`** — the one frozen artifact that
carries everything the compiler computed:

- the executable `graph`,
- per-task `snapshots: {name → TaskSnapshot}` (computed here),
- the `WorkflowVersion` (computed here, reusing `TaskSnapshot.code_hash` so the
  two divergent code-hashers collapse to one),
- the experiment `binding` (computed here, via an explicit
  `WorkflowBindingRegistry`, not the global),
- the representation codec from 01, folded on as `to_ir`/`from_ir`/`to_python`/
  `to_mermaid` methods.

The standalone `Workflow` spec class **dissolves** — its topology data is
subsumed by `CompiledWorkflow`, and `Workflow.to_dict`/`from_dict` (now thin
delegators after 01) are deleted. The execution facade
(`execute`/`start`/`run_on`) moves off the artifact onto the runtime (still
`GraphWorkflowRuntime`; renamed in 04), which now takes a `CompiledWorkflow`.

## Design

### `WorkflowCompiler` (build + compile, one pass)
- Fluent front-end unchanged in spirit: `.add(task, depends_on=…, name=…)`,
  `.task`, `.entry`, `.parallel`, `.branch`, `.reduce`.
- `.compile(*, experiment=None, registry=None) -> CompiledWorkflow`: lowers the
  registrations once (the `WorkflowGraphCompiler` logic is absorbed as an internal
  helper, invoked exactly once — no discarded intermediate), computes snapshots +
  version, performs binding, and constructs the frozen `CompiledWorkflow`.

### `CompiledWorkflow` (the single rich frozen artifact)
- Fields: `name`, `workflow_id`, `graph`, `snapshots: Mapping[str, TaskSnapshot]`,
  `version: WorkflowVersion`, `binding: WorkflowBinding | None`.
- Methods: codec — `to_ir()`/`to_python()`/`to_mermaid()` + classmethod
  `from_ir(data, *, registry=…)` (delegating to `default_codec` from 01);
  introspection — `boundary_names`/`non_boundary_names`/`reducer_dimension`/
  `subgraph`/`run_reducer`.
- **No** `execute`/`start`/`run_on` (those live on the runtime), **no** mutable
  binding global.

### `WorkflowBindingRegistry` (kills the global)
- Explicit `{experiment_id → CompiledWorkflow}` store, owned by the workspace /
  session and passed into `compile(registry=…)` (or defaulted per-process but as
  an injectable object, never a class attribute). Replaces
  `Workflow._bindings_registry` + `bind_to`/`for_experiment`/`_reset_registry`.
- Consumers (`server/routes/execution.py`, `cli/*`, `entry.py:find_workflow_for_run`)
  switch from `Workflow.for_experiment(exp)` to `registry.for_experiment(exp)`.

### Execution facade relocation
- `GraphWorkflowRuntime.execute(compiled, …)` / `.start(compiled, …)` /
  `.run_on(compiled, experiment, …)`. The old ergonomic
  `Workflow.run_on(experiment)` becomes `runtime.run_on(compiled, experiment)`.

## Files to create or modify

- `src/molexp/workflow/builder.py` + `src/molexp/workflow/_pydantic_graph/compiler.py` → consolidate into `src/molexp/workflow/compiler.py` (`WorkflowCompiler`), CFG-lowering kept as an internal helper invoked once.
- `src/molexp/workflow/spec.py` → becomes (or is replaced by `src/molexp/workflow/compiled.py`) the `CompiledWorkflow` home; delete the old `Workflow` execution/binding/codec/version methods and the `_bindings_registry` global.
- `src/molexp/workflow/binding.py` (new) — `WorkflowBindingRegistry` + `WorkflowBinding`.
- `src/molexp/workflow/codec.py` (from 01) — `spec_to_ir`/`ir_to_spec` retargeted to `CompiledWorkflow`; folded as delegating methods on `CompiledWorkflow`.
- `src/molexp/workflow/version.py` — `WorkflowVersion` built at compile; reuse `TaskSnapshot.code_hash` (collapse the duplicate hasher).
- `src/molexp/workflow/_pydantic_graph/runtime.py` — `execute`/`start`/`run_on(compiled, …)`.
- `src/molexp/workflow/__init__.py` — export `WorkflowCompiler`, `CompiledWorkflow`, `WorkflowBindingRegistry`; drop `WorkflowBuilder` and `Workflow`.
- callers — `server/routes/execution.py`, `cli/*`, `entry.py`: `WorkflowBuilder(...).build()` → `WorkflowCompiler(...).compile(...)`; `Workflow.for_experiment` → `registry.for_experiment`.

## Tasks

- [ ] Failing tests: `WorkflowCompiler(...).compile()` returns a `CompiledWorkflow` whose `snapshots` (one per task), `version`, and `graph` are populated; `WorkflowBuilder` and `Workflow` are no longer importable from `molexp.workflow`.
- [ ] Failing test: binding via `WorkflowBindingRegistry` — `compile(experiment=exp, registry=r)` makes `r.for_experiment(exp) is compiled`; no `Workflow._bindings_registry` / `_reset_registry` symbol exists (grep gate).
- [ ] Failing test: `CompiledWorkflow.to_ir()`/`from_ir()` round-trip equals the 01 codec output; execution outputs for existing fixtures are unchanged when run via `runtime.execute(compiled)`.
- [ ] Implement `WorkflowCompiler.compile` (absorb the CFG lowering; compute snapshots + version + binding); build `CompiledWorkflow`; fold codec methods on.
- [ ] Implement `WorkflowBindingRegistry` + `WorkflowBinding`; delete the global; repoint server/cli/entry callers.
- [ ] Move `execute`/`start`/`run_on` onto the runtime (taking `CompiledWorkflow`); delete `Workflow`.
- [ ] Update `__init__.py` exports; repoint all callers (`grep -rn "WorkflowBuilder\|\.build()\|for_experiment\|import Workflow\b"`).
- [ ] Run the full gate: `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/ && pytest tests/`.

## Testing strategy

Execution behavior is unchanged: `runtime.execute(compiled)` produces the same
`WorkflowResult.outputs` as the pre-refactor `Workflow.execute()` for every
existing fixture (chain / parallel / branch / loop / reduce). Binding gets a new
explicit-registry test suite replacing the global-isolation `_reset_registry`
tests. Snapshot-at-compile and version-at-compile are asserted directly on the
returned `CompiledWorkflow`. IR round-trip via `CompiledWorkflow` matches 01.

## Out of scope

- Replacing the hand-rolled `WorkflowStep` scheduler with pydantic-graph node lowering — **spec 03** (`CompiledWorkflow.graph` still wraps the old executor here).
- `WorkflowRuntime` rename + flat caching + asset-manifest re-registration — **spec 04**.
- Extending the IR to represent control/branch/loop specs.
