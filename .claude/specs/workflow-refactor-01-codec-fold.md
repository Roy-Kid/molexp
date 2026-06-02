---
title: Workflow refactor 01 — free the compiler namespace + consolidate the codec
status: draft
created: 2026-06-02
notes: |
  First spec of the workflow/ orthogonality refactor (4-spec chain). This one is
  behavior-preserving prep: it renames the representation codec off the
  "WorkflowCompiler" name and makes it the single owner of IR conversion, so the
  build+compile merge in spec 02 can claim "WorkflowCompiler" without a collision.
  No execution-path change; IR / Python / Mermaid outputs stay byte-identical.
---

# Workflow refactor 01 — free the compiler namespace + consolidate the codec

## Summary

The `workflow/` package overloads the word **"compiler"** three ways:
`serializer.WorkflowCompiler` (a representation codec: `Workflow ⇄ IR-json /
Python / Mermaid / YAML`), `_pydantic_graph.WorkflowGraphCompiler` (the CFG
lowering pass), and the refactor's planned merge of `WorkflowBuilder` into a
single `WorkflowCompiler` (build + compile in one step). The codec is the wrong
owner of that name — it converts representations, it does not compile — and it is
also a half-orphan: it is exported from the public API but has **zero internal
consumers** (`spec_to_ir`/`ir_to_spec` are thin pass-throughs to
`Workflow.to_dict`/`from_dict`, and the one real caller —
`server/routes/execution.py` — bypasses the codec and calls `Workflow.from_dict`
directly).

This spec renames the codec to **`WorkflowCodec`** (file `serializer.py` →
`codec.py`, `default_compiler` → `default_codec`) and makes it the **single
owner** of IR conversion by moving the `to_dict` / `from_dict` bodies out of
`spec.py` into the codec; `Workflow.to_dict` / `from_dict` become thin delegators
to `default_codec` (removed entirely in spec 02 when `Workflow` dissolves into
`CompiledWorkflow`). The net effect is purely structural: the `WorkflowCompiler`
name is freed for spec 02, the codec gains real consumers, and the IR / Python /
Mermaid surfaces are byte-for-byte unchanged.

This is the prerequisite step in the 4-spec chain:

1. **01 (this spec)** — free the `WorkflowCompiler` name; codec becomes `WorkflowCodec`, single IR owner.
2. **02** — merge `WorkflowBuilder` + `WorkflowGraphCompiler` → `WorkflowCompiler`; emit a `CompiledWorkflow` that holds the pydantic-graph `Graph`, per-task `TaskSnapshot`s, the `WorkflowVersion`, and the experiment binding (via an explicit `WorkflowBindingRegistry`, killing the `Workflow._bindings_registry` global); fold the codec methods onto `CompiledWorkflow`; dissolve the standalone `Workflow` spec.
3. **03** — replace the hand-rolled `WorkflowStep` level-scheduler (`_pydantic_graph/node.py`) with per-task pydantic-graph node lowering (`Fork` / `Join` / `Decision` / reducers); reuse pydantic-graph's `BaseStatePersistence` / `NodeSnapshot`.
4. **04** — rename `GraphWorkflowRuntime` → `WorkflowRuntime` with a flat `self.cache: Caching | None`; wire the per-node cache hook keyed by `CompiledWorkflow.snapshots[name].key | input_hash`; implement the asset-manifest re-registration on cache hit (content-addressed via `FileArtifactStore`).

## Design

### Rename + relocate (`serializer.py` → `codec.py`)

- `class WorkflowCompiler` → `class WorkflowCodec`. The class is already a
  stateless converter with instance methods (subclassable to swap the Mermaid
  renderer); that contract is preserved verbatim — only the name changes.
- Module-level `default_compiler` → `default_codec`.
- Rename the file `serializer.py` → `codec.py` to match the class and the
  package's "meaningful filename" convention; update the module docstring's
  `:class:`WorkflowCompiler`` references.

### Single IR owner

Today the real `to_dict` / `from_dict` logic lives on `spec.py::Workflow`
(`Workflow.to_dict` at `spec.py:285`, `from_dict` at `spec.py:300`) and
`WorkflowCodec.spec_to_ir` / `ir_to_spec` merely call into them. Invert that:

- Move the `to_dict` / `from_dict` bodies into `WorkflowCodec.spec_to_ir(spec)` /
  `WorkflowCodec.ir_to_spec(ir, *, registry=...)`.
- `Workflow.to_dict(self)` / `Workflow.from_dict(cls, data, *, registry=...)`
  become one-line delegators to `default_codec` (kept this spec only, for
  backward compatibility; deleted in spec 02). This keeps every current caller
  working byte-identically while making the codec the authoritative path.
- Repoint the one real external caller — `server/routes/execution.py:54`
  (`Workflow.from_dict(...)`) — at `default_codec.ir_to_spec(...)`.

### Public API (`workflow/__init__.py`)

- Replace the `WorkflowCompiler` / `default_compiler` imports + `__all__` entries
  with `WorkflowCodec` / `default_codec`. **`WorkflowCompiler` is intentionally
  no longer exported** — that absence is what unblocks spec 02. (No deprecation
  alias: this package's API is internal and version 02 reuses the name for a
  different class, so a lingering alias would be actively misleading.)

## Files to create or modify

- `src/molexp/workflow/serializer.py` → **rename to** `src/molexp/workflow/codec.py`; `WorkflowCompiler`→`WorkflowCodec`, `default_compiler`→`default_codec`; absorb `to_dict`/`from_dict` bodies into `spec_to_ir`/`ir_to_spec`.
- `src/molexp/workflow/spec.py` — `to_dict`/`from_dict` become thin delegators to `default_codec`.
- `src/molexp/workflow/__init__.py` — swap exports `WorkflowCompiler`/`default_compiler` → `WorkflowCodec`/`default_codec`; drop `WorkflowCompiler` from `__all__`.
- `src/molexp/server/routes/execution.py` — `Workflow.from_dict(...)` → `default_codec.ir_to_spec(...)`.
- any other references (grep `WorkflowCompiler`, `default_compiler`, `serializer`) — repoint.
- `tests/test_workflow/test_serializer*.py` → rename to `test_codec*.py`; assertions updated to the new names; add a golden round-trip test proving IR/Python/Mermaid output is unchanged from `main`.

## Tasks

- [ ] Write failing test: `from molexp.workflow import WorkflowCompiler` raises `ImportError` (name freed), while `WorkflowCodec` + `default_codec` import and `codec.spec_to_ir(codec.ir_to_spec(ir)) == ir` round-trips for the data-DAG fixtures (`tests/test_workflow/test_codec.py`).
- [ ] Write failing golden test: `default_codec.ir_to_python(ir)`, `ir_to_mermaid(ir)`, and `spec_to_ir(spec)` produce output byte-identical to a captured `main`-branch golden for the existing fixtures.
- [ ] Rename `serializer.py` → `codec.py`; rename `WorkflowCompiler`→`WorkflowCodec`, `default_compiler`→`default_codec`; move `to_dict`/`from_dict` bodies into `spec_to_ir`/`ir_to_spec`; fix docstring cross-refs.
- [ ] Make `Workflow.to_dict`/`from_dict` thin delegators to `default_codec`.
- [ ] Swap `workflow/__init__.py` exports + `__all__`; remove `WorkflowCompiler`.
- [ ] Repoint `server/routes/execution.py` and any other callers (`grep -rn "WorkflowCompiler\|default_compiler\|workflow.serializer"`).
- [ ] Rename/adjust serializer tests → codec tests; keep coverage ≥ existing.
- [ ] Run the full gate: `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/ && pytest tests/`.

## Testing strategy

Behavior-preserving refactor — the proof obligation is **byte-identical output**:
golden round-trip tests on the IR / Python / Mermaid surfaces against captured
`main` output, plus the existing serializer test suite re-pointed at
`WorkflowCodec`. No execution-path test changes (the runtime never touched the
codec). A grep-gate asserts `WorkflowCompiler` no longer appears in `workflow/`
or its public surface, so spec 02 starts from a clean namespace.

## Out of scope

- Merging `WorkflowBuilder` + `WorkflowGraphCompiler` into `WorkflowCompiler` — **spec 02**. (That spec claims the freed name.)
- Folding the codec methods onto `CompiledWorkflow` and deleting `Workflow.to_dict`/`from_dict` — **spec 02** (when `Workflow` dissolves).
- Computing `TaskSnapshot` / `WorkflowVersion` / binding at compile time — **spec 02**.
- Per-task pydantic-graph node lowering / deleting `WorkflowStep` — **spec 03**.
- `WorkflowRuntime` rename + flat caching + asset-manifest re-registration — **spec 04**.
- Extending the executable IR to represent control/branch/loop specs (`to_dict` currently raises on those) — out of the whole chain; the IR stays data-DAG-only and that limit is documented, not fixed here.
