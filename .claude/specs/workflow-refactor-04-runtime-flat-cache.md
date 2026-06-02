---
title: Workflow refactor 04 — WorkflowRuntime + flat caching + asset-manifest re-registration
status: draft
created: 2026-06-02
notes: |
  Final spec of the 4-spec workflow/ refactor. Renames GraphWorkflowRuntime →
  WorkflowRuntime, gives it a flat self.cache member, and finally WIRES the
  orphaned Caching subsystem into execution at the per-task node hook (from 03).
  Because heavy outputs are file artifacts, the cache stores an asset MANIFEST
  (Caching stays pure-JSON) and the runtime re-registers artifacts on a hit via
  the content-addressed FileArtifactStore. Opt-in (cache=None → off). Closes the
  chain; afterwards the electrolyte builder's hand-rolled _fingerprint reuse is
  deletable (tracked downstream, not here).
---

# Workflow refactor 04 — WorkflowRuntime + flat caching + asset-manifest re-registration

## Summary

The `Caching` / `CacheStore` / `TaskSnapshot` subsystem is fully built and unit-
tested (`tests/test_workflow/test_workspace_backed_cache.py`) but **orphaned**:
the runtime invokes `body.execute(task_ctx)` directly with no cache lookup, so
executing a workflow gives zero result caching. This spec connects it.

Rename `GraphWorkflowRuntime` → **`WorkflowRuntime`** and give it a **flat**
`self.cache: Caching | None` member (no policy/store tower — directive "要扁平").
Wire caching into the per-task `_TaskNode.run` hook introduced in 03: before
running a body, build the key from `CompiledWorkflow.snapshots[name].key |
input_hash(inputs)`, `cache.get`; on hit skip the body and replay; on miss run
then `cache.put`.

The wrinkle that kept this unwired: `Caching` memoizes **JSON results**
(`CacheEntry.result`), but the expensive tasks produce **file artifacts**
(prmtops, `.data`, …). Resolution (the chosen design): the cache stores an
**asset manifest** alongside the JSON result — for each artifact the task
produced (queried from the run's asset catalog after a miss): `{name, kind,
content_hash, asset_id}`. On a **hit**, the runtime re-registers those artifacts
into the *current* run via the content-addressed `FileArtifactStore` (idempotent
`put` on `(kind, content_hash)` — the bytes already live in the store keyed by
hash, so re-registration is a catalog upsert, **no recompute, no byte recopy**),
then replays the JSON result. `Caching` itself stays pure-JSON; the manifest
assembly + re-registration live in the runtime/node, keeping the cache decoupled
from the asset layer.

Caching is **opt-in**: `cache=None` (default) → no caching, behavior identical to
today. It applies to batch `Task`s only — `Actor`/streaming bodies are never
cached (their output is a drained stream).

## Design

### `WorkflowRuntime` (rename + flat cache)
- `class WorkflowRuntime` (was `GraphWorkflowRuntime`), `self.cache: Caching | None
  = None` direct field. `execute`/`start`/`run_on(compiled, …, cache=None)`; an
  explicit `cache=` arg wins, else `self.cache`, else — when a workspace
  run_context is present — auto-derive `Caching(store=run_ctx...cache.as_cache_store())`.
  The cache reaches `_TaskNode.run` via the existing `WorkflowDeps` transport (the
  single indirection pydantic-graph's deps mechanism forces; the *owner* is the
  flat runtime field).

### Cache hook (`_TaskNode.run`, from 03)
- Gate: `deps.cache is not None and not registration.is_actor`.
- `snap = compiled.snapshots[name]`; `key = Caching._compute_cache_key(snap.key,
  input_hash(inputs))` (reuse existing key logic; `inputs` already collected).
- `hit = cache.get(snap, inputs)`: on hit → re-register manifest artifacts (below),
  return `hit["result"]`, skip the body. On miss → run body, assemble manifest,
  `cache.put(snap, inputs, {"result": result, "artifacts": manifest})`.

### Asset manifest + re-registration
- After a miss runs, query the run's asset catalog for artifacts whose producer
  task is `name` (`run_ctx.run.assets.query(producer_task=name, kind="artifact")`)
  → manifest entries `{name, kind, content_hash, asset_id}`.
- On a hit, for each manifest entry: re-register into the current run — a catalog
  upsert pointing at the content-addressed store entry keyed by `content_hash`
  (idempotent; `FileArtifactStore.put_*` is keyed on `(kind, content_hash)`).
  Downstream tasks then resolve these artifacts in the current run exactly as if
  the producer had run.
- `CacheEntry.result` stays JSON (the manifest is JSON) — no `cache_store` change.

## Files to create or modify

- `src/molexp/workflow/_pydantic_graph/runtime.py` — rename class → `WorkflowRuntime`; add `self.cache`; `execute`/`start`/`run_on` gain `cache=`; auto-derive from workspace run_context.
- `src/molexp/workflow/_pydantic_graph/node.py` — cache hook in `_TaskNode.run` (gated on `cache` + non-actor); manifest assembly on miss; re-registration on hit.
- `src/molexp/workflow/_pydantic_graph/state.py` — add `cache: Caching | None` to `WorkflowDeps`; populated by the runtime.
- `src/molexp/workflow/cache.py` — small helper to (de)serialize the `{result, artifacts}` payload if needed; `Caching` core unchanged.
- workspace asset layer — a re-register-by-content-hash helper (method on the asset accessor / catalog) if one does not already exist.
- `src/molexp/workflow/__init__.py` — export `WorkflowRuntime` (drop `GraphWorkflowRuntime`); keep `Caching`/`CacheStore`/`FileCacheStore`.
- callers of `GraphWorkflowRuntime` (runtime entry points, `spec`/`compiled` execute delegators, CLI/server) — repoint to `WorkflowRuntime`.

## Tasks

- [ ] Failing test: a fixture task writes an artifact + returns a descriptor; run twice with a shared workspace-backed `Caching` → the 2nd run does NOT execute the body (a counter/sentinel proves the skip) and the artifact is present in the 2nd run via re-registration with an identical `content_hash`, byte-identical to the 1st.
- [ ] Failing test: `cache=None` never caches (body runs both times); `Actor`/streaming tasks are never cached even with a cache present.
- [ ] Failing test: changing a task's `config` or code flips the cache key (miss) — keyed on `snapshots[name].key | input_hash`.
- [ ] Failing test: `from molexp.workflow import GraphWorkflowRuntime` raises ImportError; `WorkflowRuntime` is exported and `self.cache` is a plain attribute.
- [ ] Implement `WorkflowRuntime` rename + flat `self.cache` + `cache=` plumbing through `WorkflowDeps`.
- [ ] Implement the `_TaskNode.run` cache hook: key, get/put, manifest assembly on miss.
- [ ] Implement content-hash re-registration on hit (workspace helper); wire it into the hook.
- [ ] Repoint all `GraphWorkflowRuntime` callers; update exports.
- [ ] Run the full gate: `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/ && pytest tests/`.

## Testing strategy

The headline test is the two-run skip-and-re-register: a body counter proves the
hit skipped execution, and the re-registered artifact's `content_hash` matches
the first run's (proving no recompute, content-addressed dedup). Negative tests:
`cache=None` off; `Actor` never cached; config/code change → miss. Existing
runtime/fixture tests run with `cache=None` and must stay byte-identical (the
rename + opt-in default introduce no behavior change). The legacy
`test_workspace_backed_cache.py` unit test of `Caching` remains green.

## Out of scope

- Deleting the electrolyte builder's hand-rolled `_fingerprint`/asset-tag reuse — that lives in a separate repo (`polymer_electrolyte`) and is tracked downstream; this spec only makes it *possible* by giving molexp first-class caching.
- Caching parallel fan-out per-element results (cache the join output only, if at all) — deferred; document the limitation.
- IR control-flow serialization extensions.
