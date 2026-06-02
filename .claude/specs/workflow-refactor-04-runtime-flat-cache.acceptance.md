---
slug: workflow-refactor-04-runtime-flat-cache
criteria:
  - id: ac-001
    summary: GraphWorkflowRuntime renamed to WorkflowRuntime with a flat cache member
    type: code
    pass_when: |
      `from molexp.workflow import GraphWorkflowRuntime` raises ImportError;
      WorkflowRuntime is exported and exposes `cache` as a plain instance
      attribute (Caching | None), not nested behind a policy/store object.
      Asserted in tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-002
    summary: a cache hit skips the task body
    type: code
    pass_when: |
      Running a fixture workflow twice against a shared workspace-backed Caching,
      a per-task execution counter increments on run 1 and does NOT increment on
      run 2 (body skipped). Asserted in tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-003
    summary: artifacts are re-registered on a hit via content-hash, no recompute
    type: runtime
    pass_when: |
      For an artifact-producing task, run 2 (cache hit) makes the artifact
      resolvable in the current run with a content_hash byte-identical to run 1,
      without executing the producer body; downstream tasks consume it normally.
      Asserted in tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-004
    summary: cache key is keyed on snapshot identity + input hash
    type: code
    pass_when: |
      Changing a task's config or code (changing snapshots[name].key) flips the
      key and forces a miss; identical inputs+identity hit. Asserted in
      tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-005
    summary: caching is opt-in and never caches Actors
    type: code
    pass_when: |
      With cache=None the body runs on every execution (no caching); with a cache
      present, Actor/streaming tasks are still never cached. Asserted in
      tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-006
    summary: Caching core stays pure-JSON (manifest is JSON; cache_store unchanged)
    type: code
    pass_when: |
      CacheEntry.result holds {result, artifacts:[{name,kind,content_hash,
      asset_id}]} as JSON; cache.py/cache_store.py public contracts are unchanged;
      the legacy test_workspace_backed_cache.py still passes. Asserted in
      tests/test_workflow/test_runtime_cache.py.
    status: pending
  - id: ac-007
    summary: no regression with caching off; full gate green
    type: code
    pass_when: |
      All existing runtime/fixture tests run byte-identically with cache=None;
      `ruff format --check src/ tests/ && ruff check src/ tests/ && ty check src/
      && pytest tests/` all pass.
    status: pending
---

# Acceptance — workflow-refactor-04-runtime-flat-cache

"Done" means the runtime is renamed with a flat `cache` member (ac-001), a hit
skips the body (ac-002) and re-registers artifacts by content-hash with no
recompute (ac-003), the key is snapshot-identity + input-hash (ac-004), caching
is opt-in and Actor-exempt (ac-005), `Caching` stays pure-JSON (ac-006), and
nothing regresses with caching off (ac-007). This closes the 4-spec chain and
makes molexp's `Caching` a live, execution-wired feature.
