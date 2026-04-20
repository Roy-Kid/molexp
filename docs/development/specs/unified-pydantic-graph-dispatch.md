# Unified pydantic-graph Dispatch: Merging Sweep and Backend into the Outer Graph

**Status**: Draft · **Author**: @RoyKid · **Date**: 2026-04-17

## Implementation Status

| Phase | Status | Landing date | Notes |
|-------|--------|--------------|-------|
| 1 — Sweep-level pydantic-graph (only `--local` + `-j N`) | ✅ Done | 2026-04-17 | New `molexp.sweep` package, `-j/--jobs` CLI flag, `jobs:` field convention in profiles; `molexp run --local -j N` enables sweep-level concurrency |
| 2 — Unified backend entry point (`--backend`, `--block`) | ⏳ Not started | — | Requires `molq.JobHandle.wait_async` (or wrapping `asyncio.to_thread`) first |
| 3 — Per-node backend (`@wf.task(backend=..., resources=...)`) | ⏳ Not started | — | Depends on Phase 2 |
| 4 — molq native async API | ⏳ Tracked separately | — | Depends on `monitor.py` being async in the molq repo |

**Phase 1 changes**:

- `src/molexp/sweep/__init__.py`, `src/molexp/sweep/graph.py` (`SweepReplica`, `SweepState`, `SweepRoot`, `run_sweep`)
- `src/molexp/cli/run_cmd.py`: `_execute_sweep` is split into `_discover_runs` + `_dispatch_local` (pydantic-graph) + `_dispatch_cluster` (old path retained); adds the `-j/--jobs` CLI parameter and the `_resolve_jobs` helper
- `tests/test_sweep/test_graph.py` (9 unit tests) + `tests/test_cli_run_jobs.py` (6 E2E tests)

**Current profile convention**: `jobs: <int>` is user data (schemaless — see §2 and `molcfg-profiles.md`). The CLI `-j` flag takes precedence over the profile; profile values > 1 enable parallelism; default is 1 (backwards compatible).

## 1. Motivation

`molexp run` currently has two completely independent layers of scheduling:

| Layer | Location | Parallelism model | Snapshot / resume |
|-------|----------|-------------------|-------------------|
| **Sweep** (across experiment × replica) | `for` loop in `cli/run_cmd.py:_execute_sweep` | `--local` is forced serial; `--slurm` hands off to the external scheduler | Matched only by run-id against the store — no in-process state |
| **Workflow** (task DAG) | `workflow/_pydantic_graph/` | Same-level `asyncio.gather` | pydantic-graph snapshots available |

That produces four problems:

1. **`--local` cannot parallelize.** Multiple experiments with no interdependencies get serialized by the `for` loop + blocking `asyncio.run()` (see `cli/run_cmd.py:206-209, 467-472`).
2. **Two dispatch paths.** `_local_handler` and `SubmitHandler` are separate code paths for the same "get a run running" operation (`run_cmd.py:452-482` vs `run_cmd.py:484-543`).
3. **`--slurm` is fire-and-forget and unobservable.** The parent process exits right after submission, so the sweep layer has no graph state; `molexp watch` has to reverse-engineer progress via database polling.
4. **No per-node backend.** A common pattern — `prepare_data` local, `train_model` on a GPU cluster — is impossible because `--slurm` is a script-wide switch.

**Goal**: model the sweep as a pydantic-graph too, so that `local` / molq / any future backend becomes **an `await` inside one node's body** rather than a top-level dispatch switch.

**Non-goals**:

- No changes to molq internals (the job store, reconciler, and monitor stay as-is).
- No changes to the inner workflow task protocol (`@wf.task` semantics are preserved).
- No sub-process pool for local execution. "Local" means in-process async concurrency. Real parallelism for GPU training is delegated to molq's local scheduler or a remote scheduler.

## 2. Design Principles

1. **pydantic-graph is the only orchestrator.** Every execution concern — serial/parallel, local/remote, snapshot/resume — is expressed by graph nodes + async bodies. There is no scheduling loop outside the graph.
2. **Backend is an implementation detail of the node body.** `@wf.task` doesn't care where it runs; the backend is resolved at runtime from decorator args, the profile, and CLI overrides.
3. **Zero-change default.** Existing scripts (e.g. `examples/train_allegro_qm9.py`) run unchanged on the new architecture. New capabilities are opt-in via profile / decorator args.
4. **User-controlled block semantics.** A `block=True` node `await`s until the job terminates (parent must stay alive); `block=False` returns the `job_id` immediately and delegates subsequent state to the molq store.
5. **Three-tier account/resources inheritance.** Profile default → decorator default → CLI/ctx override. Users only override at "special" nodes.
6. **Single backend selector.** The CLI exposes `--local` and `--scheduler <name>` only; there are no per-scheduler aliases.

## 3. Core Architecture

### 3.1 Two graph layers

```
SweepGraph (outer)
  Nodes: one per (experiment × replica)
  Edges: none (replicas are independent)
  Concurrency: asyncio.Semaphore(jobs) throttles same-level concurrency
  Body: await _run_one_replica(run, exp, profile_cfg)

       ├─ backend=local:
       │    inner = exp.workflow.execute(run_context=ctx)
       │    await inner                              # in-process
       │
       └─ backend=slurm/pbs/lsf:
            handle = submitor.submit(
                argv=[python, -m, worker, script, run_dir, --task <name>?],
                resources=..., execution=...)
            if block:
                record = await handle.wait_async()   # parent stays around
                return {"state": record.state.name, "job_id": handle.job_id}
            else:
                return {"job_id": handle.job_id}     # fire-and-forget

WorkflowGraph (inner, unchanged)
  Nodes: @wf.task
  Edges: depends_on
  Body:  existing async-def protocol
```

### 3.2 Per-node backend (advanced path)

Default: one remote job per experiment (the entire inner workflow is submitted as a single job). When users want hybrid execution (e.g. `prepare_data` local, `train_model` on GPU), they annotate an inner `@wf.task` with `backend=`; the framework submits just that task as an independent molq job:

```python
@wf.task(depends_on=["prepare_data"],
         backend="slurm",
         resources=TaskResources(gpus=1, time="12h", mem="64G"))
async def train_model(ctx): ...
```

On the remote side, the same `worker.py` entry point takes a `--task <name>` argument to run just that one node instead of the whole workflow (see §5.3).

### 3.3 Concurrency control

```
molexp run script.py -j 4                    # 4-way sweep concurrency
molexp run script.py -j 4 --backend slurm    # 4 simultaneous slurm submissions + awaits
molexp run script.py                         # -j defaults to 1 (= current behaviour)
```

`-j N` is implemented as an `asyncio.Semaphore(N)` inside the outer `SweepGraph` node executor — directly reusing the `_execute_parallel` pattern from `workflow/_pydantic_graph/node.py:88-124`.

## 4. User-facing API

### 4.1 CLI

```bash
# Recommended new form
molexp run script.py --backend {local,slurm,pbs,lsf,...} \
                     [-j N] [--block/--no-block] \
                     [-c molcfg.yaml] [--profile NAME]

# Old flags as aliases (zero-cost migration)
molexp run script.py --local        # ≡ --backend local -j 1
molexp run script.py --slurm        # ≡ --backend slurm --no-block (preserves fire-and-forget)
```

**Defaults**:

- `--backend`: `local`
- `-j`: `1` (backwards compatible; explicit `-j auto` = CPU count)
- `--block`: meaningless under `local` (the body already awaits); under `slurm` the default is `--no-block` to match old behaviour. A future version may flip this to `--block` by default based on community feedback.

### 4.2 Profile (molcfg.yaml)

```yaml
default: &base
  smoke: false

profiles:
  local_quick:
    <<: *base
    backend: local
    jobs: 4                        # equivalent to -j

  prod_gpu:
    <<: *base
    backend: slurm
    block: true                    # parent waits
    jobs: 8                        # 8 concurrent submit-and-wait
    slurm:
      cluster: "perlmutter"
      account: "mycompchem"
      qos: "regular"
      partition: "gpu"

  mixed:
    <<: *base
    backend: local                 # outer sweep runs locally with concurrency
    jobs: 16
    per_node_backend: true         # allow inner `backend=` overrides
    slurm:                         # inherited when an inner task asks for slurm
      cluster: "perlmutter"
      account: "mycompchem"
```

### 4.3 Decorator (in scripts)

```python
from molexp import TaskResources

@wf.task(
    depends_on=["prepare_data"],
    backend="slurm",                                        # per-node override
    resources=TaskResources(gpus=1, time="12h", mem="64G"),
)
async def train_model(ctx): ...
```

**Resolution order**: CLI override > profile > decorator > framework default. Organization-level fields like `account` / `cluster` / `qos` are **always** read from the profile, never from the script (so scripts remain portable across machines).

### 4.4 TaskContext additions

```python
@wf.task
async def some_task(ctx: TaskContext):
    # New: manual submit (advanced users)
    handle = await ctx.molq.submit(
        argv=[...], resources=..., execution=...)
    record = await handle.wait_async()
    ...
```

`ctx.molq` is available even on non-molq backends (falling back to molq's `local` scheduler), so user code stays independent of the backend selected on the CLI.

## 5. Internal Implementation

### 5.1 SweepGraph

New file `molexp/sweep/graph.py`:

```python
from pydantic_graph import BaseNode, End, GraphRunContext
import asyncio

@dataclass
class ReplicaNode(BaseNode[SweepState, SweepDeps, SweepResult]):
    mol_run: Run
    experiment: Experiment

    async def run(self, ctx) -> "ReplicaNode | End[SweepResult]":
        async with ctx.deps.semaphore:             # -j N throttle
            backend = _resolve_backend(
                ctx.deps.profile_cfg,
                self.experiment)
            if backend == "local":
                await self.experiment.workflow.execute(
                    run=self.mol_run,
                    profile_config=ctx.deps.profile_cfg)
            else:
                await _submit_and_maybe_wait(
                    backend=backend,
                    mol_run=self.mol_run,
                    profile_cfg=ctx.deps.profile_cfg)
        # ReplicaNode is a leaf — end directly
        return End(...)

def build_sweep_graph(workspaces, profile_cfg, jobs: int) -> Graph:
    # Aggregate every (project, experiment, replica) into a single-level parallel graph
    ...
```

### 5.2 `_execute_sweep` refactor

The current 94-line `_execute_sweep` in `cli/run_cmd.py` collapses to:

```python
async def _execute_sweep_async(script, profile_cfg, resume, workspace, jobs):
    workspaces = load_workspaces(script)
    runs = _discover_runs(workspaces, profile_cfg, resume)   # keep current resume logic
    graph = build_sweep_graph(runs, profile_cfg, jobs=jobs)
    await graph.run(...)

def _execute_sweep(...):
    asyncio.run(_execute_sweep_async(...))
```

`_local_handler` and `SubmitHandler` are removed (their logic migrates into `ReplicaNode.run`).

### 5.3 `worker.py` extension

`molexp/plugins/submit_molq/worker.py` gains a `--task <name>` option:

```python
# Old: python -m worker <script> <run_dir>
#      (runs the whole workflow)
# New: python -m worker <script> <run_dir> [--task <node_name>]
#      Without --task: same behaviour (whole workflow)
#      With --task: runs only the named node (used by per-node backend)
```

### 5.4 molq async adapter

molq's `JobHandle.wait()` is a blocking poll (`monitor.py:40-96`). Transitional plan:

```python
# molexp-side stopgap
async def wait_async(handle, *, timeout=None):
    return await asyncio.to_thread(handle.wait, timeout=timeout)
```

Long-term: add `JobHandle.wait_async()` to molq itself, rewriting the `threading.Event.wait(interval)` loop as `asyncio.sleep(interval)`. Tracked as a separate PR; does not block this spec.

### 5.5 Snapshot / resume

pydantic-graph's snapshot mechanism applies naturally to `SweepGraph`. New capabilities:

- **Parent-crash recovery**: the snapshot stores each replica's `job_id` (if submitted). On restart, for jobs still running the node re-awaits `handle.wait_async()`; the molq store queries state idempotently by `job_id`.
- **`--resume` semantics**: reuses the run-id matching logic at `run_cmd.py:179-198`; the entry point changes from a `for` loop to `SweepGraph.resume(snapshot)`.

## 6. Migration Path

### Phase 1: sweep-level pydantic-graph (only `--local`)

- Introduce `SweepGraph` with `backend=local` only
- `-j N` takes effect under `--local`
- Other backends keep going through the old `SubmitHandler` for now (two code paths coexist)
- **User-visible change**: `molexp run --local -j 4` starts working
- **Risk**: low — pure-addition change; old paths untouched

### Phase 2: unified backend entry point

- Delete `_local_handler` and `SubmitHandler`; every backend routes through `SweepGraph`
- `ReplicaNode.run` dispatches on `backend` (local direct-run / molq submit)
- Add `--block/--no-block`
- `worker.py` behaviour unchanged
- **User-visible change**: `--backend slurm --block` appears; `--slurm` remains available (equivalent semantics)
- **Risk**: medium — slurm path rewritten; needs real-cluster regression testing

### Phase 3: per-node backend

- `@wf.task(backend=..., resources=...)` takes effect
- `worker.py` gains `--task`
- Profile gets `per_node_backend: true` switch (default false to avoid accidental activation)
- **User-visible change**: advanced users can mix backends
- **Risk**: medium — best-practice docs on node I/O handoff needed

### Phase 4: molq async API

- Native molq `wait_async()` ships; molexp drops the `asyncio.to_thread` wrapper
- Tracked separately from Phases 1/2/3

### Time estimate

| Phase | Effort | Dependency |
|-------|--------|------------|
| 1 | 2–3 days | none |
| 2 | 3–4 days | Phase 1 |
| 3 | 3–5 days | Phase 2 |
| 4 | 2–3 days (molq-side) | independent |

## 7. Impact on Existing Scripts

Using `molnex/examples/train_allegro_qm9.py` as reference:

| Snippet | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| `experiments = [...]` (73-85) | unchanged | unchanged | unchanged |
| `@wf.task prepare_data` (174-232) | unchanged | unchanged | optional: add `backend="local"` explicitly (not required) |
| `@wf.task train_model` (235-416) | unchanged | unchanged | optional: `backend="slurm", resources=...` to enable per-node submit |
| `me.entry(ws)` (428) | unchanged | unchanged | unchanged |
| `molcfg.yaml` | can add `jobs:` | can add `backend:` / `block:` | can add `per_node_backend: true` |

**Conclusion**: user script code stays the same. Every new capability is opt-in via profile or decorator.

## 8. Compatibility

### 8.1 CLI alias table

| Old flag | New equivalent |
|----------|----------------|
| `--local` | `--backend local -j 1` |
| `--local --bg` | `--backend local -j 1` + a `nohup` wrapper (`--bg` behaviour preserved) |
| `--slurm` | `--backend slurm --no-block` |
| `--slurm --block` | `--backend slurm --block` (same semantics as today) |
| `--scheduler local` | `--backend local` (for the molq local scheduler it is still `--backend slurm` against the molq local cluster) |

### 8.2 Breaking changes

**None**. All existing invocations keep working. After Phase 2 the `SubmitHandler` class API disappears, but it is an internal implementation detail, not part of the public contract.

### 8.3 Config-file compatibility

- Old profiles without `backend:` / `jobs:` default to `backend: local, jobs: 1` = old behaviour.
- New field positions: see §4.2.

## 9. Testing Strategy

### 9.1 New unit tests

- `tests/test_sweep/test_graph.py`: `SweepGraph` basic execution, concurrency, snapshot
- `tests/test_sweep/test_backend_resolution.py`: CLI > profile > decorator precedence
- `tests/test_sweep/test_resume.py`: restart from snapshot after parent crash

### 9.2 Integration tests

- `molexp run --backend local -j 4` actual concurrency (measured via sleeps inside tasks)
- `molexp run --backend slurm --block` end-to-end on molq's `testing` scheduler (in-memory simulation)
- Mixed backends: one task `local` + one task `slurm`; verify input/output handoff

### 9.3 Regression tests

- Existing `tests/test_cli/test_run_cmd.py` must continue to pass
- `examples/train_allegro_qm9.py` runs to completion with `--backend local --smoke` after Phase 2

## 10. Open Questions

1. **Exit code for `molexp run --no-block`**: the job was submitted but hasn't finished — does `0` mean "submission OK" or does non-zero mean "completion not verified"? Leaning toward `0` + an explicit stdout note that `molexp watch` is the next step.
2. **Meaning of `-j auto`**: CPU count? GPU count? Fine for local; for slurm it could be an unreasonable "concurrent watcher cap". Likely: `auto` only applies to `local`; slurm requires an explicit `-j N`.
3. **Per-node snapshot granularity**: when a replica contains multiple nodes that each submit jobs, and one fails, should resume re-run the failed node or the whole replica? pydantic-graph defaults to per-node, which we inherit; doc clarity needed.
4. **`TaskResources` schema**: reuse molq's `JobResources` (`gpu_count` / `memory` / `time_limit`) directly, or wrap in a molexp type? Leaning toward reusing to minimize conversion.
5. **`ctx.molq` on the local backend**: fall back to molq's `local` scheduler or raise? Leaning toward the fall-back — keeps user code backend-agnostic.

## 11. References

- `cli/run_cmd.py:206-209, 452-543` — current sweep dual-path implementation
- `workflow/_pydantic_graph/node.py:88-124` — inner-level parallelism template
- `plugins/submit_molq/submit.py:52-115` — current SubmitHandler
- `plugins/submit_molq/worker.py:18-56` — existing worker entry
- `molq/submitor.py:197-251, 1120-1154` — current JobHandle surface
- `molq/monitor.py:40-96` — blocking poll awaiting async rewrite
- Related spec: `molcfg-profiles.md` (profile mechanism)
