# Parameter Sweeps

A sweep fans one workflow out over a grid of parameter cells — one content-addressed `Run` per cell — and turns the batch back into analyzable records. The whole loop is three calls: `exp.sweep(...)` seeds, `RunSet.execute()` runs, and the returned `RunSetResult` summarizes.

## Seeding a Sweep

`Experiment.sweep(workflow, params)` expands `params` and materializes one run per cell. A plain `{axis: [values]}` dict is auto-upgraded to a `GridSpace` (Cartesian product); every axis must map to a *list* — a scalar axis fails fast rather than being guessed at. The workflow may be an uncompiled `WorkflowCompiler`; it is compiled automatically and bound to the experiment, so the CLI and server discover the same declaration.

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="scan")


@wf.task
def simulate(lr: float, batch: int) -> float:
    return lr * batch


@wf.task(depends_on=["simulate"])
def score(simulate: float) -> float:
    return 1.0 / simulate


ws = me.Workspace("./lab", name="lab")
exp = ws.project("demo").experiment("lr-scan")
runset = exp.sweep(wf, {"lr": [0.1, 0.2], "batch": [16, 32]})
print(len(runset), "runs seeded")
```

Each run's id derives from its parameters, so seeding is idempotent: re-declaring the same sweep rediscovers the same four runs, never duplicates. Each run's parameters bind by name to the workflow's root-task parameters at execution time — `simulate(lr, batch)` receives its cell directly.

## Executing the Batch

`RunSet.execute()` drives every **pending** run through the same tracked execution path as `molexp run` (run lifecycle, status machine, persisted per-task outputs). `parallel=` bounds how many runs execute concurrently:

```python
summary = runset.execute(parallel=2)
for row in summary.to_records():
    print(row["lr"], row["batch"], row["status"], row["score"])
```

Runs outside the pending domain are left alone — a succeeded run is never silently recomputed, and retrying a failure stays an explicit per-run verb (`run.execute(wf)` to resume, `fresh=True` to rerun). They still appear in the summary with their current status and persisted outputs.

## Reading the Summary

`RunSetResult.to_records()` flattens each run into one dict: the parameter cell, every task's output keyed by task name, and the reserved identity columns `run_id` / `status` / `error`. `min_by` / `max_by` pick the extreme record by any column, in pure Python:

```python
best = summary.max_by("score")
print(best["lr"], best["batch"], best["run_id"])
```

The records are plain dicts — molexp deliberately ships no analysis-stack bridge. Feed `to_records()` to whatever you already use.

## Failures Are Recorded, Not Propagated

One cell blowing up must not cost you the other ninety-nine. A failing run is persisted as `failed` with its error captured, the summary reports it honestly, and its siblings keep executing:

```python
risky = WorkflowCompiler(name="risky")


@risky.task
def invert(x: float) -> float:
    return 1.0 / x


outcome = ws.project("demo").experiment("edge-cases").sweep(risky, {"x": [0.0, 2.0]}).execute()
for row in outcome.to_records():
    print(row["x"], row["status"], row["error"])
print(len(outcome.failed), "run(s) failed")
```

The `x=0.0` row comes back `failed` with `error="ZeroDivisionError: float division by zero"`; the `x=2.0` row succeeds. Retry the failed cells individually (`run.execute(wf)` resumes them) or fix the workflow and rerun with `fresh=True`.

## Re-running and Reading Back

Because seeding is idempotent and `execute` skips non-pending runs, the whole sweep script is safely re-runnable — a second invocation recomputes nothing and still returns the full summary from persisted outputs:

```python
summary2 = exp.sweep(wf, {"lr": [0.1, 0.2], "batch": [16, 32]}).execute()
print(all(row["status"] == "succeeded" for row in summary2.to_records()))
```

A later session (or another tool) reads the finished sweep back without touching the workflow at all:

```python
records = exp.runs().collect().to_records()
print(len(records))
```

`Experiment.runs()` wraps the on-disk runs in a `RunSet`; `collect()` summarizes their current state — statuses and each run's latest persisted per-task outputs — without executing anything.

## Random Sampling

Any `ParamSpace` drops in where the grid dict goes. `UniformSpace` samples cells instead of enumerating the full product:

```python
from molexp import UniformSpace

space = UniformSpace({"lr": [0.1, 0.2, 0.3], "batch": [16, 32]}, n_samples=3, seed=42)
sampled = exp.sweep(wf, space)
print(len(sampled))
```

Identical sampled cells collapse onto the same content-addressed run, so the `RunSet` may contain fewer distinct runs than `n_samples`.

## Scaling Past One Machine

`RunSet.execute` runs in-process. The same seeded runs are equally drivable by the CLI — `molexp run train.py` with `--scheduler`/`--compute-target` submits them to a cluster, and `--resume`/`--rerun` retry the failed subset — because both fronts share one execution path per run. See [CLI and Profiles](../getting-started/cli-and-profiles.md) and [molq Integration](molq.md).
