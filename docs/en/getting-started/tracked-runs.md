# Track a Run

The workflow layer describes computation. The workspace layer preserves the record. As soon as you care about reruns, parameter sweeps, or artifacts, you need both.

## The Persistent Hierarchy

MolExp stores state as four nested levels:

| Level | Role | Example |
|---|---|---|
| **Workspace** | Root directory | `./lab` |
| **Project** | Groups related work | `qm9` |
| **Experiment** | One repeatable definition | `baseline` |
| **Run** | One concrete execution | `run-abc123` |

An experiment says *what should be repeatable*. A run records *what actually happened*. That separation is the heart of the model.

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="baseline")

@wf.task
def train(lr: float) -> dict:
    return {"loss": lr * 100}

@wf.task(depends_on=["train"])
def report(loss: float) -> float:
    return loss

# Create the hierarchy
ws = me.Workspace("./lab", name="lab")
exp = ws.project("qm9").experiment("baseline")
```

## Execute One Tracked Run

`run.execute(wf)` is the entire driver — lifecycle, execution, and persistence in one call:

```python
run = exp.add_run(params={"lr": 1e-3})
result = run.execute(wf)
print(run.status, result.outputs["report"])  # succeeded 0.1
```

The run's directory now holds `run.json` (identity and provenance), `_ops/run.json` (status and ownership), and `executions/<exec_id>/` (per-task outputs). Read it back in a later session:

```python
same_run = exp.get_run(run.id)
print(same_run.status, same_run.get_result("report"))  # succeeded 0.1
```

## Failure, Resume, Rerun

A failing task raises `RunFailedError` and the run is persisted as `failed`. MolExp **never falls back silently** — each state either executes, resumes, or raises with instructions:

| Run status | `run.execute(wf)` behavior |
|---|---|
| `pending` | Execute from the top |
| `failed` / `cancelled` | **Resume** — seed completed tasks, recompute the rest |
| `failed` / `cancelled` + `rerun=True` | **Rerun** — fresh attempt from the top |
| `succeeded` | Refuses — read results instead |
| `running` | Refuses — cancel first (`run.cancel()`) |

```python
# docs: skip — resume/rerun only applies to failed/cancelled runs; the run above succeeded
# Resume a failed run (reuse completed task outputs)
run.execute(wf)

# Rerun from scratch in a new attempt
run.execute(wf, rerun=True)

# Rerun and also bypass the content-addressed cache
run.execute(wf, rerun=True, fresh=True)
```

## Sweeps

One workflow on many parameter cells. `exp.sweep()` materializes one run per cell; `RunSet.execute()` drives every pending run:

```python
scan = ws.project("qm9").experiment("lr-scan").sweep(wf, {"lr": [1e-3, 1e-4, 1e-5]})
summary = scan.execute()

# Each row: params + status + per-task outputs
for row in summary.to_records():
    print(row["lr"], row["status"], row["report"])

# Find the best
best = summary.min_by("report")
print(best["lr"], best["run_id"])
```

`to_records()` yields plain dicts — use any analysis stack you want. `min_by()` / `max_by()` find the run with the smallest or largest value for a given output key.

## CLI Registration

Bind the compiled workflow to the experiment so `molexp run` can discover it:

```python
exp.run(wf.compile(), params={"lr": [1e-3, 5e-4]})
```

Now `molexp run` owns run selection, profiles, resume flags, and scheduler-backed execution over the exact same runs. See [CLI and Profiles](cli-and-profiles.md).

## Add a Run with a Fixed ID

For reproducible references, pin a run id explicitly:

```python
pinned = exp.add_run(params={"lr": 5e-4}, id="baseline-default")
```

Sweep-derived runs get content-addressed ids from their parameters, so re-declaring the same sweep is idempotent.

## Runnable Example

`examples/getting_started/03_tracked_run.py` executes one tracked run and prints the on-disk layout.
