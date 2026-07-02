# Track a Run

The workflow layer is enough to describe computation, but it is not enough to preserve a research record. As soon as you care about reruns, parameter sweeps, artifacts, or profile metadata, you need the workspace layer.

## The Persistent Hierarchy

MolExp stores persistent state as `Workspace -> Project -> Experiment -> Run`. The workspace is the root directory. The project groups related work. The experiment is the repeatable definition of one workflow plus one parameter sweep. The run is one concrete execution attempt under that definition.

That split is the heart of the model. An experiment is not the same thing as a run. The experiment says what should be repeatable. The run records what actually happened this time.

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


ws = me.Workspace("./lab", name="lab")
exp = ws.project("qm9").experiment("baseline")
```

`Workspace(...)` itself is lightweight; calling `ws.project(...)` is enough to create the real on-disk hierarchy.

## Executing One Tracked Run

`run.execute(workflow)` is the whole driver: it opens the run's tracked lifecycle (status transitions, execution history, ownership heartbeat — the same machinery `molexp run` uses), executes the graph with the run's `params` bound to the root task, persists every task's output under the run directory, and returns the result.

```python
run = exp.add_run(params={"lr": 1e-3})
result = run.execute(wf)
print(run.status, result.outputs["report"])
```

The run's directory now holds `run.json` (identity and provenance), the `_ops/` status sidecar, and one `executions/<exec_id>/` attempt with per-task outputs. Because those outputs are persisted, a later session can read them back without re-executing:

```python
same_run = exp.get_run(run.id)
print(same_run.status, same_run.get_result("report"))
```

## Failure, Resume, and Rerun

A failing task does not vanish into a return value: `run.execute` raises `RunFailedError` (carrying the partial result on `.result`), and the run is persisted as `failed` with the error recorded. The retry semantics follow the run's status:

- a **failed or cancelled** run: `run.execute(wf)` *resumes* — it reopens the same execution attempt, seeds the already-completed tasks from disk, and recomputes only the rest; `run.execute(wf, rerun=True)` *reruns* from the top in a new attempt instead (add `fresh=True` to also bypass the content-addressed cache read, mirroring `molexp run --rerun --fresh`).
- a **succeeded** run always refuses to re-execute — read its results instead; a different question is a different run (declare one with different params).
- a **running** run always refuses — cancel it first (`run.cancel()`).

Nothing ever falls back silently: each state either executes, resumes, or raises with instructions.

## Run Identity Is Content-Addressed

Runs seeded through a sweep derive their ids from their parameters, so re-declaring the same sweep is idempotent — repeated invocations rediscover the same runs instead of creating duplicates. When you create runs directly, `exp.add_run(params)` generates a fresh id unless you pass `id=...` yourself:

```python
pinned = exp.add_run({"lr": 5e-4}, id="baseline-default")
```

## Sweeps: Batch Execution and Summaries

One workflow usually wants many parameter cells. `exp.sweep(workflow, params)` materializes one content-addressed run per cell of the grid and returns a `RunSet`; `RunSet.execute()` drives every pending run through the same tracked path (a failing cell is recorded honestly and never interrupts its siblings), and the returned summary flattens each run into one record — params, per-task outputs, `run_id`, `status`, `error`:

```python
scan = ws.project("qm9").experiment("lr-scan").sweep(wf, {"lr": [1e-3, 1e-4]})
summary = scan.execute()
for row in summary.to_records():
    print(row["lr"], row["status"], row["report"])
best = summary.min_by("report")
print(best["lr"], best["run_id"])
```

`execute(parallel=n)` bounds concurrent runs, `summary.to_records()` yields plain dicts for any analysis stack, and `exp.runs().collect()` reads a finished sweep back from disk in a later session. See [Parameter Sweeps](../guide/sweeps.md) for the full tour.

## Declaring for the CLI

The same experiment can declare its workflow and sweep for CLI-driven execution:

```python
exp.run(wf.compile(), params={"lr": [1e-3]})
```

That binds the compiled workflow to the experiment, records its graph IR, and registers the workspace for discovery — `molexp run train.py` then owns run selection, profiles, resume flags, and scheduler-backed execution over the exact same runs. See [CLI and Profiles](cli-and-profiles.md).

## Beyond Results: Artifacts and Assets

Results are only one kind of persistent state. For files a run produces (`ctx.artifact.save(...)`), structured driver-side values (`ctx.set_result(...)`), logs, and reusable data assets, the run exposes a driver-side `RunContext` through `run.start()` — the advanced surface `run.execute` manages for you. See [Workspace API](../guide/workspace-api.md) and the [Unified Asset Model](../guide/assets.md).

## Why This Layer Matters

Without the workspace layer, a workflow run is just one process execution. With it, the same run becomes a durable record with parameters, profile data, execution history, artifacts, and reusable assets. That is the point where MolExp stops being only a graph runtime and starts becoming a research execution system.

The next practical step is usually to let the CLI discover and drive that same workspace, which is what [CLI and Profiles](cli-and-profiles.md) covers.

## Runnable Example

`examples/getting_started/03_tracked_run.py` executes one tracked run and prints the resulting on-disk layout plus the persisted run fields.
