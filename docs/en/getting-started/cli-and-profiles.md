# CLI and Profiles

Once your script can create a workspace and bind experiments, replace `asyncio.run(...)` with `molexp run`. This is also where profiles become useful — one script, many execution shapes.

## Register for CLI Discovery

`molexp run` discovers workspaces through the fluent declaration chain. The key line is `Experiment.run(workflow, params=...)` — it seeds the runs, binds the workflow, and registers the workspace:

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="sum")

@wf.task
def fetch(scale: float) -> list[float]:
    return [1.0 * scale, 2.0 * scale, 3.0 * scale]

(
    me.Workspace("./lab", name="lab")
    .project("demo")
    .experiment("sum")
    .run(wf.compile(), params={"scale": [1.0, 2.0]})
)
```

Now the CLI can drive it:

```bash
molexp run train.py
```

The CLI resolves the script, scans the workspace hierarchy, finds eligible (`pending`) runs, and executes them. You no longer create runs manually in Python just to execute them.

## Profiles: Execution Variants in YAML

When one script needs several execution shapes, put the variation in `molcfg.yaml` — don't clone the script:

```yaml
version: 1

defaults:
  epochs: 100
  optimizer:
    lr: 0.001

profiles:
  smoke:
    epochs: 3

  dry-run:
    epochs: 1
    skip_heavy_compute: true
```

Task parameters receive profile fields by name:

```python
@wf.task(depends_on=["fetch"])
def compute(fetch: list[float], optimizer: dict | None = None, skip_heavy_compute: bool = False) -> float:
    if skip_heavy_compute:
        return 0.0
    lr = (optimizer or {}).get("lr", 1.0)
    return sum(fetch) * lr
```

`fetch` binds from the upstream task (graph edge). `optimizer` and `skip_heavy_compute` bind from the resolved profile. Each falls back to its default when the active profile omits the field.

The important design choice: MolExp stores the profile and injects it, but **your task code decides what the keys mean**. There is no built-in meaning for `epochs` or `skip_heavy_compute`.

## CLI Verbs

Each verb owns a **disjoint** job. Nothing overlaps, nothing falls back silently:

| Verb | Domain | What it does |
|---|---|---|
| `molexp run` | `pending` only | Creates missing runs, executes pending ones |
| `--resume` | `failed` / `cancelled` | Reopens the existing execution, seeds completed tasks |
| `--rerun` | `failed` / `cancelled` | Opens a fresh execution from the top |
| `--rerun --fresh` | `failed` / `cancelled` | Like `--rerun` but also bypasses the cache read |

`succeeded` and `running` runs are always skipped — that's by design. Retrying is always explicit.

```bash
molexp run train.py --profile smoke
molexp run train.py --profile smoke --override optimizer.lr=0.0005
molexp run train.py --profile smoke --resume
molexp run train.py --profile smoke --rerun --fresh
```

`--resume` and `--rerun` are mutually exclusive. Both are profile-aware: the resolved profile is part of the run's identity, so a different profile addresses a different run.

## Why Content-Addressed IDs Matter

The CLI folds parameters, profile, and replica index into the run id. Running `molexp run train.py` twice with the same params discovers the *same* run — the second invocation sees it's already `succeeded` and skips it. That idempotence means CI scripts can call `molexp run` without guards.

## Next Steps

- For the deeper profile model, see [Run Profiles and Reproducible CLI Execution](../guide/run-profiles.md).
- For what gets written to disk, see [Workflow Persistence](../guide/workflow-persistence.md).

## Runnable Example

`examples/getting_started/04_cli_and_profiles/` ships a `train.py` + `molcfg.yaml` pair.
