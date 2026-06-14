# CLI and Profiles

Once a workflow script can create a workspace and bind one or more experiments, the next step is usually to stop driving it manually with `asyncio.run(...)` and let `molexp run` do the work. That shift is also where profile-driven execution becomes useful.

## Registering the Workspace for CLI Discovery

`molexp run` does not inspect your source tree by magic. It imports the script and discovers the workspaces registered by the fluent declaration chain.

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="sum")
# ... @wf.task definitions ...

(
    me.Workspace("./lab", name="lab")
    .project("demo")
    .experiment("sum")
    .run(wf.compile(), params={"scale": [1.0]})
)
```

`Experiment.run(workflow, params=...)` is the bridge between a normal Python module and the CLI's discovery path: it seeds the runs, binds the workflow, and registers the workspace as a CLI entry point. (The low-level `me.entry(ws)` primitive still exists for registering a pre-built workspace, but the fluent chain calls it for you.)

## Running the Script

Once the workspace is registered, the CLI can resolve the script, scan the workspace hierarchy it exposes, and execute eligible runs:

```bash
molexp run train.py
```

From that point on, you no longer need to create the run manually in Python just to execute it. The CLI handles run selection, deterministic run ids, and the `RunContext` lifecycle for you.

## Putting Execution Variants in `molcfg.yaml`

As soon as one script needs several execution shapes, it is better to keep that variation in config rather than by cloning the script or piling more ad hoc flags into task code.

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

Tasks then read the active configuration through `ctx.config`:

```python
@wf.task(depends_on=["fetch"])
async def compute(ctx: TaskContext) -> float:
    if ctx.config.get("skip_heavy_compute"):
        return 0.0
    return sum(ctx.inputs) * ctx.config.get("optimizer", {}).get("lr", 1.0)
```

The important design choice is that MolExp stores the profile and injects it, but does not assign domain meaning to its keys. Your task code decides what `epochs`, `skip_heavy_compute`, or `optimizer.lr` actually mean.

## Profiles, Overrides, and Resume

The CLI can resolve one profile, apply one-off overrides after that resolution, and then persist the chosen config onto the run metadata:

```bash
molexp run train.py --profile smoke
molexp run train.py --profile smoke --override optimizer.lr=0.0005
molexp run train.py --profile smoke --resume
```

`--resume` is profile-aware. A resumed run must match the selected profile and must not already be `succeeded`. This keeps resume tied to one execution stream rather than letting a new profile reinterpret an old run.

The CLI also folds parameters, replica index, and active profile metadata into the run id it generates. That is why repeated `molexp run` invocations can rediscover the same run — the same content-addressing that makes `exp.run(..., params=...)` idempotent. Only `exp.add_run(...)` without an explicit `id=` creates a fresh run identity.

## Where to Go for More Detail

This page is the operational starting point, not the full reference. If you want the deeper profile model, continue with [Run Profiles and Reproducible CLI Execution](../guide/run-profiles.md). If you want the persistence side of what gets written into `run.json`, continue with [Workflow Persistence](../guide/workflow-persistence.md).

## Runnable Example

`examples/getting_started/04_cli_and_profiles/` ships a `train.py` plus `molcfg.yaml` pair — execute it with `molexp run examples/getting_started/04_cli_and_profiles/train.py --profile smoke`.
