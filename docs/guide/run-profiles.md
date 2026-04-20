# Run Profiles and Reproducible CLI Execution

One `molexp` script usually needs more than one execution shape. You may want a fast smoke run for local iteration, a conservative default for everyday work, and a heavier configuration for production or cluster submission. `molcfg` exists so those variants stay in data instead of leaking into ad-hoc flags or duplicated scripts.

The framework treats a profile as opaque user data. It loads a config file, resolves one named profile, injects the merged mapping into `ctx.config`, and records the chosen profile on the run. The framework does not assign special meaning to keys like `epochs`, `dataset`, or `skip_heavy_compute`; task code reads those fields explicitly.

## A Config File Makes Variants Explicit

`molcfg.yaml` is a small schema with `defaults` and `profiles`. Defaults are merged into every execution. A profile overlays those defaults and may inherit from another profile through `extends`.

```yaml
version: 1

defaults:
  dataset: qm9
  epochs: 100
  batch_size: 32
  optimizer:
    lr: 0.001

profiles:
  smoke:
    epochs: 3
    batch_size: 8

  dry-run:
    epochs: 1
    skip_heavy_compute: true

  large-batch:
    extends: smoke
    batch_size: 64
```

This arrangement keeps one script responsible for workflow structure while the config file owns execution shape. It also makes review easier: changing the experiment recipe becomes a diff in `molcfg.yaml`, not a rewrite of control flow.

## Task Code Reads the Active Profile Through `ctx.config`

Once a profile has been selected, every task sees the same read-only `ProfileConfig` through `TaskContext.config`. That object behaves like a mapping, so normal dictionary access patterns work.

```python
from molexp.workflow import Task, TaskContext


class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        epochs = ctx.config.get("epochs", 100)
        lr = ctx.config.get("optimizer", {}).get("lr", 1e-3)

        if ctx.config.get("skip_heavy_compute"):
            return {"mode": "lightweight", "epochs": 1, "lr": lr}

        return train_model(epochs=epochs, lr=lr)
```

The important design choice is that the task decides what those fields mean. `molexp` does not translate `dry-run` into special runtime behavior, and it does not reserve names for particular semantics. If your workflow wants `skip_heavy_compute`, you add it to the profile and read it yourself.

Even when no profile is selected, `ctx.config` still exists. In that case it is simply an empty defaults-only mapping, which is why `.get()` is the safest access pattern for optional fields.

## The CLI Selects, Refines, and Replays a Profile

The usual entry point is `molexp run SCRIPT`. If you do not pass `--config`, the command looks in the current working directory for `molcfg.yaml`, `molcfg.yml`, or `molcfg.json`. If no config file is found and no profile is requested, execution still works with an empty config. If you ask for `--profile` without a config file, the command aborts.

```bash
# Defaults only
molexp run train.py

# Explicit config file + named profile
molexp run train.py --config molcfg.yaml --profile smoke

# Use the implicit config in the current working directory
molexp run train.py --profile dry-run
```

Profile names normalize dashes to underscores when `molexp` stores them. That means `--profile dry-run` and a YAML key named `dry_run` resolve to the same stored profile name, `dry_run`.

Once a profile has been resolved, `--override` lets you patch individual values without editing the file. Overrides are applied after profile resolution, they accept `KEY=VALUE`, and dot notation works for nested mappings.

```bash
molexp run train.py --profile smoke --override optimizer.lr=0.0005
molexp run train.py --profile smoke --override epochs=5 --override batch_size=16
```

Values are coerced from strings into `bool`, `int`, `float`, or `str`, in that order. This makes one-off exploratory runs cheap while keeping the canonical configuration in versioned files.

`--resume` is scoped to the selected profile. A resumed run is eligible only when its persisted profile matches the one requested on the CLI and its status is not `succeeded`.

```bash
molexp run train.py --profile smoke --resume
```

That behavior is deliberate. Resume is meant to continue one execution stream, not to reinterpret an old run under a new profile.

## Run Metadata Preserves the Chosen Profile

Before execution begins, `molexp run` writes profile information into the run metadata. The run stores the normalized profile name, the fully merged config payload, and a deterministic content hash of that payload.

```json
{
  "profile": "smoke",
  "config": {
    "dataset": "qm9",
    "epochs": 3,
    "batch_size": 8,
    "optimizer": {"lr": 0.001}
  },
  "config_hash": "..."
}
```

This matters for two reasons. First, different profiles of the same experiment become distinct run identities instead of colliding in the same run directory. Second, replay and debugging stay grounded in real metadata: you can inspect `run.json`, recover the exact merged config, and understand which execution slice produced the artifacts on disk.

That same persisted metadata feeds other user-visible behaviors. The run monitor can distinguish profiles, replay tooling can reconstruct the chosen config, and task code running under `molexp execute RUN_DIR` sees the same `ctx.config` payload the original run used.

## One Script Can Cover Local Iteration and Cluster Submission

Profiles become more valuable when the same script moves across environments. A local smoke run and a cluster run usually differ in batch size, epoch count, or other task-level settings, but the workflow topology often stays identical. `molexp` lets the config file hold that distinction while the backend flags stay focused on transport.

```bash
# Quick local iteration
molexp run train.py --profile smoke

# Production-like cluster submission
molexp run train.py --profile production --scheduler slurm --partition gpu --gpus 1 --cpus 8
```

The workflow remains one source file, the profile remains one named config slice, and the scheduler flags remain a separate concern. That separation keeps the authoring model stable even as the execution target changes.

## The Next Layer of Detail

If you want the end-to-end authoring path, continue with the [Quick Start](../getting-started/quick-start.md). If you need the persistence details behind `profile`, `config`, and `config_hash`, see [Workflow Persistence](workflow-persistence.md). For the design rationale and the original replacement of `dry-run` with profiles, see [molcfg profiles](../development/specs/molcfg-profiles.md).

## Runnable Example

`examples/operations/run_profiles/` ships a `train.py` and a matching `molcfg.yaml` defining `smoke`, `dry-run`, and `large-batch`. Invoke it through the CLI, e.g. `molexp run examples/operations/run_profiles/train.py --profile smoke`.
