# molexp

molexp is a workflow-and-agent platform for research experiment management. It combines a typed task-graph framework, a file-system-backed `Workspace → Project → Experiment → Run` hierarchy, a `molcfg` profile system for repeatable execution variants, and a FastAPI server that can serve a bundled React UI.

```
Workflow authoring → WorkflowSpec → Graph runtime → Workspace → FastAPI → React UI
        ↑                                ↑
   Task / Actor API                molcfg profiles
```

## Platform Capabilities

- **Typed workflow authoring** with a functional DSL (`@wf.task`, `@wf.actor`) and an OOP builder (`WorkflowBuilder`).
- **Deterministic execution** backed by `pydantic-graph`, including same-level task parallelism and persisted run lifecycles.
- **Tracked experiment state** through a `Workspace → Project → Experiment → Run` hierarchy with scoped asset libraries and atomic metadata writes.
- **Repeatable execution variants** through `molcfg.yaml` profiles, exposed to user code as `ctx.config` and persisted onto every run.
- **An optional agent layer** built on PydanticAI, loaded lazily as `molexp.plugins.agent_pydanticai`.
- **A server and web UI** for browsing workspace state, runs, assets, and execution metadata.

## Core Concepts

At a high level, `molexp` has three separate but connected models:

- **Workflow concepts** describe computation. A `Task` runs once and returns one value. An `Actor` is the streaming variant. A `WorkflowSpec` is the compiled DAG: it knows task names, dependency edges, and the deterministic `workflow_id`, but it is still independent from any particular run on disk.
- **Workspace concepts** describe experiment state. A `Workspace` is the root container. A `Project` groups related work. An `Experiment` is the repeatable definition of a workflow plus parameters and replica settings. A `Run` is one concrete execution attempt, with status, artifacts, errors, execution history, and persisted config metadata.
- **Plugin concepts** describe optional execution capabilities. Core local workflow execution has no heavy optional dependency. The `submit_molq` plugin is the scheduler bridge used by `molexp run --slurm`, `--pbs`, `--lsf`, and `--scheduler <name>` when `molq` is installed.

Those models deliberately stay separate:

- The workflow says what should run.
- The workspace says where that execution is recorded.
- The plugin says how execution is transported to a remote scheduler when local execution is not enough.

## Define a Workflow

The smallest useful workflow is just a few typed tasks:

```python
from molexp.workflow import TaskContext, workflow

wf = workflow(name="demo")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def reduce(ctx: TaskContext) -> float:
    scale = ctx.config.get("scale", 1.0)
    return sum(ctx.inputs) * scale


spec = wf.build()
```

The same graph can also be authored with reusable task classes:

```python
from molexp.workflow import Task, TaskContext, WorkflowBuilder


class Fetch(Task):
    async def execute(self, ctx: TaskContext) -> list[float]:
        return [1.0, 4.0, 9.0]


class Reduce(Task):
    async def execute(self, ctx: TaskContext) -> float:
        scale = ctx.config.get("scale", 1.0)
        return sum(ctx.inputs) * scale


spec = (
    WorkflowBuilder(name="demo")
    .add(Fetch())
    .add(Reduce(), depends_on=["fetch"])
    .build()
)
```

Any third-party object with `async def execute(self, ctx)` also works. The public protocol is structural, not inheritance-based.

## Bind It to a Workspace

Workflows become reproducible experiment runs when you bind them to a workspace entity graph:

```python
import molexp as me

ws = me.Workspace("./lab")
project = ws.project("QM9")
exp = project.experiment(
    "baseline",
    params={"lr": 1e-3},
    n_replicas=3,
    workflow_source="train.py",
)
exp.set_workflow(spec)

# Register the workspace so `molexp run train.py` can discover it.
me.entry(ws)
```

All child factories are idempotent. Recreating the same project, experiment, or run returns the existing on-disk object instead of duplicating state.

The distinction between the four workspace levels is important:

- `Workspace` is the lab root on disk.
- `Project` groups related experiments, such as one paper track or one model family.
- `Experiment` captures one repeatable workflow definition plus its parameter set and replica policy.
- `Run` captures one realized execution of that experiment, including profile metadata, artifacts, logs, error info, and execution history.

That split is what lets one workflow definition be executed many times without losing the difference between definition and outcome.

## Control Run Variants with `molcfg`

`molexp` keeps profile semantics deliberately simple: the framework loads a named config slice, stores it on the run, and injects it into task code as `ctx.config`. The meaning of each field is entirely up to your workflow.

```yaml
# molcfg.yaml
version: 1

defaults:
  lr: 0.001
  epochs: 100
  batch_size: 32

profiles:
  smoke:
    epochs: 3
    batch_size: 8

  dry-run:
    epochs: 1
    skip_heavy_compute: true
```

From there, one script can be executed in several reproducible modes:

```bash
# Defaults only
molexp run train.py

# Explicit config file + named profile
molexp run train.py --config molcfg.yaml --profile smoke

# Apply one-off overrides after profile resolution
molexp run train.py --profile smoke --override model.n_layers=3

# Re-execute failed or cancelled runs for one profile only
molexp run train.py --profile smoke --resume
```

Some details matter in practice:

- `--config` accepts YAML or JSON.
- When `--config` is omitted, `molexp run` searches the current working directory for `molcfg.yaml`, `molcfg.yml`, or `molcfg.json`.
- Asking for `--profile` without a config file is an error.
- Profile names normalize `-` to `_`, so `dry-run` becomes `dry_run` in persisted metadata.
- Different resolved profile payloads produce different run identities because the profile name and config hash are stored on the run before execution.

## CLI Surface

```text
molexp run SCRIPT           # Execute workflows registered via me.entry()
molexp execute RUN_DIR      # Worker entry used by cluster backends
molexp serve                # API server + bundled SPA when available
molexp init [PATH]          # Initialize a workspace
molexp info                 # Show workspace status
molexp explore              # Open the interactive workspace explorer

molexp project   create|list|info
molexp experiment create|list
molexp runs      create|list|info|cancel|prune
molexp asset     list
```

Execution backend selection lives on `molexp run`:

```bash
# Local foreground execution
molexp run train.py

# Local background execution; logs land in <workspace>/molexp_bg.log
molexp run train.py --bg

# Cluster submission through molq
molexp run train.py --slurm --partition gpu --cpus 8 --mem 32G --gpus 1
molexp run train.py --pbs   --queue batch --time 12:00:00
molexp run train.py --lsf   --queue batch --cpus 16

# Block after submission and keep the run monitor open
molexp run train.py --slurm --partition gpu --block
```

## The `molq` Scheduler Plugin

Remote scheduler submission is intentionally optional. Local workflows and workspace browsing do not require `molq`, but cluster execution does.

When you run a cluster command such as:

```bash
molexp run train.py --slurm --partition gpu --gpus 1
```

`molexp` loads `molexp.plugins.submit_molq` on demand. That plugin does three things:

- It validates that `molq` is installed and that the requested scheduler backend exists.
- It converts `molexp` CLI options like `--cpus`, `--mem`, `--time`, `--queue`, `--account`, and `--qos` into `molq` submission objects.
- It submits a worker command of the form `python -m molexp.cli execute <run_dir>` and persists normalized executor metadata such as scheduler name, cluster name, and scheduler job ID into the run metadata.

The plugin does not change workflow semantics. Your workflow still runs through the same `WorkflowSpec` and the same `RunContext`; the plugin only changes where that worker process is launched.

## Installation and Development

```bash
# Python-only editable install
pip install -e .

# Optional extras
pip install -e ".[agent]"
pip install -e ".[all]"
pip install -e ".[dev]"
```

For day-to-day development, the backend and frontend stay on separate loops:

```bash
# Backend
molexp serve ./lab --port 8000

# Frontend dev server
npm run dev
```

If `src/molexp/_webapp/` is empty, `molexp serve` runs API-only and points you at the dev server. To ship the UI inside the Python package, build it ahead of time:

```bash
npm run build:ui
python -m build --wheel
```

The wheel includes `src/molexp/_webapp/**`, and `create_app()` auto-detects that bundle via `importlib.resources`.

## Documentation Map

- **[Documentation Index](docs/index.md)** — vision, scope, design principles.
- **[Concepts](docs/concept/index.md)** — workflow / workspace / plugin mental model.
- **[Quick Start](docs/getting-started/quick-start.md)** — end-to-end from task definition to tracked execution.
- **[Guide](docs/guide/index.md)** — topical usage guides (task/actor, runtime, profiles, workspace, server, molq).
- **[Development](docs/development/index.md)** — compiler internals, task protocols, active design specs.

## License

BSD 3-Clause License. See [LICENSE](./LICENSE).
