# molexp

molexp is a tiny yet fully-typed task-graph framework built on top of Pydantic. It contains a
pure functional task abstraction, a static compiler that produces deterministic graph orders, a
runtime engine, and a tiny DSL for common data-flow patterns. This repository is intentionally
minimal to highlight how each layer works without hidden magic.

```
+-----------+       +-----------+       +---------+
|   Task    |  -->  | Compiler  |  -->  | Engine  |
+-----------+       +-----------+       +---------+
        ^                  |                  |
        |                  v                  v
        +----------- DSL abstractions --------+
```

## Features

- **Task Graph Framework**: Pure functional task abstraction with deterministic compilation
- **Project-Experiment-Run Architecture**: Scientific workflow organization with full reproducibility
- **Asset Management**: Content-addressable storage with automatic deduplication
- **CLI Tools**: Command-line interface for workspace and workflow management
- **Type Safety**: Full Pydantic v2 integration for data validation

## Quick Example: Task Graph

```python
from molexp.task_base import Task, EmptyConfig
from molexp.engine import TaskEngine

class MultiplyTask(Task[EmptyConfig, int]):
    cfg_model = EmptyConfig
    out_model = None

    def forward(self, value: int, cfg: EmptyConfig) -> int:
        return value * 2

mult = MultiplyTask(name="multiply")
engine = TaskEngine()
result = engine.run(mult)
print(result)
```

## Project-Experiment-Run Architecture

molexp provides a complete organization system for scientific workflows:

- **Project**: Top-level container for a research area
- **Experiment**: Repeatable workflow with parameter space definition
- **Run**: Single execution instance with full reproducibility
- **Asset**: Reusable data artifacts with content-based deduplication

### CLI Usage

```bash
# Initialize workspace
molexp init

# Create project
molexp project create my-project --name "My Research Project"

# Create experiment
molexp experiment create my-project exp-1 \
  --name "Parameter Sweep" \
  --workflow workflow.py

# List runs
molexp run list my-project exp-1

# View assets
molexp asset list
```

### Python API

```python
from molexp.workspace import Workspace
from molexp.context import RunContext, use_run_context
from molexp.assets import AssetRepo, register_asset

# Create workspace
workspace = Workspace.from_env()

# Create project and experiment
project = workspace.create_project("my-project", name="My Project")
experiment = workspace.create_experiment(
    project_id="my-project",
    experiment_id="exp-1",
    name="Experiment 1",
    workflow_source="workflow.py",
)

# Create run
run = workspace.create_run(
    project_id="my-project",
    experiment_id="exp-1",
    parameters={"param": 1.0},
    workflow_file="workflow.py",
)

# Execute with asset tracking
ctx = RunContext(
    asset_repo=AssetRepo(),
    run_metadata=run,
    workspace=workspace,
)

with use_run_context(ctx):
    # Your workflow code here
    # Assets registered with register_asset() are automatically tracked
    register_asset("output.txt", label="results")
```

## Installation

```bash
pip install -e .
```

## Documentation

See the [docs](./docs/README.md) for an in-depth tour of the architecture, compiler, engine, and DSL
usage.

See [examples/project_experiment_run_example.py](./examples/project_experiment_run_example.py) for a complete example of the Project-Experiment-Run workflow.
