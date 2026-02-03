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

# Create workspace
workspace = Workspace.from_env()

# Create project (ID auto-generated from name)
project = workspace.create_project(
    name="My Project",
    description="Research project description"
)

# Create experiment through project (hierarchical API)
experiment = project.create_experiment(
    name="Experiment 1",
    workflow_source="workflow.py"
)

# Create run through experiment (hierarchical API)
run = experiment.create_run(
    parameters={"param": 1.0},
    workflow_file="workflow.py"
)

# Use hierarchical asset libraries
# Workspace-level assets (global)
workspace.assets.create_asset("bert_model", "/models/bert.pt")

# Project-level assets (shared within project)
project.assets.create_asset("dataset", "/data/qm9.tar.bz2")

# Experiment-level assets (shared within experiment)
experiment.assets.create_asset("features", "/data/features.h5")

# Run-level assets (specific to this run)
run.assets.create_asset("output", "/outputs/results.txt")
```

## Installation

```bash
pip install -e .
```

## Documentation

See the [docs](./docs/README.md) for an in-depth tour of the architecture, compiler, engine, and DSL
usage.

See [examples/project_experiment_run_example.py](./examples/project_experiment_run_example.py) for a complete example of the Project-Experiment-Run workflow.
