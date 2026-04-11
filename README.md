# molexp

molexp is a workflow-and-agent platform for research experiment management. It provides a typed task-graph framework, a Project-Experiment-Run workspace hierarchy, content-addressed asset storage, and a FastAPI server with React UI.

```
WorkflowSpec → Runtime → Workspace → FastAPI → React UI
                             ↑
                      AgentService (PydanticAI)
```

## Features

- **Workflow Layer**: DAG-based task graphs with automatic parallelization
- **Project-Experiment-Run Architecture**: Scientific workflow organization with full reproducibility
- **Asset Management**: Content-addressable storage with automatic deduplication
- **Agent Layer**: Goal-driven autonomous execution built on PydanticAI
- **CLI + Web UI**: Command-line interface and React-based experiment browser

## Quick Example: Workflow

**Functional DSL (decorator-based):**

```python
from molexp.workflow import workflow, TaskContext

wf = workflow(name="data-pipeline")

@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 2.0, 3.0]

@wf.task(depends_on=["fetch"])
async def process(ctx: TaskContext) -> float:
    if ctx.dry_run:
        return 0.0
    return sum(ctx.inputs)

spec = wf.build()
result = await spec.execute()
```

**OOP builder (subclass-based):**

```python
from molexp.workflow import Task, WorkflowBuilder, TaskContext

class FetchTask(Task):
    async def execute(self, ctx: TaskContext) -> list[float]:
        return [1.0, 2.0, 3.0]

class ProcessTask(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return sum(ctx.inputs)

spec = (
    WorkflowBuilder(name="data-pipeline")
    .add(FetchTask())
    .add(ProcessTask(), depends_on=["fetch"])
    .build()
)
result = await spec.execute()
```

## Project-Experiment-Run Architecture

molexp provides a complete organization system for scientific workflows:

- **Project**: Top-level container for a research area
- **Experiment**: Repeatable workflow with parameter space definition
- **Run**: Single execution instance with full reproducibility
- **Asset**: Reusable data artifacts with content-based deduplication

### Python API

```python
from molexp.workspace import Workspace

workspace = Workspace.from_path("./lab")

# Hierarchical API: workspace → project → experiment → run
project = workspace.create_project(name="My Project")
experiment = project.create_experiment(name="Param Sweep")
run = experiment.create_run(parameters={"lr": 0.01})

# Scoped asset libraries at every level
workspace.assets.create_asset("bert_model", "/models/bert.pt")
project.assets.create_asset("dataset", "/data/qm9.tar.bz2")
experiment.assets.create_asset("features", "/data/features.h5")
run.assets.create_asset("output", "/outputs/results.txt")
```

### CLI Usage

```bash
# Start server (hot-reload dev mode)
molexp serve --dev

# Execute a workflow in dry-run mode
molexp run train.py --dry-run

# Initialize a workspace
molexp init [path]
```

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[workflow]"   # pydantic-graph execution backend
pip install -e ".[agent]"      # PydanticAI agent layer
pip install -e ".[remote]"     # Remote/HPC execution
pip install -e ".[dev]"        # Development tools
```

## Documentation

See the [docs](./docs/index.md) for an in-depth tour of the architecture, workflow layer, workspace, and agent integration.

## License

BSD 3-Clause License — see [LICENSE](./LICENSE) for details.
