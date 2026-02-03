# Workflow Serialization Guide

## Overview

MolExp workflows are **persistable and replayable**. To guarantee this, workflows:

- Use **Pydantic-only** task configuration.
- Require **explicit link mappings** (no implicit name matching).
- Use **deterministic task type IDs** (`module.path.ClassName`).
- Require explicit **task registration** before serialization.

## Quick Start

### 1. Define and Register Tasks

```python
from pydantic import BaseModel
from molexp.workflow import Task, register_task

class MyTaskConfig(BaseModel):
    param1: str
    param2: int

class MyTask(Task[MyTaskConfig, dict]):
    config_type = MyTaskConfig
    inputs = {"input_value": int}
    outputs = {"result": int}

    def execute(self, ctx=None, **inputs):
        return {"result": inputs["input_value"] * self.config.param2}

# Explicit registration is required for persistence
register_task(MyTask)
```

### 2. Create a Workflow With Explicit Mappings

```python
from molexp.workflow import Workflow, Link

# Create tasks
first = MyTask(param1="hello", param2=10)
second = MyTask(param1="world", param2=20)

# Explicit output -> input mapping
link = Link(
    source=first.task_id,
    target=second.task_id,
    mapping={"result": "input_value"},
)

workflow = Workflow.from_tasks(
    tasks=[first, second],
    links=[link],
    name="my_workflow",
)
```

### 3. Save and Load

```python
from pathlib import Path

workflow.save(Path("workflow.json"))
loaded = Workflow.load(Path("workflow.json"))
```

### 4. Execute

```python
from molexp.workflow import WorkflowEngine
from molexp.workspace import Workspace

workspace = Workspace(root=Path("workspace"), name="Demo")
workspace.materialize()
project = workspace.create_project(name="Project")
experiment = project.create_experiment(name="Experiment")
run = experiment.create_run(parameters={})

with run.context() as ctx:
    engine = WorkflowEngine(loaded)
    results = engine.execute(ctx, input_value=5)
```

## TaskConfig Format

```python
class TaskConfig(BaseModel):
    task_id: str
    task_type: str  # module.path.ClassName
    config: dict
```

The `task_type` is a deterministic registry ID (e.g., `molexp.workflow.nodes.MyTask`).

## Registration and Replay

- **Registration is required**: the engine resolves task classes from `task_type`.
- **Missing registration is an error** during load or execution.

## Notes

- Control-flow tasks that carry runtime callables or nested tasks are **not replayable** and will be rejected when building a persisted workflow.
- All link mappings must be explicit; missing or invalid mappings are compile-time errors.
