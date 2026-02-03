# Task Protocol (runtime-only)

MolExp still defines a structural `TaskProtocol` interface, but **workflow persistence and replay require Pydantic-based `Task` classes**. Non-Task protocol objects (including `molpy.Compute` classes) are **not supported for registration or persisted workflows**.

## What This Means

- You may call protocol-compatible classes directly in Python (in-process use).
- You must use `molexp.workflow.Task` with a Pydantic `config_type` for any workflow that is serialized and replayed.
- The registry only accepts `Task` classes with Pydantic configs.

## Recommended Path for molpy.Compute

If you need to include a `molpy.Compute` in a workflow, wrap it in a `Task` with a Pydantic config and explicit inputs/outputs.

Example (wrapper pattern):

```python
from pydantic import BaseModel
from molexp.workflow import Task
from molpy.compute import MCDCompute

class MCDConfig(BaseModel):
    tags: list[str]
    max_dt: float
    dt: float

class MCDTask(Task[MCDConfig, dict]):
    config_type = MCDConfig
    inputs = {"input": object}
    outputs = {"result": object}

    def execute(self, ctx=None, **inputs):
        compute = MCDCompute(**self.config.model_dump())
        return compute.execute(**inputs)
```

This keeps workflows replayable and deterministic.
