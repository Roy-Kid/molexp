# MolExp Overview

Before diving deep into MolExp's components, let's understand the system's design philosophy and overall architecture.

## What MolExp Is

MolExp is a task-graph framework. Its core idea is to decompose complex computational workflows into a series of interconnected task nodes. Each task is a pure functional computation unit that accepts inputs and produces outputs. Tasks are connected through links, forming a directed acyclic graph (DAG).

Unlike traditional script-based workflows, MolExp provides a separation between static compilation and execution. This means you can validate workflow correctness before execution, generate deterministic execution plans, and clearly track inputs and outputs of each task.

## Why This Design

Scientific computing workflows often have these characteristics: complex dependencies between tasks, need for parallel execution to improve efficiency, intelligent failure propagation and recovery, and results that need to be reproducible and traceable. Traditional scripting approaches struggle to meet these requirements, and MolExp's design addresses these problems.

First, static compilation allows us to discover potential issues before execution, such as circular dependencies, missing inputs, etc. This is much more efficient than discovering errors at runtime. Second, deterministic execution order ensures result reproducibility, which is crucial for scientific computing. Finally, explicit dependency relationships make parallel execution natural and safe, as the engine can automatically identify which tasks can execute in parallel.

Additionally, MolExp uses protocol-based design rather than requiring inheritance. This means components from other frameworks (like MolNex's `DataNode` and `DataPipeline`) can be used directly in MolExp workflows without modification, as long as they implement the required interface. This enables seamless interoperability and allows you to leverage existing code without rewriting it.

## How to Use It

Let's understand MolExp's basic usage through a simple example. Suppose we want to process a batch of data: load data, transform it, then save the results.

```python
from molexp.workflow.node import Task
from molexp.ir.models import WorkflowIR, Workflow, Task as IRTask, Link
from pydantic import BaseModel

# Define configuration models
class LoadConfig(BaseModel):
    file_path: str

class TransformConfig(BaseModel):
    scale: float = 1.0

# Define tasks
class LoadTask(Task[LoadConfig, list]):
    config_type = LoadConfig
    
    def execute(self, *inputs):
        # Load data
        with open(self.config.file_path) as f:
            data = [float(line.strip()) for line in f]
        return data

class TransformTask(Task[TransformConfig, list]):
    config_type = TransformConfig
    
    def execute(self, data: list):
        # Transform data
        return [x * self.config.scale for x in data]

class SaveTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, data: list):
        # Save results
        output_path = "output.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(map(str, data)))
        return output_path

# Create task instances
load = LoadTask(task_id="load", file_path="input.txt")
transform = TransformTask(load, task_id="transform", scale=2.0)
save = SaveTask(transform, task_id="save")

# Tasks can be executed directly or used in workflows
load = LoadTask(task_id="load", file_path="input.txt")
transform = TransformTask(load, task_id="transform", scale=2.0)
save = SaveTask(transform, task_id="save")

# Execute the workflow
result = save()  # Executes load -> transform -> save
```

This example demonstrates MolExp's core concepts: task definition, configuration models, dependency relationships, and workflow execution. In the following chapters, we'll explore the design and usage of each component in detail.

For advanced workflow definitions using IR (Intermediate Representation) models, see the [Developer Documentation](developer/ir-and-compiler.md).
