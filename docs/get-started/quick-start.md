# Quick Start

Let's create a complete MolExp workflow in five minutes, from defining tasks to execution, experiencing MolExp's complete workflow.

## Step 1: Define Your First Task

A Task is the most basic computation unit in MolExp. Each task needs to inherit from the `Task` base class, specify a configuration type, and implement the `execute` method.

What is a task? A task is a pure functional computation unit that accepts outputs from upstream tasks as inputs, performs computation, and produces outputs. Task configuration is determined at construction time, ensuring task behavior is predictable and serializable.

Why this design? Static configuration allows us to validate configuration correctness before execution and enables complete workflow serialization. This is crucial for reproducibility.

Let's create a simple task that receives a number and returns its square:

```python
from molexp.workflow.node import Task
from pydantic import BaseModel

class SquareConfig(BaseModel):
    """Configuration model defining task parameters"""
    pass  # This task doesn't need extra configuration

class SquareTask(Task[SquareConfig, int]):
    """Compute the square of an input number"""
    config_type = SquareConfig
    
    def execute(self, value: int) -> int:
        return value * value
```

## Step 2: Connect Tasks to Build a Workflow

Now let's create multiple tasks and connect them. Suppose we want to square a number, then add 10:

```python
class AddConfig(BaseModel):
    offset: int = 10

class AddTask(Task[AddConfig, int]):
    config_type = AddConfig
    
    def execute(self, value: int) -> int:
        return value + self.config.offset

# Create task instances and establish dependencies
square = SquareTask(task_id="square")
add = AddTask(square, task_id="add", offset=10)
```

Note that `AddTask`'s first parameter is `square`, indicating that `add` depends on `square`'s output. This declarative dependency relationship makes the workflow structure clear at a glance.

## Step 3: Register and Execute Tasks

To execute tasks in a workflow, you need to register them with the task registry. The registry allows the execution engine to find and instantiate your tasks:

```python
from molexp.ir.registry import registry

# Register tasks with the registry
@registry.register("square", SquareConfig)
class SquareTask(Task[SquareConfig, int]):
    config_type = SquareConfig
    
    def execute(self, value: int) -> int:
        return value * value

@registry.register("add", AddConfig)
class AddTask(Task[AddConfig, int]):
    config_type = AddConfig
    
    def execute(self, value: int) -> int:
        return value + self.config.offset
```

Once tasks are registered, you can execute them directly or use them in workflows. For simple cases, you can call tasks directly:

```python
# Execute tasks directly
square = SquareTask(task_id="square")
result = square(5)  # Returns 25

add = AddTask(square, task_id="add", offset=10)
final_result = add(5)  # Returns 35 (square(5) + 10)
```

For more complex workflows with multiple tasks and dependencies, you'll typically use the workspace API or load workflows from JSON files. The workflow compilation and execution happens automatically behind the scenes.

## Step 4: Use Workspace to Manage Experiments

In real scientific computing scenarios, we usually need to manage multiple experiments and runs. MolExp provides a complete workspace architecture:

```python
from molexp.workspace.core import Workspace
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo

# Create workspace
workspace = Workspace.from_path("./my_workspace")

# Create project
project = workspace.create_project(
    id="my_project",
    name="My Research Project",
)

# Create experiment
experiment = workspace.create_experiment(
    id="my_project",
    id="exp_1",
    name="First Experiment",
    workflow_source="workflow.py",
)

# Create run
run = workspace.create_run(
    id="my_project",
    id="exp_1",
    parameters={"input_value": 5},
    workflow_file="workflow.py",
)

# Use run context during execution
ctx = RunContext(
    asset_repo=AssetRepo(),
    id=run.id,
    run_metadata=run,
    workspace=workspace,
)

with use_run_context(ctx):
    # Execute your workflow here
    # Workflows are typically loaded from files or defined via the API
    # All registered assets will be automatically tracked
    pass
```

## Complete Example

Here's a complete runnable example showing the full flow from task definition to execution:

```python
from molexp.workflow.node import Task
from molexp.ir.registry import registry
from molexp.workspace.core import Workspace
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo
from pydantic import BaseModel

# Define configuration models
class SquareConfig(BaseModel):
    pass

class AddConfig(BaseModel):
    offset: int = 10

# Register and define tasks
@registry.register("square", SquareConfig)
class SquareTask(Task[SquareConfig, int]):
    config_type = SquareConfig
    
    def execute(self, value: int) -> int:
        return value * value

@registry.register("add", AddConfig)
class AddTask(Task[AddConfig, int]):
    config_type = AddConfig
    
    def execute(self, value: int) -> int:
        return value + self.config.offset

def main():
    # Create tasks and execute directly
    square = SquareTask(task_id="square")
    add = AddTask(square, task_id="add", offset=10)
    
    # Execute
    result = add(5)
    print(f"Result: {result}")  # (5^2) + 10 = 35
    
    # Or use with workspace for full tracking
    workspace = Workspace.from_path("./workspace")
    project = workspace.create_project("demo", name="Demo")
    experiment = workspace.create_experiment(
        "demo", "exp_1", "Experiment 1", "workflow.py"
    )
    run = workspace.create_run("demo", "exp_1", {}, "workflow.py")
    
    ctx = RunContext(
        asset_repo=AssetRepo(),
        id=run.id,
        run_metadata=run,
        workspace=workspace,
    )
    
    with use_run_context(ctx):
        result = add(5)
        print(f"Result with context: {result}")

if __name__ == "__main__":
    main()
```

For advanced workflow definitions using IR models and compilation, see the [Developer Documentation](developer/ir-and-compiler.md).

Congratulations! You've completed your first MolExp workflow. In the following chapters, we'll explore the design details and advanced usage of each component.
