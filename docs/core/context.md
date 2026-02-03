# Context Management

Context management in MolExp provides a way to share runtime information across tasks without explicitly passing it through function parameters. This enables features like asset tracking, run metadata access, and workspace integration.

## What Context Is

`RunContext` is a dataclass that holds runtime information for a workflow execution. It includes the asset repository for tracking produced artifacts, run metadata for identifying the current execution, workspace reference for persistence operations, and an extensible `extras` dictionary for custom data.

Context is propagated implicitly using Python's `contextvars`, which means tasks can access the current context without it being passed as a parameter. This design keeps task signatures clean while enabling powerful features like automatic asset registration.

## Why This Design

Using context variables instead of explicit parameters has several advantages. First, it keeps task interfaces clean—tasks don't need to accept context parameters just to register assets or access metadata. This makes tasks more reusable and easier to test.

Second, context variables are thread-safe and automatically propagate through async boundaries, making them ideal for parallel execution scenarios. Each execution thread gets its own context, preventing cross-contamination between parallel tasks.

Third, the context manager pattern (`use_run_context`) provides explicit control over context scope, making it clear when context is available and ensuring proper cleanup. This is especially important when integrating with workspace persistence.

Finally, the design allows for graceful degradation: if no context is set, tasks can still execute (though features like asset registration won't work). This makes MolExp flexible for both standalone execution and workspace-integrated scenarios.

## How to Use

### Setting Up Context

To use context in your workflow execution, create a `RunContext` and use the context manager:

```python
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo
from molexp.workspace.core import Workspace

# Create workspace and run (if using workspace)
workspace = Workspace.from_path("./workspace")
run = workspace.create_run(...)

# Create context
ctx = RunContext(
    asset_repo=AssetRepo(),
    id=run.id,
    run_metadata=run,
    workspace=workspace,
)

# Use context during execution
with use_run_context(ctx):
    # Execute your workflow here
    # All tasks can now access the context
    engine.execute(compiled_workflow, id=run.id)
```

### Accessing Context in Tasks

Tasks can access the current context using `get_current_context()` or `require_current_context()`:

```python
from molexp.workflow.context import get_current_context, require_current_context
from molexp.assets import register_asset

class ProcessTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, data: list) -> str:
        # Get context (returns None if not set)
        ctx = get_current_context()
        if ctx:
            print(f"Running in context: {ctx.id}")
        
        # Process data
        result = process(data)
        
        # Register asset (requires context)
        output_path = "result.txt"
        with open(output_path, "w") as f:
            f.write(result)
        
        register_asset(output_path, label="processed_data")
        
        return output_path
```

`get_current_context()` returns `None` if no context is set, while `require_current_context()` raises an exception. Use the latter when context is required for the task to function correctly.

### Registering Assets

The most common use of context is automatic asset registration. When you call `register_asset()`, it automatically uses the current context:

```python
from molexp.assets import register_asset

class SaveTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, data: list) -> str:
        output_path = "output.txt"
        
        # Save data
        with open(output_path, "w") as f:
            f.write("\n".join(map(str, data)))
        
        # Register asset - automatically uses current context
        register_asset(
            output_path,
            label="output_data",
            meta={"row_count": len(data)},
        )
        
        return output_path
```

If context is available with workspace integration, `register_asset()` will:
1. Compute content hash for deduplication
2. Store asset in global repository
3. Create asset reference linking to the run
4. Update run's asset references

### Accessing Run Metadata

Tasks can access run metadata through the context:

```python
from molexp.workflow.context import get_current_context

class LogTask(Task[BaseModel, None]):
    config_type = BaseModel
    
    def execute(self, data: any) -> None:
        ctx = get_current_context()
        if ctx and ctx.run_metadata:
            print(f"Run ID: {ctx.run_metadata.id}")
            print(f"Project: {ctx.run_metadata.id}")
            print(f"Experiment: {ctx.run_metadata.id}")
            print(f"Parameters: {ctx.run_metadata.parameters}")
        
        # Process data...
        return None
```

### Using Extras for Custom Data

The `extras` dictionary allows you to pass custom data through context:

```python
# Set up context with custom data
ctx = RunContext(
    asset_repo=AssetRepo(),
    id="run_001",
    extras={
        "user_id": "alice",
        "experiment_config": {"temperature": 300, "pressure": 1.0},
    },
)

with use_run_context(ctx):
    # Tasks can access extras
    task_context = get_current_context()
    user_id = task_context.extras.get("user_id")
    config = task_context.extras.get("experiment_config")
```

### Standalone Execution Without Context

Tasks can execute without context, though some features won't be available:

```python
# Execute without context
engine = WorkflowEngine()
status = engine.execute(compiled_workflow, id="standalone_run")

# Tasks will execute, but:
# - register_asset() will only work in-memory
# - get_current_context() will return None
# - Workspace integration won't be available
```

This flexibility allows MolExp to work in both standalone and workspace-integrated scenarios.

### Complete Example

Here's a complete example showing context setup and usage:

```python
from molexp.workspace.core import Workspace
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo, register_asset
from molexp.ir.engine import WorkflowEngine
# For workflow compilation, see Developer Documentation

def main():
    # Set up workspace
    workspace = Workspace.from_path("./workspace")
    
    # Create project, experiment, and run
    project = workspace.create_project("my_project", name="My Project")
    experiment = workspace.create_experiment(
        id="my_project",
        id="exp_1",
        name="Experiment 1",
        workflow_source="workflow.py",
    )
    run = workspace.create_run(
        id="my_project",
        id="exp_1",
        parameters={"input_file": "data.txt"},
        workflow_file="workflow.py",
    )
    
    # Create context
    ctx = RunContext(
        asset_repo=AssetRepo(),
        id=run.id,
        run_metadata=run,
        workspace=workspace,
        extras={"debug": True},
    )
    
    # Compile workflow
    compiled = WorkflowCompiler().compile(workflow_ir)
    
    # Execute with context
    with use_run_context(ctx):
        engine = WorkflowEngine()
        status = engine.execute(compiled, id=run.id)
        
        # Assets registered during execution are automatically tracked
        print(f"Execution status: {status}")
        
        # Retrieve asset references
        asset_refs = workspace.get_asset_refs(
            project.id,
            experiment.id,
            run.id,
        )
        print(f"Produced assets: {len(asset_refs.outputs) if asset_refs else 0}")

if __name__ == "__main__":
    main()
```

Context management provides a clean and powerful way to share runtime information across tasks, enabling features like asset tracking and workspace integration while keeping task interfaces simple and reusable.

