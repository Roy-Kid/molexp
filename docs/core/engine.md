# Execution Engine

The execution engine is the runtime component that executes compiled workflows. It handles parallel execution, failure propagation, timeouts, and provides hooks for monitoring workflow execution.

## What the Engine Is

The `WorkflowEngine` is responsible for executing workflow IR that has been compiled and validated. It manages task execution order, handles parallel execution of independent tasks, propagates failures to dependent tasks, and provides execution hooks for monitoring and logging.

The engine uses a thread pool executor to run tasks in parallel, automatically identifying which tasks can execute concurrently based on their dependencies. When a task fails, the engine automatically cancels all dependent tasks, preventing unnecessary computation.

## Why This Design

The engine's design addresses several critical requirements for scientific computing workflows. First, parallel execution significantly improves performance when dealing with independent tasks. The engine automatically identifies parallelizable tasks without requiring manual scheduling.

Second, intelligent failure propagation ensures that when a task fails, all tasks that depend on it are automatically cancelled. This prevents wasted computation and provides clear error reporting. The engine tracks execution status for each task, allowing you to see exactly which tasks succeeded, failed, or were cancelled.

Third, execution hooks provide a flexible mechanism for monitoring, logging, and custom behavior during workflow execution. You can attach callbacks to track task start, success, failure, and workflow completion events.

Finally, timeout support prevents tasks from hanging indefinitely, which is crucial for production environments where resource limits must be enforced.

## How to Use

### Basic Execution

The simplest way to execute a workflow is to create an engine and call its `execute` method. In most cases, workflows are loaded from files or created through the workspace API, and the engine handles execution automatically.

For direct execution with workflow IR (Intermediate Representation), see the [Developer Documentation](developer/ir-and-compiler.md) for details on building and compiling workflows.

```python
from molexp.ir.engine import WorkflowEngine

# Create engine with default settings
engine = WorkflowEngine()

# Execute workflow (workflow IR is typically loaded from file or API)
# status = engine.execute(workflow=compiled_workflow_ir, id="run_001")
```

The `execute` method returns a dictionary mapping task IDs to their final execution status (`SUCCEEDED`, `FAILED`, `CANCELLED`, etc.).

### Configuring Parallel Execution

You can configure the maximum number of concurrent workers:

```python
# Allow up to 8 parallel tasks
engine = WorkflowEngine(max_workers=8)

status = engine.execute(compiled_workflow_ir, id="run_002")
```

The engine will automatically identify independent tasks and execute them in parallel up to the `max_workers` limit.

### Setting Timeouts

You can set a timeout for each task to prevent hanging:

```python
# Each task times out after 60 seconds
engine = WorkflowEngine(
    max_workers=4,
    node_timeout=60.0,
)

status = engine.execute(compiled_workflow_ir, id="run_003")
```

If a task exceeds the timeout, it will be marked as failed and the failure will propagate to dependent tasks.

### Using Execution Hooks

Execution hooks allow you to monitor workflow execution in real-time:

```python
from molexp.ir.engine import WorkflowEngine, ExecutionHooks

def on_node_start(id: str, node):
    print(f"[{id}] Starting task: {node.id}")

def on_node_success(id: str, node, result):
    print(f"[{id}] Task {node.id} succeeded with result: {result}")

def on_node_failure(id: str, node, error):
    print(f"[{id}] Task {node.id} failed: {error}")

def on_workflow_complete(id: str, status: dict):
    print(f"[{id}] Workflow complete. Status: {status}")

# Create hooks
hooks = ExecutionHooks(
    on_node_start=on_node_start,
    on_node_success=on_node_success,
    on_node_failure=on_node_failure,
    on_workflow_complete=on_workflow_complete,
)

# Create engine with hooks
engine = WorkflowEngine(hooks=hooks)

status = engine.execute(compiled_workflow_ir, id="run_004")
```

Hooks are called at various points during execution, allowing you to implement custom logging, progress tracking, or integration with monitoring systems.

### Partial Execution

You can execute only a subset of tasks by specifying target node IDs:

```python
# Only execute specific tasks
status = engine.execute(
    workflow=compiled_workflow_ir,
    id="run_005",
    node_ids=["task_1", "task_2"],  # Only execute these tasks
)
```

The engine will automatically identify and execute all dependencies of the specified tasks.

### Handling Execution Results

## Generator-Based Submission

Tasks submit external work by implementing `submit()` and setting `config.submit`:

```python
class MyConfig(BaseModel):
    submit: str | None = None

def execute(self, ctx=None):
    return {"status": "local"}

def submit(self, ctx=None):
    job_id = yield {
        "execution": {"cmd": ["echo", "hello"], "block": False},
        "resources": {},
        "cluster": {},
    }
    return {"job_id": job_id}
```

When `config.submit="molq"`, the engine drives `submit()` and uses the registered
submitor backend. If no submitor is configured for the backend, execution fails
with a clear error.

The engine stores execution results in the execution context. You can access them through hooks or by examining the execution context:

```python
from molexp.ir.engine import WorkflowEngine, ExecutionContext

results = {}

def on_node_success(id: str, node, result):
    results[node.id] = result

hooks = ExecutionHooks(on_node_success=on_node_success)
engine = WorkflowEngine(hooks=hooks)

status = engine.execute(compiled_workflow_ir, id="run_006")

# Access results
print(f"Task results: {results}")
print(f"Final status: {status}")
```

### Complete Example

Here's a complete example showing engine usage with monitoring hooks:

```python
from molexp.ir.engine import WorkflowEngine, ExecutionHooks
import time

def main():
    # Set up monitoring hooks
    hooks = ExecutionHooks(
        on_node_start=lambda id, node: print(f"Starting {node.id}"),
        on_node_success=lambda id, node, result: print(f"{node.id} succeeded"),
        on_node_failure=lambda id, node, error: print(f"{node.id} failed: {error}"),
        on_workflow_complete=lambda id, status: print(f"Complete: {status}"),
    )
    
    # Create and configure engine
    engine = WorkflowEngine(
        max_workers=4,
        node_timeout=30.0,
        hooks=hooks,
    )
    
    # Execute workflow (typically loaded from file or workspace)
    # For building workflows from IR, see Developer Documentation
    # start_time = time.time()
    # status = engine.execute(workflow_ir, id="example_run")
    # elapsed = time.time() - start_time
    # print(f"\nExecution completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
```

For examples of building and compiling workflows from IR models, see the [Developer Documentation](developer/ir-and-compiler.md).

The execution engine provides a robust foundation for running workflows with parallel execution, failure handling, and monitoring capabilities. Combined with the compiler's static validation, it ensures reliable and efficient workflow execution.
