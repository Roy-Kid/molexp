# Tasks

Tasks are the fundamental building blocks of MolExp workflows. This guide covers the different types of tasks and how to use them effectively.

## Task Types

### HamiltonTask

The `HamiltonTask` is designed for computational workflows using the Hamilton framework:

```python
from molexp import HamiltonTask

task = HamiltonTask(
    name="my_calculation",
    func=my_function,
    inputs={"param1": "value1"},
    outputs=["result"]
)
```

### ShellTask

For executing shell commands:

```python
from molexp import ShellTask

task = ShellTask(
    name="run_analysis",
    command="python analysis.py --input {input_file}",
    inputs={"input_file": "data.txt"}
)
```

### LocalTask

For running Python functions locally:

```python
from molexp import LocalTask

def my_function(x, y):
    return x + y

task = LocalTask(
    name="add_numbers",
    func=my_function,
    inputs={"x": 1, "y": 2}
)
```

### RemoteTask

For executing tasks on remote systems:

```python
from molexp import RemoteTask

task = RemoteTask(
    name="remote_calculation",
    func=my_function,
    inputs={"data": "remote_data"},
    host="compute.server.com",
    user="username"
)
```

## Task Properties

All tasks share common properties:

- **name**: Unique identifier for the task
- **inputs**: Dictionary of input parameters
- **outputs**: List of expected output names
- **dependencies**: Other tasks this task depends on
- **status**: Current execution status
- **result**: Task execution result

## Task Execution

Tasks can be executed individually or as part of workflows:

```python
# Execute single task
result = task.execute()

# Add to task pool
from molexp import TaskPool
pool = TaskPool()
pool.add_task(task)
```

## Best Practices

1. **Naming**: Use descriptive names that reflect the task's purpose
2. **Dependencies**: Clearly define task dependencies for proper execution order
3. **Error Handling**: Include appropriate error handling in task functions
4. **Resource Management**: Consider computational requirements when designing tasks
