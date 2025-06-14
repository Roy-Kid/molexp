# Task Classes

This section provides detailed API documentation for all task classes in MolExp.

## Base Task Class

::: molexp.task.Task
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute
        - set_inputs
        - get_outputs
        - add_dependency
        - remove_dependency

## Hamilton Task

::: molexp.task.HamiltonTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute
        - build_config
        - create_driver

## Shell Task

::: molexp.task.ShellTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute
        - format_command
        - parse_output

## Local Task

::: molexp.task.LocalTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute
        - validate_function
        - prepare_inputs

## Remote Task

::: molexp.task.RemoteTask
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute
        - establish_connection
        - transfer_files
        - cleanup_remote

## Task Status

Tasks in MolExp can have the following statuses:

- `PENDING`: Task is created but not yet executed
- `RUNNING`: Task is currently executing
- `COMPLETED`: Task completed successfully
- `FAILED`: Task execution failed
- `CANCELLED`: Task was cancelled before completion

## Task Properties

### Common Properties

All task classes inherit these properties:

- **name** (str): Unique identifier for the task
- **inputs** (dict): Input parameters for the task
- **outputs** (list): Expected output names
- **dependencies** (list): List of dependent tasks
- **status** (TaskStatus): Current execution status
- **result** (Any): Task execution result
- **error** (Exception): Error information if task failed
- **start_time** (datetime): Task start timestamp
- **end_time** (datetime): Task completion timestamp
- **duration** (float): Task execution duration in seconds

### Task-Specific Properties

#### HamiltonTask

- **func** (callable): Hamilton function to execute
- **config** (dict): Hamilton driver configuration
- **driver** (hamilton.Driver): Hamilton driver instance

#### ShellTask

- **command** (str): Shell command template
- **working_dir** (str): Working directory for command execution
- **env** (dict): Environment variables
- **timeout** (float): Command timeout in seconds

#### LocalTask

- **func** (callable): Python function to execute
- **args** (tuple): Positional arguments for function
- **kwargs** (dict): Keyword arguments for function

#### RemoteTask

- **host** (str): Remote host address
- **user** (str): Remote username
- **ssh_key** (str): SSH key path
- **remote_dir** (str): Remote working directory
- **connection** (paramiko.SSHClient): SSH connection

## Task Methods

### Execution Methods

```python
# Execute a single task
result = task.execute()

# Execute with custom inputs
result = task.execute(inputs={'param': 'value'})

# Execute with timeout
result = task.execute(timeout=300)
```

### Input/Output Management

```python
# Set task inputs
task.set_inputs({'param1': 'value1', 'param2': 'value2'})

# Get task outputs
outputs = task.get_outputs()

# Check if task has required inputs
if task.has_required_inputs():
    result = task.execute()
```

### Dependency Management

```python
# Add dependencies
task.add_dependency(other_task)
task.add_dependencies([task1, task2, task3])

# Remove dependencies
task.remove_dependency(other_task)
task.clear_dependencies()

# Check dependencies
if task.dependencies_satisfied():
    result = task.execute()
```

### Status and Monitoring

```python
# Check task status
if task.is_completed():
    print(f"Task result: {task.result}")
elif task.is_failed():
    print(f"Task error: {task.error}")

# Get execution information
info = task.get_execution_info()
print(f"Duration: {info['duration']}s")
print(f"Memory used: {info['memory_usage']}MB")
```

## Task Creation Examples

### Creating a Hamilton Task

```python
from molexp import HamiltonTask

def calculate_energy(atoms: list, method: str = 'dft') -> float:
    # Hamilton function implementation
    pass

task = HamiltonTask(
    name="energy_calculation",
    func=calculate_energy,
    inputs={'atoms': molecule_atoms, 'method': 'dft'},
    outputs=['energy']
)
```

### Creating a Shell Task

```python
from molexp import ShellTask

task = ShellTask(
    name="run_simulation",
    command="python simulate.py --input {input_file} --output {output_file}",
    inputs={'input_file': 'input.xyz', 'output_file': 'output.log'},
    working_dir="/path/to/simulation",
    timeout=3600
)
```

### Creating a Local Task

```python
from molexp import LocalTask

def process_data(data, method='standard'):
    # Processing implementation
    return processed_data

task = LocalTask(
    name="data_processing",
    func=process_data,
    inputs={'data': raw_data, 'method': 'advanced'},
    outputs=['processed_data']
)
```

### Creating a Remote Task

```python
from molexp import RemoteTask

task = RemoteTask(
    name="remote_calculation",
    func=heavy_computation,
    inputs={'large_dataset': data},
    host="compute.cluster.org",
    user="username",
    ssh_key="~/.ssh/id_rsa",
    remote_dir="/scratch/username/job"
)
```

## Error Handling

Tasks provide comprehensive error handling:

```python
try:
    result = task.execute()
except TaskExecutionError as e:
    print(f"Task failed: {e}")
    print(f"Error details: {task.error}")
    print(f"Traceback: {task.traceback}")
except TaskTimeoutError as e:
    print(f"Task timed out: {e}")
except TaskDependencyError as e:
    print(f"Dependency not satisfied: {e}")
```

## Advanced Features

### Custom Task Classes

You can create custom task classes by inheriting from the base Task class:

```python
from molexp.task import Task

class CustomTask(Task):
    def __init__(self, name, custom_param, **kwargs):
        super().__init__(name, **kwargs)
        self.custom_param = custom_param
    
    def execute(self, inputs=None, **kwargs):
        # Custom execution logic
        try:
            # Your implementation here
            result = self._custom_execution()
            self.status = TaskStatus.COMPLETED
            self.result = result
            return result
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = e
            raise
```

### Task Serialization

Tasks can be serialized for storage and distributed execution:

```python
import pickle

# Serialize task
task_data = task.serialize()
with open('task.pkl', 'wb') as f:
    pickle.dump(task_data, f)

# Deserialize task
with open('task.pkl', 'rb') as f:
    task_data = pickle.load(f)
task = Task.deserialize(task_data)
```
