# Execution Engine

This section provides API documentation for MolExp's execution engine components.

## Executor

The main execution engine for running tasks and workflows.

::: molexp.executor.Executor
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - execute_task
        - execute
        - configure
        - get_status
        - get_performance_stats

### Basic Execution

```python
from molexp import Executor, LocalTask

# Create executor
executor = Executor(max_workers=4)

# Execute single task
task = LocalTask("example", func=lambda x: x*2, inputs={'x': 5})
result = executor.execute_task(task)
print(f"Result: {result}")

# Execute task graph
from molexp import TaskGraph
graph = TaskGraph()
graph.add_task(task)
results = executor.execute(graph)
```

### Configuration Options

```python
# Configure executor with various options
executor = Executor(
    max_workers=8,              # Number of parallel workers
    execution_mode='parallel',   # 'sequential', 'parallel', 'auto'
    timeout=3600,               # Global timeout in seconds
    memory_limit_mb=4096,       # Memory limit per worker
    retry_policy={              # Default retry policy
        'max_attempts': 3,
        'backoff_factor': 2.0,
        'initial_delay': 1.0
    }
)

# Runtime configuration
executor.configure(
    enable_monitoring=True,
    log_level='INFO',
    checkpoint_interval=100,
    resource_monitoring=True
)
```

## ExperimentExecutor

High-level executor specifically designed for running complete experiments.

::: molexp.workflow.ExperimentExecutor
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - run
        - get_executable_tasks
        - mark_task_running
        - mark_task_completed
        - mark_task_failed
        - get_execution_status
        - is_execution_completed
        - is_execution_failed
        - reset_execution
        - get_execution_summary

### Basic Experiment Execution

```python
from molexp import ExperimentExecutor, Experiment, LocalTask

# Create experiment
experiment = Experiment(name="molecular_analysis")

# Add tasks to experiment
task1 = LocalTask(
    name="prepare_data", 
    commands=["python prepare.py"]
)
task2 = LocalTask(
    name="analyze", 
    commands=["python analyze.py"],
    deps=["prepare_data"]
)

experiment.add_task(task1)
experiment.add_task(task2)

# Create experiment executor
executor = ExperimentExecutor(experiment)

# Run the experiment
results = executor.run()
print(f"Execution results: {results}")

# Check execution status
print(f"Completed: {executor.is_execution_completed()}")
print(f"Status summary: {executor.get_execution_status()}")
```

### Experiment Execution with Parameters

```python
from molexp import ExperimentExecutor, Experiment, LocalTask
from molexp.param import Param

# Create parameterized experiment
experiment = Experiment(name="parameter_study")

# Add parameterized tasks
prep_task = LocalTask(
    name="prepare_simulation",
    commands=["python setup_simulation.py --temp ${temperature} --pressure ${pressure}"]
)

sim_task = LocalTask(
    name="run_simulation", 
    commands=["python simulate.py --input prepared_system.inp"],
    deps=["prepare_simulation"]
)

analysis_task = LocalTask(
    name="analyze_results",
    commands=["python analyze.py --results simulation_output.dat"],
    deps=["run_simulation"]
)

experiment.add_task(prep_task)
experiment.add_task(sim_task)
experiment.add_task(analysis_task)

# Create executor
executor = ExperimentExecutor(experiment, name="parameter_study_executor")

# Run with specific parameters
params = Param({
    'temperature': 300.0,
    'pressure': 1.0,
    'ensemble': 'NPT'
})

results = executor.run(params)

# Get detailed execution summary
summary = executor.get_execution_summary()
print(f"Experiment: {summary['experiment_name']}")
print(f"Tasks executed: {summary['task_count']}")
print(f"Status: {summary['status_summary']}")
```

### Advanced Experiment Control

```python
# Manual execution control
executor = ExperimentExecutor(experiment)

# Check which tasks are ready to execute
executable_tasks = executor.get_executable_tasks()
print(f"Ready to execute: {executable_tasks}")

# Manual task status management
for task_name in executable_tasks:
    executor.mark_task_running(task_name)
    
    try:
        # Custom task execution logic here
        result = custom_execute_task(task_name)
        executor.mark_task_completed(task_name, result)
    except Exception as e:
        executor.mark_task_failed(task_name, e)

# Reset and re-run if needed
if not executor.is_execution_completed():
    executor.reset_execution()
    results = executor.run()
```
    resume_from_checkpoint=False,
    save_intermediate_results=True,
    enable_dynamic_scaling=True
)
```

### Workflow Control

```python
import time
import threading

# Start workflow in background thread
def run_workflow():
    return workflow_executor.execute_workflow(long_running_workflow)

workflow_thread = threading.Thread(target=run_workflow)
workflow_thread.start()

# Monitor and control execution
time.sleep(10)
workflow_executor.pause()
print("Workflow paused")

time.sleep(5)
workflow_executor.resume()
print("Workflow resumed")

# Get execution state
state = workflow_executor.get_execution_state()
print(f"Completed tasks: {state['completed_tasks']}")
print(f"Running tasks: {state['running_tasks']}")
print(f"Pending tasks: {state['pending_tasks']}")
```

## Task Status Management

Constants and utilities for tracking task execution status.

::: molexp.executor.TaskStatus
    options:
      show_root_heading: true
      show_source: true

### Status Tracking

```python
from molexp import TaskStatus

# Check task status
if task.status == TaskStatus.PENDING:
    print("Task is waiting to be executed")
elif task.status == TaskStatus.RUNNING:
    print("Task is currently executing")
elif task.status == TaskStatus.COMPLETED:
    print("Task completed successfully")
elif task.status == TaskStatus.FAILED:
    print("Task failed during execution")

# Status transitions
task.status = TaskStatus.RUNNING
# ... execute task ...
task.status = TaskStatus.COMPLETED
```

## Execution Backends

### LocalBackend

Executes tasks on the local machine.

```python
from molexp.dispatch import LocalBackend

backend = LocalBackend(
    max_workers=4,
    temp_dir='/tmp/molexp',
    cleanup_on_exit=True
)

# Configure executor with backend
executor = Executor(backend=backend)
```

### RemoteBackend

Executes tasks on remote systems via SSH.

```python
from molexp.dispatch import RemoteBackend

backend = RemoteBackend(
    hosts=['compute1.example.com', 'compute2.example.com'],
    username='researcher',
    ssh_key_path='~/.ssh/id_rsa',
    max_workers_per_host=4,
    load_balancing=True
)

executor = Executor(backend=backend)
```

### ClusterBackend

Executes tasks on HPC clusters using job schedulers.

```python
from molexp.dispatch import ClusterBackend

backend = ClusterBackend(
    scheduler='slurm',          # 'slurm', 'pbs', 'sge'
    queue='compute',
    max_jobs=10,
    resources_per_job={
        'nodes': 1,
        'cpus_per_task': 4,
        'memory': '8GB',
        'time': '02:00:00'
    }
)

executor = Executor(backend=backend)
```

## Performance Monitoring

### ExecutionMonitor

Monitor execution performance and resource usage.

```python
from molexp import ExecutionMonitor

# Create monitor
monitor = ExecutionMonitor(
    enable_cpu_monitoring=True,
    enable_memory_monitoring=True,
    enable_io_monitoring=True,
    sampling_interval=1.0
)

# Use with executor
executor = Executor(monitor=monitor)
results = executor.execute(workflow)

# Get performance report
performance_report = monitor.get_performance_report()
print(f"Peak CPU usage: {performance_report['peak_cpu_percent']:.1f}%")
print(f"Peak memory usage: {performance_report['peak_memory_mb']:.1f} MB")
print(f"Total I/O operations: {performance_report['total_io_operations']}")
```

### Real-time Monitoring Dashboard

```python
from molexp import MonitoringDashboard

# Create dashboard
dashboard = MonitoringDashboard(
    update_interval=2.0,
    enable_web_interface=True,
    web_port=8080
)

# Start monitoring
dashboard.start()
print("Monitoring dashboard available at http://localhost:8080")

# Execute with dashboard monitoring
executor = Executor(dashboard=dashboard)
results = executor.execute(workflow)

# Stop dashboard
dashboard.stop()
```

## Error Handling and Recovery

### Exception Handling

```python
from molexp.exceptions import (
    TaskExecutionError,
    WorkflowExecutionError, 
    ResourceLimitError,
    TimeoutError
)

try:
    results = executor.execute(workflow)
except TaskExecutionError as e:
    print(f"Task execution failed: {e.task_name}")
    print(f"Error details: {e.details}")
except WorkflowExecutionError as e:
    print(f"Workflow execution failed: {e}")
    # Get partial results
    partial_results = e.partial_results
except ResourceLimitError as e:
    print(f"Resource limit exceeded: {e}")
except TimeoutError as e:
    print(f"Execution timed out: {e}")
```

### Recovery Strategies

```python
# Configure recovery strategies
executor = Executor(
    recovery_strategy='restart_failed',  # 'ignore', 'restart_failed', 'restart_all'
    max_recovery_attempts=3,
    recovery_delay=10.0,
    partial_results_threshold=0.8  # Accept partial results if 80% complete
)

# Execute with recovery
try:
    results = executor.execute(workflow)
except WorkflowExecutionError as e:
    if e.completion_ratio > 0.5:  # More than 50% completed
        print("Accepting partial results due to high completion ratio")
        results = e.partial_results
    else:
        print("Attempting recovery...")
        results = executor.recover_and_continue(workflow, e.checkpoint)
```

## Distributed Execution

### DistributedExecutor

Execute workflows across multiple machines.

```python
from molexp import DistributedExecutor

# Configure distributed execution
distributed_executor = DistributedExecutor(
    cluster_config={
        'master_host': 'master.cluster.com',
        'worker_hosts': [
            'worker1.cluster.com',
            'worker2.cluster.com', 
            'worker3.cluster.com'
        ],
        'communication_backend': 'mpi',  # 'mpi', 'tcp', 'redis'
        'load_balancing': 'dynamic'
    },
    fault_tolerance={
        'enable_worker_recovery': True,
        'heartbeat_interval': 30,
        'max_worker_failures': 2
    }
)

# Execute distributed workflow
results = distributed_executor.execute(
    workflow=large_workflow,
    partition_strategy='automatic',  # How to partition work
    data_locality=True,             # Optimize for data locality
    communication_optimization=True  # Optimize inter-node communication
)
```

### Cloud Execution

Execute workflows on cloud platforms.

```python
from molexp.cloud import AWSExecutor, GCPExecutor, AzureExecutor

# AWS execution
aws_executor = AWSExecutor(
    region='us-west-2',
    instance_types=['c5.xlarge', 'c5.2xlarge'],
    max_instances=10,
    spot_instances=True,
    s3_bucket='molexp-results'
)

# Execute on AWS
results = aws_executor.execute(
    workflow=workflow,
    auto_scaling=True,
    cost_optimization=True
)

# Get cost report
cost_report = aws_executor.get_cost_report()
print(f"Total cost: ${cost_report['total_cost']:.2f}")
```

## Execution Optimization

### Performance Tuning

```python
# Auto-tune executor performance
from molexp import AutoTuner

tuner = AutoTuner(
    optimization_target='throughput',  # 'throughput', 'latency', 'cost'
    tuning_iterations=10,
    test_workload=sample_workflow
)

# Find optimal configuration
optimal_config = tuner.optimize(
    parameter_ranges={
        'max_workers': (1, 16),
        'batch_size': (1, 100),
        'memory_limit_mb': (512, 8192)
    }
)

# Apply optimal configuration
executor = Executor(**optimal_config)
```

### Resource Allocation

```python
# Dynamic resource allocation
from molexp import ResourceManager

resource_manager = ResourceManager(
    total_cpu_cores=16,
    total_memory_gb=64,
    allocation_strategy='fair',     # 'fair', 'priority', 'demand'
    enable_oversubscription=False,
    resource_monitoring=True
)

# Create resource-aware executor
executor = Executor(
    resource_manager=resource_manager,
    dynamic_scaling=True,
    resource_constraints={
        'max_cpu_per_task': 4,
        'max_memory_per_task_gb': 8
    }
)
```

## Integration Examples

### Custom Execution Backend

```python
from molexp.dispatch.base import BaseBackend

class CustomBackend(BaseBackend):
    """Custom execution backend implementation."""
    
    def __init__(self, custom_config):
        super().__init__()
        self.config = custom_config
        
    def execute_task(self, task, **kwargs):
        """Execute a single task."""
        # Custom execution logic
        try:
            result = self._run_custom_execution(task)
            return result
        except Exception as e:
            raise TaskExecutionError(f"Custom backend execution failed: {e}")
    
    def execute_graph(self, graph, **kwargs):
        """Execute a task graph."""
        # Custom graph execution logic
        return super().execute_graph(graph, **kwargs)
    
    def _run_custom_execution(self, task):
        """Custom execution implementation."""
        # Your custom execution logic here
        pass

# Register and use custom backend
executor = Executor(backend=CustomBackend(custom_config={}))
```

### Execution Plugins

```python
from molexp.plugins import ExecutionPlugin

class ProfilingPlugin(ExecutionPlugin):
    """Plugin to profile task execution."""
    
    def before_task_execution(self, task):
        """Called before task execution."""
        import time
        task._start_time = time.time()
        
    def after_task_execution(self, task, result):
        """Called after task execution."""
        import time
        execution_time = time.time() - task._start_time
        print(f"Task {task.name} took {execution_time:.2f} seconds")
        
    def on_task_failure(self, task, error):
        """Called when task fails."""
        print(f"Task {task.name} failed: {error}")

# Use plugin with executor
executor = Executor(plugins=[ProfilingPlugin()])
```

This comprehensive documentation covers MolExp's execution engine, providing both basic usage patterns and advanced features for scalable scientific computing workflows.
