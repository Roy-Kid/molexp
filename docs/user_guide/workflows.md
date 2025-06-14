# Workflow Execution

MolExp provides sophisticated workflow execution capabilities through experiments and executors. This guide covers how to orchestrate complex workflows efficiently using the `ExperimentExecutor`.

## Core Concepts

### Experiment-Based Workflows

MolExp uses an experiment-centric approach where workflows are defined as experiments containing tasks:

```python
from molexp import Experiment, ExperimentExecutor, LocalTask

# Create an experiment
experiment = Experiment(name="molecular_analysis_workflow")

# Define workflow tasks
prep_task = LocalTask(
    name="prepare_data",
    commands=["python scripts/prepare_data.py"]
)

analysis_task = LocalTask(
    name="run_analysis", 
    commands=["python scripts/analyze.py"],
    deps=["prepare_data"]
)

viz_task = LocalTask(
    name="create_visualization",
    commands=["python scripts/visualize.py"],
    deps=["run_analysis"]
)

# Add tasks to experiment
experiment.add_task(prep_task)
experiment.add_task(analysis_task)
experiment.add_task(viz_task)

# Execute the workflow
executor = ExperimentExecutor(experiment)
results = executor.run()
```

### Architecture Overview

1. **Experiment**: Defines the workflow and contains tasks
2. **ExperimentExecutor**: Manages experiment execution using internal components
3. **TaskGraph**: Handles dependency resolution (used internally)
4. **Executor**: Performs actual task execution (used internally)

## Workflow Execution Patterns

### Pipeline Pattern

Linear sequence of processing steps:

```python
from molexp import Experiment, ExperimentExecutor, LocalTask

# Create pipeline experiment
pipeline = Experiment(name="data_pipeline")

# Sequential tasks
preprocess = LocalTask(
    name="preprocess",
    commands=["python preprocess.py data/raw/ data/clean/"]
)

analyze = LocalTask(
    name="analyze", 
    commands=["python analyze.py data/clean/ results/"],
    deps=["preprocess"]
)

visualize = LocalTask(
    name="visualize",
    commands=["python visualize.py results/ plots/"],
    deps=["analyze"]
)

pipeline.add_task(preprocess)
pipeline.add_task(analyze) 
pipeline.add_task(visualize)

# Execute pipeline
executor = ExperimentExecutor(pipeline)
results = executor.run()
```

### Fan-out Pattern

One task creates work for multiple parallel tasks:

One task feeds multiple independent tasks:

```python
# Data preparation → Multiple parallel analyses
graph = TaskGraph()
graph.add_task(prepare_task)
graph.add_task(analysis1_task, dependencies=[prepare_task])
graph.add_task(analysis2_task, dependencies=[prepare_task])
graph.add_task(analysis3_task, dependencies=[prepare_task])
```

### Fan-in Pattern

Multiple tasks feed into a single task:

```python
# Multiple data sources → Combined analysis
graph = TaskGraph()
graph.add_task(source1_task)
graph.add_task(source2_task)
graph.add_task(source3_task)
graph.add_task(combine_task, dependencies=[source1_task, source2_task, source3_task])
```

### Diamond Pattern

Complex dependency relationships:

```python
# A → B,C → D (B and C both depend on A, D depends on both B and C)
graph = TaskGraph()
graph.add_task(task_a)
graph.add_task(task_b, dependencies=[task_a])
graph.add_task(task_c, dependencies=[task_a])
graph.add_task(task_d, dependencies=[task_b, task_c])
```

## Advanced Execution Features

### Conditional Execution

Execute tasks based on conditions:

```python
def should_execute(task, context):
    return context.get('enable_analysis', True)

executor.set_condition_check(should_execute)
```

### Retry Logic

Automatic retry for failed tasks:

```python
executor.configure(
    retry_attempts=3,
    retry_delay=1.0,
    retry_exponential_backoff=True
)
```

### Resource Management

Control computational resources:

```python
executor.configure(
    max_memory_per_task="4GB",
    cpu_affinity=True,
    priority_scheduling=True
)
```

### Progress Monitoring

Track execution progress:

```python
def progress_callback(task, status):
    print(f"Task {task.name}: {status}")

executor.set_progress_callback(progress_callback)
```

## Workflow Execution Strategies

### Batch Processing

Process tasks in batches:

```python
batch_executor = BatchExecutor(batch_size=10)
results = batch_executor.execute(task_pool)
```

### Streaming Processing

Process tasks as they become available:

```python
stream_executor = StreamExecutor()
for result in stream_executor.execute_stream(task_graph):
    process_result(result)
```

### Adaptive Execution

Dynamically adjust execution based on system resources:

```python
adaptive_executor = AdaptiveExecutor()
adaptive_executor.configure(
    monitor_system_resources=True,
    adjust_parallelism=True,
    memory_threshold=0.8
)
```

## Error Handling and Recovery

### Fault Tolerance

Handle task failures gracefully:

```python
executor.configure(
    fault_tolerance='continue',  # Continue with other tasks
    error_propagation=False,     # Don't fail entire workflow
    partial_results=True         # Return partial results
)
```

### Checkpoint and Resume

Save workflow state for recovery:

```python
executor.configure(
    checkpoint_interval=100,  # Checkpoint every 100 tasks
    checkpoint_dir="checkpoints/",
    auto_resume=True
)

# Resume from checkpoint
executor.resume_from_checkpoint("checkpoints/workflow_state.pkl")
```

## Best Practices

1. **Dependency Design**: Minimize dependencies to maximize parallelization
2. **Resource Planning**: Consider memory and CPU requirements when designing workflows
3. **Error Handling**: Implement robust error handling and recovery mechanisms
4. **Monitoring**: Use progress tracking and logging for long-running workflows
5. **Testing**: Test workflows with small datasets before full-scale execution
6. **Documentation**: Document workflow logic and dependencies clearly
