# Quick Start

This guide will walk you through creating your first MolExp workflow in just a few minutes.

## Your First Workflow

Let's create a simple data processing workflow with three tasks:

```python
import molexp as mx

# Step 1: Create tasks with dependencies
data_prep = mx.Task(
    name="prepare_data",
    readme="Download and prepare input data",
    args=["--source", "dataset.csv", "--output", "clean_data.csv"],
    outputs=["clean_data.csv"]
)

analysis = mx.Task(
    name="analyze_data", 
    readme="Perform statistical analysis",
    args=["--input", "clean_data.csv", "--output", "results.json"],
    deps=["prepare_data"],  # This task depends on prepare_data
    outputs=["results.json"]
)

report = mx.Task(
    name="generate_report",
    readme="Create final report",
    args=["--results", "results.json", "--output", "report.pdf"],
    deps=["analyze_data"],  # This task depends on analyze_data
    outputs=["report.pdf"]
)

# Step 2: Create a task pool and add tasks
workflow = mx.TaskPool(name="data_pipeline")
workflow.add_task(data_prep)
workflow.add_task(analysis)
workflow.add_task(report)

print(f"Created workflow with {len(workflow.tasks)} tasks")

# Step 3: Create experiment and executor
experiment = mx.Experiment(name="quickstart", task_pool=workflow)
executor = mx.ExperimentExecutor(experiment)

# Step 4: Execute the workflow
print("Running workflow...")
results = executor.run()

# Step 5: Check results
print("\\nExecution completed!")
for task_name, result in results.items():
    status = result.get('status', 'unknown')
    print(f"  {task_name}: {status}")

print(f"\\nWorkflow completed: {executor.is_execution_completed()}")
```

## Understanding the Workflow

### Task Dependencies

In our example:
- `prepare_data` has no dependencies (runs first)
- `analyze_data` depends on `prepare_data` 
- `generate_report` depends on `analyze_data`

MolExp automatically determines the execution order: `prepare_data` → `analyze_data` → `generate_report`

### Task Outputs

Tasks can specify expected outputs:
```python
task = mx.Task(
    name="process_data",
    outputs=["result1.txt", "result2.json"]
)
```

This helps with:
- Workflow validation
- Dependency tracking
- Result verification

### Execution Status

Monitor workflow progress:
```python
# Check if workflow completed successfully
if executor.is_execution_completed():
    print("All tasks completed!")

# Check if any tasks failed
if executor.is_execution_failed():
    print("Some tasks failed!")

# Get detailed status
status = executor.get_execution_status()
print(f"Completed: {status.get('completed', 0)} tasks")
print(f"Failed: {status.get('failed', 0)} tasks")
```

## Working with Experiments

For more complex workflows, use the Experiment class:

```python
# Create an experiment
experiment = mx.Experiment(
    name="my_research_project",
    readme="Comprehensive data analysis pipeline"
)

# Add tasks to experiment
experiment.add_task(data_prep)
experiment.add_task(analysis)
experiment.add_task(report)

# Validate experiment
experiment.validate_experiment()

# Save experiment configuration
experiment.to_yaml("my_experiment.yaml")

# Create executor from experiment
executor = mx.ExperimentExecutor(experiment)
results = executor.run()
```

## Next Steps

Now that you've created your first workflow, explore:

1. **[Basic Concepts](concepts.md)** - Understand core MolExp concepts
2. **[Examples](../examples/basic_usage.md)** - Learn from detailed examples
3. **[User Guide](../user_guide/tasks.md)** - Dive deeper into task types and features
4. **[API Reference](../api/tasks.md)** - Complete API documentation

## Common Patterns

### Sequential Processing
```python
# Tasks run one after another
task1 = mx.Task(name="step1")
task2 = mx.Task(name="step2", deps=["step1"])
task3 = mx.Task(name="step3", deps=["step2"])
```

### Parallel Processing
```python
# Tasks run in parallel (no dependencies between them)
task1 = mx.Task(name="parallel1", deps=["preparation"])
task2 = mx.Task(name="parallel2", deps=["preparation"])
task3 = mx.Task(name="parallel3", deps=["preparation"])
```

### Fan-out and Fan-in
```python
# One task feeds into multiple parallel tasks
prep = mx.Task(name="preparation")
analysis1 = mx.Task(name="analysis1", deps=["preparation"])
analysis2 = mx.Task(name="analysis2", deps=["preparation"])

# Multiple tasks feed into one final task
summary = mx.Task(name="summary", deps=["analysis1", "analysis2"])
```
