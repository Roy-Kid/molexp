# Experiments

Experiments in MolExp provide a high-level interface for managing complex scientific workflows. They combine parameter studies, task execution, and result management into a cohesive framework.

## Creating Experiments

```python
from molexp import Experiment

experiment = Experiment(
    name="molecular_dynamics_study",
    description="Comparative MD simulation study"
)
```

## Parameter Studies

Experiments excel at parameter exploration:

```python
from molexp import ParameterSpace, FloatParameter, IntParameter

# Define parameter space
param_space = ParameterSpace({
    'temperature': FloatParameter(min=300, max=400, step=25),
    'pressure': FloatParameter(min=1, max=10, step=1),
    'steps': IntParameter(min=1000, max=10000, step=1000)
})

experiment.set_parameter_space(param_space)
```

## Adding Tasks to Experiments

```python
# Add preprocessing task
preprocess_task = ShellTask(
    name="preprocess",
    command="prepare_system.py --temp {temperature} --press {pressure}"
)

# Add simulation task
simulation_task = LocalTask(
    name="simulation",
    func=run_md_simulation,
    inputs={"steps": None, "system": None}  # Will be filled from parameters
)

experiment.add_task(preprocess_task)
experiment.add_task(simulation_task)
```

## Execution and Results

```python
# Execute the experiment
results = experiment.run()

# Access results
for result in results:
    print(f"Parameters: {result.parameters}")
    print(f"Outputs: {result.outputs}")
    print(f"Status: {result.status}")
```

## Experiment Configuration

Experiments can be configured with various options:

```python
experiment.configure(
    max_parallel=4,          # Maximum parallel executions
    retry_failed=True,       # Retry failed tasks
    save_intermediates=True, # Save intermediate results
    output_dir="results/"    # Output directory
)
```

## Data Management

Experiments automatically handle data organization:

- **Input Management**: Automatic parameter injection
- **Output Collection**: Centralized result storage
- **Metadata**: Execution timestamps, status tracking
- **Reproducibility**: Parameter and configuration logging

## Advanced Features

### Custom Workflows

```python
# Define custom execution workflow
def custom_workflow(experiment):
    # Custom pre-processing
    experiment.setup_environment()
    
    # Execute with custom logic
    for param_set in experiment.parameter_sets:
        if meets_criteria(param_set):
            experiment.execute_parameter_set(param_set)
    
    # Custom post-processing
    experiment.cleanup()

experiment.set_workflow(custom_workflow)
```

### Result Analysis

```python
# Built-in analysis tools
analysis = experiment.analyze_results()
print(f"Success rate: {analysis.success_rate}")
print(f"Average runtime: {analysis.avg_runtime}")

# Custom analysis
def custom_analysis(results):
    # Your analysis code here
    return analysis_summary

experiment.add_analysis(custom_analysis)
```

## Best Practices

1. **Modularity**: Break complex experiments into smaller, reusable components
2. **Documentation**: Include clear descriptions and parameter documentation
3. **Validation**: Implement parameter validation and sanity checks
4. **Monitoring**: Use logging and progress tracking for long-running experiments
5. **Reproducibility**: Always save experiment configurations and random seeds
