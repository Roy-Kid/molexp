# Parameter Studies

Parameter studies are a core feature of MolExp, enabling systematic exploration of parameter spaces in scientific computing. This guide covers how to define, execute, and analyze parameter studies.

## Parameter Types

MolExp supports various parameter types for comprehensive studies:

### Numeric Parameters

```python
from molexp import FloatParameter, IntParameter

# Continuous parameters
temperature = FloatParameter(
    name="temperature",
    min=273.15,
    max=373.15,
    step=10.0,
    unit="K"
)

# Discrete parameters  
num_particles = IntParameter(
    name="num_particles",
    min=100,
    max=1000,
    step=100
)
```

### Categorical Parameters

```python
from molexp import CategoricalParameter

solvent = CategoricalParameter(
    name="solvent",
    choices=["water", "ethanol", "acetone", "dmso"]
)

method = CategoricalParameter(
    name="method",
    choices=["dft", "mp2", "ccsd"],
    default="dft"
)
```

### Boolean Parameters

```python
from molexp import BooleanParameter

use_constraints = BooleanParameter(
    name="use_constraints",
    default=True
)
```

## Parameter Spaces

Combine multiple parameters into a parameter space:

```python
from molexp import ParameterSpace

param_space = ParameterSpace({
    'temperature': FloatParameter(min=300, max=400, step=25),
    'pressure': FloatParameter(min=1, max=10, step=2),
    'solvent': CategoricalParameter(choices=["water", "ethanol"]),
    'use_pbc': BooleanParameter(default=True)
})
```

## Sampling Strategies

### Grid Sampling

Systematic exploration of all parameter combinations:

```python
# Full factorial design
sampler = GridSampler()
parameter_sets = sampler.sample(param_space, max_samples=None)

# Limited grid sampling
parameter_sets = sampler.sample(param_space, max_samples=100)
```

### Random Sampling

Random exploration of parameter space:

```python
from molexp import RandomSampler

sampler = RandomSampler(seed=42)
parameter_sets = sampler.sample(param_space, n_samples=50)
```

### Latin Hypercube Sampling

Efficient space-filling sampling:

```python
from molexp import LatinHypercubeSampler

sampler = LatinHypercubeSampler()
parameter_sets = sampler.sample(param_space, n_samples=100)
```

### Sobol Sequence Sampling

Low-discrepancy quasi-random sampling:

```python
from molexp import SobolSampler

sampler = SobolSampler()
parameter_sets = sampler.sample(param_space, n_samples=64)  # Power of 2 recommended
```

## Advanced Sampling

### Adaptive Sampling

Iteratively refine sampling based on results:

```python
from molexp import AdaptiveSampler

def objective_function(params, results):
    return results.get('energy', float('inf'))

sampler = AdaptiveSampler(
    objective=objective_function,
    strategy='minimize'
)

# Initial sampling
initial_sets = sampler.initial_sample(param_space, n_initial=20)

# Adaptive refinement
for iteration in range(5):
    new_sets = sampler.adaptive_sample(param_space, n_samples=10)
    # Execute experiments with new_sets
    # Update sampler with results
```

### Constrained Sampling

Apply constraints to parameter combinations:

```python
def constraint_function(params):
    # Temperature and pressure must satisfy ideal gas relationship
    return params['pressure'] * params['volume'] > params['temperature'] * 0.08314

constrained_sampler = ConstrainedSampler(
    base_sampler=RandomSampler(),
    constraints=[constraint_function]
)
```

## Parameter Study Execution

### Basic Execution

```python
from molexp import Experiment

experiment = Experiment("parameter_study")
experiment.set_parameter_space(param_space)
experiment.set_sampler(GridSampler())

# Add tasks that use parameters
experiment.add_task(simulation_task)
experiment.add_task(analysis_task)

# Execute parameter study
results = experiment.run()
```

### Parallel Execution

```python
experiment.configure(
    max_parallel=8,
    batch_size=4,
    execution_mode='parallel'
)
```

### Streaming Execution

For large parameter studies:

```python
for parameter_set in experiment.parameter_stream():
    result = experiment.execute_parameter_set(parameter_set)
    process_result_immediately(result)
```

## Result Analysis

### Basic Statistics

```python
from molexp import ParameterAnalyzer

analyzer = ParameterAnalyzer(results)

# Summary statistics
summary = analyzer.summarize()
print(f"Mean energy: {summary.energy.mean}")
print(f"Std energy: {summary.energy.std}")

# Parameter correlations
correlations = analyzer.correlate()
print(correlations)
```

### Sensitivity Analysis

```python
# Sobol sensitivity indices
sensitivity = analyzer.sobol_sensitivity('energy')
print(f"First-order indices: {sensitivity.first_order}")
print(f"Total indices: {sensitivity.total}")

# Morris screening
morris_results = analyzer.morris_screening('energy')
```

### Visualization

```python
# Parameter space visualization
analyzer.plot_parameter_space()

# Response surface
analyzer.plot_response_surface('temperature', 'pressure', 'energy')

# Parallel coordinates plot
analyzer.plot_parallel_coordinates(['temperature', 'pressure'], 'energy')

# Correlation heatmap
analyzer.plot_correlation_heatmap()
```

### Optimization

Find optimal parameter combinations:

```python
# Find minimum/maximum
optimal_params = analyzer.find_optimum('energy', mode='minimize')
print(f"Optimal parameters: {optimal_params.parameters}")
print(f"Optimal value: {optimal_params.value}")

# Pareto frontier (multi-objective)
pareto_set = analyzer.pareto_frontier(['energy', 'accuracy'])
```

## Best Practices

### Design of Studies

1. **Start Small**: Begin with coarse sampling, then refine
2. **Consider Physics**: Use domain knowledge to set reasonable parameter ranges
3. **Balance Coverage**: Trade-off between thoroughness and computational cost

### Execution Strategies

1. **Checkpointing**: Save intermediate results for long studies
2. **Resource Management**: Monitor memory and CPU usage
3. **Error Handling**: Implement robust failure recovery

### Analysis and Interpretation

1. **Statistical Significance**: Ensure adequate sampling for statistical conclusions
2. **Visualization**: Use multiple visualization techniques for insights
3. **Validation**: Cross-validate findings with independent datasets

### Computational Efficiency

1. **Parallel Execution**: Leverage multiple cores/nodes when possible
2. **Caching**: Cache expensive computations when parameters repeat
3. **Early Stopping**: Implement criteria to stop unpromising parameter sets

## Example: Complete Parameter Study

```python
from molexp import *

# Define parameter space
param_space = ParameterSpace({
    'temperature': FloatParameter(min=300, max=400, step=25),
    'concentration': FloatParameter(min=0.1, max=1.0, step=0.1),
    'ph': FloatParameter(min=6, max=8, step=0.5),
    'catalyst': CategoricalParameter(choices=['A', 'B', 'C'])
})

# Create experiment
experiment = Experiment("catalysis_study")
experiment.set_parameter_space(param_space)
experiment.set_sampler(LatinHypercubeSampler())

# Add simulation task
simulation_task = LocalTask(
    name="reaction_simulation",
    func=simulate_reaction,
    inputs={},  # Parameters auto-injected
    outputs=['yield', 'selectivity', 'rate']
)
experiment.add_task(simulation_task)

# Execute study
results = experiment.run(max_parallel=4)

# Analyze results
analyzer = ParameterAnalyzer(results)
optimal = analyzer.find_optimum('yield', mode='maximize')
analyzer.plot_response_surface('temperature', 'concentration', 'yield')

print(f"Best yield: {optimal.value:.2f} at {optimal.parameters}")
```
