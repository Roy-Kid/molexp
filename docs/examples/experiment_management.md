# Experiment Management

This example shows how to use MolExp's Experiment class for managing complex parameter studies and workflows.

## Overview

We'll create an experiment that studies the effect of temperature and pressure on a chemical reaction, demonstrating:
1. Parameter space definition
2. Experiment setup and configuration
3. Result collection and analysis

## Code Example

```python
from molexp import (
    Experiment, ParameterSpace, FloatParameter, 
    LocalTask, GridSampler
)
import numpy as np

# Define reaction simulation function
def simulate_reaction(temperature, pressure, catalyst="default"):
    """Simulate chemical reaction under given conditions."""
    import random
    random.seed(int(temperature * pressure))  # Reproducible results
    
    # Mock reaction kinetics
    rate_constant = np.exp(-5000 / temperature) * (pressure ** 0.5)
    
    # Add some realistic noise
    noise = random.gauss(0, 0.1)
    yield_percent = min(100, max(0, rate_constant * 100 + noise))
    
    return {
        'yield': yield_percent,
        'rate_constant': rate_constant,
        'conversion': yield_percent * 0.8  # Mock conversion
    }

def analyze_reaction_data(yield_data, rate_data):
    """Analyze reaction results."""
    return {
        'mean_yield': np.mean(yield_data),
        'max_yield': np.max(yield_data),
        'optimal_conditions': yield_data.index(max(yield_data)),
        'yield_variance': np.var(yield_data),
        'rate_efficiency': np.mean(rate_data)
    }

# Define parameter space
param_space = ParameterSpace({
    'temperature': FloatParameter(
        name='temperature',
        min=298.15,   # Room temperature
        max=373.15,   # Boiling point of water
        step=25.0,
        unit='K'
    ),
    'pressure': FloatParameter(
        name='pressure', 
        min=1.0,      # 1 atm
        max=5.0,      # 5 atm
        step=1.0,
        unit='atm'
    )
})

# Create experiment
experiment = Experiment(
    name="reaction_optimization",
    description="Temperature and pressure optimization for chemical reaction"
)

# Configure experiment
experiment.set_parameter_space(param_space)
experiment.set_sampler(GridSampler())  # Full factorial design

# Add simulation task
simulation_task = LocalTask(
    name="reaction_simulation",
    func=simulate_reaction,
    inputs={'catalyst': 'platinum'},  # Fixed parameter
    outputs=['reaction_data']
)

# Add analysis task  
analysis_task = LocalTask(
    name="data_analysis",
    func=analyze_reaction_data,
    outputs=['analysis_results']
)

experiment.add_task(simulation_task)
experiment.add_task(analysis_task, aggregate_inputs=True)  # Collect all simulation data

# Configure execution
experiment.configure(
    max_parallel=4,
    save_intermediates=True,
    output_dir="reaction_study_results/"
)

# Execute experiment
print("Starting reaction optimization experiment...")
results = experiment.run()

# Analyze results across parameter space
print(f"\nExperiment completed. Processed {len(results)} parameter combinations.")

# Find best conditions
best_result = max(results, key=lambda x: x.outputs['reaction_data']['yield'])
print(f"\nBest yield: {best_result.outputs['reaction_data']['yield']:.2f}%")
print(f"Optimal conditions: T={best_result.parameters['temperature']:.1f}K, "
      f"P={best_result.parameters['pressure']:.1f}atm")

# Summary statistics
yields = [r.outputs['reaction_data']['yield'] for r in results]
print(f"\nYield statistics:")
print(f"  Mean: {np.mean(yields):.2f}%")
print(f"  Std:  {np.std(yields):.2f}%")
print(f"  Range: {np.min(yields):.2f}% - {np.max(yields):.2f}%")

# Parameter correlation analysis
temps = [r.parameters['temperature'] for r in results]
pressures = [r.parameters['pressure'] for r in results]

temp_yield_corr = np.corrcoef(temps, yields)[0, 1]
pressure_yield_corr = np.corrcoef(pressures, yields)[0, 1]

print(f"\nParameter correlations with yield:")
print(f"  Temperature: {temp_yield_corr:.3f}")
print(f"  Pressure:    {pressure_yield_corr:.3f}")
```

## Advanced Features

### Custom Sampling Strategy

```python
from molexp import RandomSampler, LatinHypercubeSampler

# Use Latin Hypercube sampling for better space coverage
lhs_sampler = LatinHypercubeSampler(seed=42)
experiment.set_sampler(lhs_sampler)

# Or random sampling with specific number of points
random_sampler = RandomSampler(seed=123)
parameter_sets = random_sampler.sample(param_space, n_samples=50)
```

### Result Export and Visualization

```python
# Export results to different formats
experiment.export_results("results.csv", format="csv")
experiment.export_results("results.json", format="json") 
experiment.export_results("results.xlsx", format="excel")

# Generate visualization plots
experiment.plot_parameter_sweep('temperature', 'yield')
experiment.plot_2d_heatmap('temperature', 'pressure', 'yield')
experiment.plot_correlation_matrix()
```

### Experiment Resumption

```python
# Save experiment state
experiment.save_checkpoint("experiment_checkpoint.pkl")

# Resume from checkpoint
resumed_experiment = Experiment.load_checkpoint("experiment_checkpoint.pkl")
additional_results = resumed_experiment.continue_run()
```

## Key Features Demonstrated

1. **Parameter Spaces**: Systematic definition of experimental variables
2. **Experiment Configuration**: Setting up parallel execution and output management
3. **Task Integration**: Combining simulation and analysis tasks
4. **Result Management**: Automatic collection and organization of results
5. **Statistical Analysis**: Built-in tools for analyzing parameter studies

## Best Practices

1. **Start Small**: Test with a few parameter combinations before full runs
2. **Save Checkpoints**: Enable checkpointing for long-running experiments
3. **Organize Outputs**: Use descriptive output directories and file naming
4. **Validate Results**: Include sanity checks and validation tasks
5. **Document Parameters**: Provide clear descriptions and units for all parameters

## Next Steps

- Explore adaptive sampling strategies
- Add visualization and reporting tasks
- Integrate with external simulation software
- Implement multi-objective optimization
