# Parameters API Reference

This section provides detailed API documentation for MolExp's parameter management system, which enables flexible parameter space definition and sampling for experiments.

## Overview

The parameter system in MolExp consists of:
- **Param**: A dictionary-like container for parameter sets
- **ParamSpace**: Definition of parameter spaces with possible values
- **ParamSampler**: Protocol for different sampling strategies
- **Concrete Samplers**: CartesianSampler, RandomSampler, CombinationSampler

## Core Classes

### Param

A parameter container that extends Python's built-in `dict` class.

::: molexp.param.Param
    options:
      show_root_heading: true
      show_source: true

#### Basic Usage

```python
from molexp.param import Param

# Create a parameter set
params = Param({
    'temperature': 300.0,
    'pressure': 1.0,
    'num_particles': 1000,
    'solvent': 'water'
})

# Access parameters like a dictionary
print(f"Temperature: {params['temperature']}")
print(f"Pressure: {params.get('pressure', 1.0)}")

# Update parameters
params['temperature'] = 350.0
params.update({'pressure': 2.0})

# Iterate over parameters
for key, value in params.items():
    print(f"{key}: {value}")
```

#### Integration with Functions

```python
def simulate_system(**kwargs):
    """Example simulation function that accepts parameters"""
    temp = kwargs.get('temperature', 298.15)
    pressure = kwargs.get('pressure', 1.0)
    # ... simulation logic ...
    return f"Simulation at {temp}K, {pressure}atm"

# Use Param with function
params = Param({'temperature': 320.0, 'pressure': 1.5})
result = simulate_system(**params)
print(result)  # "Simulation at 320.0K, 1.5atm"
```

### ParamSpace

A parameter space definition that extends `dict` to define the possible values for each parameter.

::: molexp.param.ParamSpace
    options:
      show_root_heading: true
      show_source: true

#### Basic Usage

```python
from molexp.param import ParamSpace

# Define parameter space
space = ParamSpace({
    'temperature': [273.15, 298.15, 323.15, 373.15],
    'pressure': [0.5, 1.0, 1.5, 2.0],
    'solvent': ['water', 'ethanol', 'acetone', 'dmso'],
    'concentration': [0.1, 0.5, 1.0, 2.0]
})

# Access parameter options
temp_options = space['temperature']
print(f"Temperature options: {temp_options}")

# Get all parameter names
param_names = list(space.keys())
print(f"Parameters: {param_names}")

# Check if parameter exists
if 'pH' in space:
    print("pH parameter defined")
else:
    print("pH parameter not defined")
```

#### Dynamic Parameter Spaces

```python
# Build parameter space dynamically
space = ParamSpace()

# Add continuous parameters
space['temperature'] = list(range(250, 401, 25))  # 250K to 400K in 25K steps
space['pressure'] = [0.1 * i for i in range(1, 21)]  # 0.1 to 2.0 in 0.1 steps

# Add categorical parameters
space['catalyst'] = ['Pd', 'Pt', 'Ru', 'Rh', 'Ir']
space['ligand'] = ['PPh3', 'dppe', 'dppp', 'dppf']

# Add boolean-like parameters
space['use_microwave'] = [True, False]
space['inert_atmosphere'] = ['N2', 'Ar', None]

print(f"Total combinations: {len(space['temperature']) * len(space['pressure']) * len(space['catalyst']) * len(space['ligand']) * len(space['use_microwave']) * len(space['inert_atmosphere'])}")
```

## Sampling Protocols

### ParamSampler

The base protocol that defines the interface for parameter sampling strategies.

::: molexp.param.ParamSampler
    options:
      show_root_heading: true
      show_source: true

#### Protocol Methods

```python
from molexp.param import ParamSampler, ParamSpace, Param
from typing import Generator

class CustomSampler(ParamSampler):
    """Example custom sampler implementation"""
    
    def __init__(self, strategy: str = "default"):
        self.strategy = strategy
    
    def sample(self, space: ParamSpace) -> Generator[Param, None, None]:
        """Generate parameter samples from the space"""
        # Custom sampling logic here
        keys = list(space.keys())
        values = [space[key] for key in keys]
        
        if self.strategy == "first_only":
            # Only sample first value from each parameter
            yield Param({key: values[i][0] for i, key in enumerate(keys)})
        elif self.strategy == "random_single":
            # Random single sample
            import random
            sampled = {key: random.choice(space[key]) for key in keys}
            yield Param(sampled)
    
    def __call__(self, space: ParamSpace) -> Param:
        """Get single parameter sample"""
        return next(self.sample(space))

# Usage
custom_sampler = CustomSampler("random_single")
sample = custom_sampler(space)
print(f"Random sample: {sample}")
```

## Concrete Samplers

### CartesianSampler

Generates all possible combinations of parameters using Cartesian product.

::: molexp.param.CartesianSampler
    options:
      show_root_heading: true
      show_source: true

#### Basic Usage

```python
from molexp.param import CartesianSampler, ParamSpace

# Define small parameter space
space = ParamSpace({
    'method': ['dft', 'mp2'],
    'basis': ['6-31g', 'cc-pvdz'],
    'charge': [0, 1, -1]
})

# Create Cartesian sampler
sampler = CartesianSampler()

# Generate all combinations
print("All parameter combinations:")
for i, params in enumerate(sampler.sample(space)):
    print(f"  {i+1}: {params}")

# Expected output:
# 1: {'method': 'dft', 'basis': '6-31g', 'charge': 0}
# 2: {'method': 'dft', 'basis': '6-31g', 'charge': 1}
# 3: {'method': 'dft', 'basis': '6-31g', 'charge': -1}
# 4: {'method': 'dft', 'basis': 'cc-pvdz', 'charge': 0}
# ... and so on (total: 2 × 2 × 3 = 12 combinations)
```

#### Practical Applications

```python
# Comprehensive parameter sweep for optimization
optimization_space = ParamSpace({
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'hidden_layers': [1, 2, 3],
    'activation': ['relu', 'tanh']
})

sampler = CartesianSampler()
results = []

print(f"Running {2*3*3*3} experiments...")
for params in sampler.sample(optimization_space):
    # Train model with these parameters
    accuracy = train_model(**params)
    results.append((params, accuracy))
    print(f"Params: {params} -> Accuracy: {accuracy:.3f}")

# Find best parameters
best_params, best_accuracy = max(results, key=lambda x: x[1])
print(f"Best: {best_params} with accuracy {best_accuracy:.3f}")
```

#### Memory Considerations

```python
# For large parameter spaces, use generator to avoid memory issues
large_space = ParamSpace({
    'param1': list(range(100)),
    'param2': list(range(50)),
    'param3': list(range(20))
})

sampler = CartesianSampler()

# Don't do this - will consume too much memory:
# all_samples = list(sampler.sample(large_space))  # 100 × 50 × 20 = 100,000 items

# Instead, process one at a time:
count = 0
for params in sampler.sample(large_space):
    # Process each parameter set
    process_parameters(params)
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} parameter sets...")
```

### RandomSampler

Randomly samples parameter combinations from the space.

::: molexp.param.RandomSampler
    options:
      show_root_heading: true
      show_source: true

#### Basic Usage

```python
from molexp.param import RandomSampler, ParamSpace

# Define parameter space
space = ParamSpace({
    'temperature': list(range(200, 401, 10)),  # 200K to 400K
    'pressure': [0.1 * i for i in range(1, 51)],  # 0.1 to 5.0 atm
    'catalyst': ['Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Au', 'Ag'],
    'solvent': ['toluene', 'dioxane', 'dmf', 'thf', 'dcm', 'acetone']
})

# Create random sampler
sampler = RandomSampler(num_samples=10)

# Generate random samples
print("Random parameter samples:")
for i, params in enumerate(sampler.sample(space)):
    print(f"  {i+1}: {params}")

# Get single random sample
single_sample = sampler(space)
print(f"Single sample: {single_sample}")
```

#### Monte Carlo Simulations

```python
# Monte Carlo parameter exploration
mc_space = ParamSpace({
    'initial_velocity': [v/10 for v in range(1, 101)],  # 0.1 to 10.0
    'angle': list(range(0, 361, 5)),  # 0° to 360° in 5° steps
    'mass': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'friction': [0.01 * i for i in range(1, 21)]  # 0.01 to 0.20
})

# Large random sampling for Monte Carlo
mc_sampler = RandomSampler(num_samples=1000)

distances = []
for params in mc_sampler.sample(mc_space):
    # Simulate projectile motion
    distance = simulate_projectile(**params)
    distances.append(distance)

# Statistical analysis
import statistics
print(f"Mean distance: {statistics.mean(distances):.2f}")
print(f"Std deviation: {statistics.stdev(distances):.2f}")
print(f"Max distance: {max(distances):.2f}")
print(f"Min distance: {min(distances):.2f}")
```

#### Hyperparameter Optimization

```python
# Random search for hyperparameter optimization
hp_space = ParamSpace({
    'learning_rate': [10**i for i in range(-5, -1)],  # 1e-5 to 1e-2
    'dropout_rate': [0.1 * i for i in range(0, 8)],  # 0.0 to 0.7
    'num_layers': list(range(1, 11)),  # 1 to 10 layers
    'layer_size': [2**i for i in range(4, 10)],  # 16 to 512
    'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad'],
    'batch_size': [16, 32, 64, 128, 256]
})

# Random search with limited budget
budget = 50  # Number of experiments
random_sampler = RandomSampler(num_samples=budget)

best_score = 0
best_config = None
experiment_results = []

for i, params in enumerate(random_sampler.sample(hp_space)):
    print(f"Experiment {i+1}/{budget}: {params}")
    
    # Train and evaluate model
    score = train_and_evaluate(**params)
    experiment_results.append((params.copy(), score))
    
    if score > best_score:
        best_score = score
        best_config = params.copy()
        print(f"  New best score: {score:.4f}")
    
    print(f"  Score: {score:.4f}")

print(f"\nBest configuration: {best_config}")
print(f"Best score: {best_score:.4f}")
```

### CombinationSampler

Samples combinations of parameters from the Cartesian product.

::: molexp.param.CombinationSampler
    options:
      show_root_heading: true
      show_source: true

#### Basic Usage

```python
from molexp.param import CombinationSampler, ParamSpace

# Define parameter space
space = ParamSpace({
    'reagent_a': ['compound1', 'compound2', 'compound3'],
    'reagent_b': ['catalyst1', 'catalyst2'],
    'solvent': ['dmf', 'thf', 'toluene'],
    'temperature': [80, 100, 120]
})

# Sample 2-combinations from all possible combinations
sampler = CombinationSampler(r=2)

print("2-combinations of parameter sets:")
for i, params in enumerate(sampler.sample(space)):
    print(f"  {i+1}: {params}")
    if i >= 5:  # Show only first few for brevity
        print("  ...")
        break
```

#### Chemical Reaction Screening

```python
# Screen combinations of reaction conditions
reaction_space = ParamSpace({
    'catalyst': ['Pd(PPh3)4', 'Pd(dppf)Cl2', 'Pd(OAc)2'],
    'base': ['K2CO3', 'Cs2CO3', 'DBU'],
    'solvent': ['DMF', 'dioxane', 'toluene'],
    'temperature': [80, 100, 120],
    'time': [2, 6, 12]  # hours
})

# Test combinations of 3 parameters at a time
combo_sampler = CombinationSampler(r=3)

reaction_results = []
for combo_params in combo_sampler.sample(reaction_space):
    # Run reaction with this combination
    yield_percent = run_reaction(**combo_params)
    reaction_results.append((combo_params, yield_percent))
    
    print(f"Conditions: {combo_params}")
    print(f"Yield: {yield_percent}%\n")

# Find best combinations
best_reactions = sorted(reaction_results, key=lambda x: x[1], reverse=True)[:5]
print("Top 5 reaction conditions:")
for i, (params, yield_val) in enumerate(best_reactions):
    print(f"{i+1}. {params} -> {yield_val}% yield")
```

## Advanced Usage Examples

### Parameter Space Validation

```python
from molexp.param import ParamSpace, CartesianSampler

def validate_parameters(params: dict) -> bool:
    """Validate parameter combinations for physical constraints"""
    temp = params.get('temperature', 298)
    pressure = params.get('pressure', 1.0)
    
    # Temperature-pressure constraints
    if temp > 373 and pressure < 1.0:
        return False  # High temp requires high pressure
    
    # Solvent-temperature constraints
    solvent = params.get('solvent', 'water')
    if solvent == 'water' and temp > 373:
        return False  # Water boils at 373K at 1 atm
    
    return True

# Define parameter space
space = ParamSpace({
    'temperature': [298, 323, 373, 423, 473],
    'pressure': [0.5, 1.0, 2.0, 5.0],
    'solvent': ['water', 'ethanol', 'toluene']
})

# Filter valid combinations
sampler = CartesianSampler()
valid_params = []

for params in sampler.sample(space):
    if validate_parameters(params):
        valid_params.append(params)

print(f"Total combinations: {3*4*5} = {3*4*5}")
print(f"Valid combinations: {len(valid_params)}")
print("\nValid parameter sets:")
for params in valid_params[:5]:  # Show first 5
    print(f"  {params}")
```

### Multi-Stage Parameter Sampling

```python
# Two-stage parameter optimization
# Stage 1: Coarse grid search
coarse_space = ParamSpace({
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 128, 512],
    'hidden_size': [64, 256, 1024]
})

# Stage 1: Find best region
stage1_sampler = CartesianSampler()
stage1_results = []

for params in stage1_sampler.sample(coarse_space):
    score = quick_evaluate(**params)
    stage1_results.append((params, score))

# Find best parameters from stage 1
best_coarse = max(stage1_results, key=lambda x: x[1])[0]
print(f"Best coarse parameters: {best_coarse}")

# Stage 2: Fine-tune around best region
fine_space = ParamSpace({
    'learning_rate': [best_coarse['learning_rate'] * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]],
    'batch_size': [best_coarse['batch_size']],  # Keep best batch size
    'hidden_size': [best_coarse['hidden_size'] + offset for offset in [-64, -32, 0, 32, 64]]
})

# Stage 2: Fine sampling
stage2_sampler = RandomSampler(num_samples=20)
stage2_results = []

for params in stage2_sampler.sample(fine_space):
    score = detailed_evaluate(**params)
    stage2_results.append((params, score))

# Final best parameters
final_best = max(stage2_results, key=lambda x: x[1])
print(f"Final best parameters: {final_best[0]}")
print(f"Final best score: {final_best[1]:.4f}")
```

### Integration with Experimental Design

```python
# Design of Experiments (DoE) using parameter sampling
doe_space = ParamSpace({
    'factor_a': [-1, 0, 1],  # Coded levels
    'factor_b': [-1, 0, 1],
    'factor_c': [-1, 0, 1],
    'factor_d': [-1, 0, 1]
})

# Full factorial design
factorial_sampler = CartesianSampler()

# Generate design matrix
design_matrix = []
responses = []

for params in factorial_sampler.sample(doe_space):
    # Convert coded levels to actual values
    actual_params = {
        'temperature': 300 + params['factor_a'] * 50,  # 250, 300, 350
        'pressure': 1.0 + params['factor_b'] * 0.5,   # 0.5, 1.0, 1.5
        'concentration': 0.5 + params['factor_c'] * 0.3,  # 0.2, 0.5, 0.8
        'ph': 7.0 + params['factor_d'] * 1.0          # 6.0, 7.0, 8.0
    }
    
    design_matrix.append(list(params.values()))
    
    # Run experiment
    response = run_experiment(**actual_params)
    responses.append(response)
    
    print(f"Run: {params} -> Response: {response:.3f}")

# Analyze results (simple main effects)
import numpy as np
design_array = np.array(design_matrix)
response_array = np.array(responses)

# Calculate main effects
factors = ['factor_a', 'factor_b', 'factor_c', 'factor_d']
for i, factor in enumerate(factors):
    high_avg = np.mean(response_array[design_array[:, i] == 1])
    low_avg = np.mean(response_array[design_array[:, i] == -1])
    effect = high_avg - low_avg
    print(f"{factor} main effect: {effect:.3f}")
```

## Best Practices

### 1. Parameter Space Design

```python
# Good: Structured parameter space
good_space = ParamSpace({
    'temperature': list(range(250, 401, 25)),  # Regular intervals
    'pressure': [0.5, 1.0, 1.5, 2.0, 2.5],   # Reasonable range
    'catalyst': ['Pd', 'Pt', 'Ru'],           # Discrete choices
    'use_microwave': [True, False]            # Boolean parameter
})

# Avoid: Irregular or too sparse/dense spacing
avoid_space = ParamSpace({
    'temperature': [250, 267, 298, 399, 400],  # Irregular spacing
    'pressure': [i*0.01 for i in range(1, 501)],  # Too many values
    'catalyst': ['Pd'],                        # Only one choice
})
```

### 2. Sampler Selection

```python
# Use CartesianSampler for:
# - Small parameter spaces
# - Comprehensive coverage needed
# - Deterministic results required

small_space = ParamSpace({'a': [1, 2], 'b': [3, 4]})
complete_sampler = CartesianSampler()  # 4 combinations

# Use RandomSampler for:
# - Large parameter spaces
# - Limited computational budget
# - Exploratory studies

large_space = ParamSpace({
    'param' + str(i): list(range(10)) for i in range(10)
})
budget_sampler = RandomSampler(num_samples=100)  # From 10^10 possibilities

# Use CombinationSampler for:
# - Interaction studies
# - Screening designs
# - Combinatorial optimization

interaction_sampler = CombinationSampler(r=3)  # 3-way interactions
```

### 3. Memory Management

```python
# For large parameter spaces, avoid loading all samples into memory
def process_large_space(space, sampler, processor_func):
    """Process large parameter spaces efficiently"""
    batch_size = 100
    batch = []
    
    for params in sampler.sample(space):
        batch.append(params)
        
        if len(batch) >= batch_size:
            # Process batch
            results = [processor_func(p) for p in batch]
            yield results
            batch = []
    
    # Process remaining
    if batch:
        results = [processor_func(p) for p in batch]
        yield results

# Usage
huge_space = ParamSpace({f'p{i}': list(range(100)) for i in range(5)})
sampler = RandomSampler(num_samples=10000)

for batch_results in process_large_space(huge_space, sampler, my_processor):
    # Handle batch results
    process_batch_results(batch_results)
```

This completes the comprehensive API documentation for the parameters system. The documentation covers all classes, methods, practical examples, and best practices for using MolExp's parameter management system.
