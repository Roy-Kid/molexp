# Basic Usage

This example demonstrates the fundamental concepts of MolExp through a simple computational workflow.

## Overview

We'll create a basic workflow that:
1. Prepares input data
2. Runs a simple calculation
3. Analyzes the results

## Code Example

```python
from molexp import LocalTask, TaskGraph, Executor

# Define a simple calculation function
def calculate_energy(atoms, method='dft'):
    """Simulate energy calculation for molecular system."""
    import time
    time.sleep(0.1)  # Simulate computation time
    
    # Mock calculation based on atom count
    base_energy = -100.0 * len(atoms)
    method_factor = {'dft': 1.0, 'mp2': 1.1, 'ccsd': 1.2}
    
    return base_energy * method_factor.get(method, 1.0)

def analyze_results(energy, threshold=-1000.0):
    """Analyze calculation results."""
    stability = "stable" if energy < threshold else "unstable"
    return {
        'energy': energy,
        'stability': stability,
        'binding_energy': abs(energy) / 10  # Mock binding energy
    }

# Create tasks
preparation_task = LocalTask(
    name="prepare_system",
    func=lambda: ['C', 'C', 'H', 'H', 'H', 'H'],  # Simple molecule
    outputs=['atoms']
)

calculation_task = LocalTask(
    name="energy_calculation",
    func=calculate_energy,
    inputs={'method': 'dft'},
    outputs=['energy']
)

analysis_task = LocalTask(
    name="result_analysis", 
    func=analyze_results,
    outputs=['analysis']
)

# Build workflow graph
workflow = TaskGraph()
workflow.add_task(preparation_task)
workflow.add_task(calculation_task, dependencies=[preparation_task])
workflow.add_task(analysis_task, dependencies=[calculation_task])

# Execute workflow
executor = Executor()
results = executor.execute(workflow)

# Display results
for task_name, result in results.items():
    print(f"{task_name}: {result}")
```

## Expected Output

```
prepare_system: ['C', 'C', 'H', 'H', 'H', 'H']
energy_calculation: -600.0
result_analysis: {'energy': -600.0, 'stability': 'stable', 'binding_energy': 60.0}
```

## Key Concepts Demonstrated

1. **Task Creation**: Using `LocalTask` for Python function execution
2. **Dependency Management**: Linking tasks with dependencies
3. **Workflow Execution**: Using `TaskGraph` and `Executor`
4. **Result Handling**: Accessing task outputs

## Next Steps

- Try modifying the calculation method parameter
- Add more complex analysis functions
- Experiment with different molecular systems
- Explore parallel execution with multiple tasks

See other examples for more advanced features like parameter studies, remote execution, and Hamilton integration.
