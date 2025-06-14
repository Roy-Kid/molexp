# Parameter Exploration

This example demonstrates systematic parameter exploration using MolExp's parameter study capabilities.

## Overview

Parameter exploration is essential for:
- Optimization studies
- Sensitivity analysis  
- Design space exploration
- Method validation
- Uncertainty quantification

## Basic Parameter Study

```python
from molexp import (
    Experiment, ParameterSpace, FloatParameter, IntParameter, 
    CategoricalParameter, LocalTask, GridSampler
)
import numpy as np
import matplotlib.pyplot as plt

# Define the objective function to study
def molecular_dynamics_simulation(temperature, pressure, timesteps, ensemble):
    """Simulate molecular dynamics with given parameters."""
    import random
    random.seed(int(temperature * pressure * timesteps))
    
    # Mock MD simulation
    kinetic_energy = 1.5 * 8.314 * temperature  # 3/2 * R * T per particle
    potential_energy = -1000 + pressure * 10 + random.gauss(0, 50)
    
    # Simulation quality depends on timesteps
    convergence_factor = min(1.0, timesteps / 10000)
    noise_level = 100 * (1 - convergence_factor)
    
    total_energy = kinetic_energy + potential_energy + random.gauss(0, noise_level)
    
    # Ensemble effects
    ensemble_corrections = {
        'NVE': 0,
        'NVT': -10,
        'NPT': -20,
        'NσT': -15
    }
    
    final_energy = total_energy + ensemble_corrections.get(ensemble, 0)
    
    return {
        'total_energy': final_energy,
        'kinetic_energy': kinetic_energy,
        'potential_energy': potential_energy,
        'convergence': convergence_factor,
        'ensemble': ensemble
    }

# Define parameter space
param_space = ParameterSpace({
    'temperature': FloatParameter(
        name='temperature',
        min=250.0,      # 250 K
        max=350.0,      # 350 K  
        step=25.0,
        unit='K',
        description='Simulation temperature'
    ),
    'pressure': FloatParameter(
        name='pressure',
        min=0.5,        # 0.5 atm
        max=2.0,        # 2.0 atm
        step=0.5,
        unit='atm', 
        description='System pressure'
    ),
    'timesteps': IntParameter(
        name='timesteps',
        min=5000,
        max=20000,
        step=5000,
        description='Number of MD timesteps'
    ),
    'ensemble': CategoricalParameter(
        name='ensemble',
        choices=['NVE', 'NVT', 'NPT', 'NσT'],
        description='MD ensemble'
    )
})

# Create experiment
experiment = Experiment(
    name="md_parameter_study", 
    description="Molecular dynamics parameter exploration"
)

experiment.set_parameter_space(param_space)
experiment.set_sampler(GridSampler())  # Full factorial design

# Add simulation task
simulation_task = LocalTask(
    name="md_simulation",
    func=molecular_dynamics_simulation,
    outputs=['simulation_results']
)

experiment.add_task(simulation_task)

# Configure parallel execution
experiment.configure(
    max_parallel=4,
    save_intermediates=True,
    output_dir="md_parameter_study_results/"
)

print("Starting MD parameter study...")
print(f"Parameter space size: {len(param_space.sample_grid())}")

# Execute experiment
results = experiment.run()

print(f"Completed {len(results)} simulations")
```

## Advanced Sampling Strategies

```python
from molexp import RandomSampler, LatinHypercubeSampler, SobolSampler

# Latin Hypercube Sampling for better space coverage
def latin_hypercube_study():
    """Example using Latin Hypercube Sampling."""
    
    # Define continuous parameter space
    continuous_space = ParameterSpace({
        'temperature': FloatParameter(min=200, max=400, unit='K'),
        'pressure': FloatParameter(min=0.1, max=5.0, unit='atm'),
        'concentration': FloatParameter(min=0.01, max=1.0, unit='M'),
        'ph': FloatParameter(min=4.0, max=10.0)
    })
    
    # Use Latin Hypercube Sampling
    lhs_sampler = LatinHypercubeSampler(seed=42)
    
    experiment = Experiment("lhs_study")
    experiment.set_parameter_space(continuous_space)
    experiment.set_sampler(lhs_sampler)
    
    # Sample 100 points for efficient space coverage
    parameter_sets = lhs_sampler.sample(continuous_space, n_samples=100)
    
    return experiment, parameter_sets

# Sobol sequence for quasi-random sampling
def sobol_study():
    """Example using Sobol sequence sampling."""
    
    sobol_sampler = SobolSampler()
    
    experiment = Experiment("sobol_study")
    experiment.set_parameter_space(param_space)
    experiment.set_sampler(sobol_sampler)
    
    # Sobol sequences work best with powers of 2
    parameter_sets = sobol_sampler.sample(param_space, n_samples=64)
    
    return experiment, parameter_sets

# Random sampling with constraints
def constrained_sampling():
    """Example with parameter constraints."""
    
    def constraint_function(params):
        # Temperature and pressure must be physically reasonable
        if params['temperature'] < 273 and params['pressure'] > 1:
            return False  # No high pressure at very low temperature
        
        # Timesteps should be sufficient for high temperatures
        if params['temperature'] > 300 and params['timesteps'] < 10000:
            return False
            
        return True
    
    constrained_sampler = ConstrainedSampler(
        base_sampler=RandomSampler(seed=123),
        constraints=[constraint_function],
        max_attempts=1000
    )
    
    experiment = Experiment("constrained_study")
    experiment.set_parameter_space(param_space)
    experiment.set_sampler(constrained_sampler)
    
    return experiment
```

## Multi-Objective Parameter Studies

```python
def multi_objective_study():
    """Parameter study with multiple objectives."""
    
    def multi_objective_simulation(temp, pressure, method):
        """Simulation with multiple objectives to optimize."""
        import random
        random.seed(int(temp * pressure * 1000))
        
        # Objective 1: Energy (minimize)
        energy = -1000 + temp * 2 + pressure * 10 + random.gauss(0, 20)
        
        # Objective 2: Accuracy (maximize) 
        accuracy = 0.95 - abs(temp - 300) * 0.001 - abs(pressure - 1) * 0.01
        accuracy += random.gauss(0, 0.02)
        accuracy = max(0, min(1, accuracy))
        
        # Objective 3: Computational cost (minimize)
        method_costs = {'fast': 1, 'medium': 3, 'accurate': 10}
        cost = method_costs.get(method, 5) * (1 + temp/1000 + pressure/10)
        
        return {
            'energy': energy,
            'accuracy': accuracy, 
            'cost': cost,
            'efficiency': accuracy / cost  # Composite metric
        }
    
    # Define parameter space
    multi_obj_space = ParameterSpace({
        'temp': FloatParameter(min=250, max=350, step=20),
        'pressure': FloatParameter(min=0.5, max=2.0, step=0.3),
        'method': CategoricalParameter(choices=['fast', 'medium', 'accurate'])
    })
    
    # Create experiment
    experiment = Experiment("multi_objective_study")
    experiment.set_parameter_space(multi_obj_space)
    
    # Add simulation task
    task = LocalTask(
        name="multi_obj_simulation",
        func=multi_objective_simulation,
        outputs=['objectives']
    )
    experiment.add_task(task)
    
    # Execute study
    results = experiment.run()
    
    # Analyze multi-objective results
    analyze_pareto_frontier(results)
    
    return results

def analyze_pareto_frontier(results):
    """Analyze Pareto frontier for multi-objective optimization."""
    import numpy as np
    
    # Extract objectives
    energies = [r.outputs['objectives']['energy'] for r in results]
    accuracies = [r.outputs['objectives']['accuracy'] for r in results]
    costs = [r.outputs['objectives']['cost'] for r in results]
    
    # Find Pareto frontier (minimize energy and cost, maximize accuracy)
    pareto_points = []
    
    for i, result in enumerate(results):
        is_pareto = True
        
        for j, other_result in enumerate(results):
            if i == j:
                continue
                
            # Check if other point dominates this point
            other_better_energy = other_result.outputs['objectives']['energy'] <= result.outputs['objectives']['energy']
            other_better_accuracy = other_result.outputs['objectives']['accuracy'] >= result.outputs['objectives']['accuracy']
            other_better_cost = other_result.outputs['objectives']['cost'] <= result.outputs['objectives']['cost']
            
            # Strict domination in at least one objective
            other_strictly_better = (
                other_result.outputs['objectives']['energy'] < result.outputs['objectives']['energy'] or
                other_result.outputs['objectives']['accuracy'] > result.outputs['objectives']['accuracy'] or
                other_result.outputs['objectives']['cost'] < result.outputs['objectives']['cost']
            )
            
            if other_better_energy and other_better_accuracy and other_better_cost and other_strictly_better:
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(result)
    
    print(f"Found {len(pareto_points)} Pareto optimal points out of {len(results)} total points")
    
    # Display best solutions
    for i, point in enumerate(pareto_points[:5]):  # Show top 5
        obj = point.outputs['objectives']
        print(f"Pareto point {i+1}:")
        print(f"  Parameters: {point.parameters}")
        print(f"  Energy: {obj['energy']:.2f}")
        print(f"  Accuracy: {obj['accuracy']:.3f}")  
        print(f"  Cost: {obj['cost']:.1f}")
        print(f"  Efficiency: {obj['efficiency']:.4f}")
        print()
    
    return pareto_points
```

## Adaptive Parameter Exploration

```python
from molexp import AdaptiveSampler

def adaptive_optimization():
    """Example of adaptive parameter exploration."""
    
    def objective_function(params, results):
        """Objective function for adaptive sampling."""
        if not results:
            return float('inf')  # No results yet
        
        # Minimize energy while maintaining accuracy > 0.8
        obj = results.get('simulation_results', {})
        energy = obj.get('total_energy', float('inf'))
        accuracy = obj.get('convergence', 0)
        
        if accuracy < 0.8:
            return float('inf')  # Constraint violation
        
        return energy
    
    # Create adaptive sampler
    adaptive_sampler = AdaptiveSampler(
        objective=objective_function,
        strategy='minimize',
        acquisition_function='expected_improvement',
        n_initial=20,  # Initial random sampling
        n_iterations=10  # Adaptive iterations
    )
    
    experiment = Experiment("adaptive_optimization")
    experiment.set_parameter_space(param_space)
    experiment.set_sampler(adaptive_sampler)
    
    # Add simulation task
    simulation_task = LocalTask(
        name="adaptive_simulation",
        func=molecular_dynamics_simulation,
        outputs=['simulation_results']
    )
    experiment.add_task(simulation_task)
    
    # Run adaptive optimization
    print("Starting adaptive optimization...")
    
    for iteration in range(10):
        print(f"Iteration {iteration + 1}")
        
        # Get next batch of parameters to evaluate
        next_params = adaptive_sampler.suggest_next_batch(n_suggestions=5)
        
        # Evaluate parameters
        batch_results = []
        for params in next_params:
            result = simulation_task.execute(inputs=params)
            batch_results.append((params, result))
        
        # Update sampler with results
        adaptive_sampler.update_with_results(batch_results)
        
        # Get current best
        best_params, best_value = adaptive_sampler.get_best()
        print(f"  Current best: {best_value:.2f} at {best_params}")
    
    return adaptive_sampler.get_all_results()
```

## Sensitivity Analysis

```python
def sensitivity_analysis():
    """Perform sensitivity analysis on parameters."""
    
    # Sobol sensitivity analysis
    def sobol_sensitivity_study():
        """Sobol indices for global sensitivity analysis."""
        from SALib.sample import sobol
        from SALib.analyze import sobol as sobol_analyze
        
        # Define problem for SALib
        problem = {
            'num_vars': 3,
            'names': ['temperature', 'pressure', 'timesteps'],
            'bounds': [[250, 350], [0.5, 2.0], [5000, 20000]]
        }
        
        # Generate samples
        param_values = sobol.sample(problem, 1024)
        
        # Run simulations
        outputs = []
        for params in param_values:
            temp, pressure, timesteps = params
            result = molecular_dynamics_simulation(
                temperature=temp,
                pressure=pressure, 
                timesteps=int(timesteps),
                ensemble='NVT'
            )
            outputs.append(result['total_energy'])
        
        # Analyze sensitivity
        sobol_indices = sobol_analyze.analyze(problem, np.array(outputs))
        
        print("Sobol Sensitivity Analysis:")
        print("First-order indices (main effects):")
        for i, name in enumerate(problem['names']):
            print(f"  {name}: {sobol_indices['S1'][i]:.3f}")
        
        print("\nTotal indices (including interactions):")
        for i, name in enumerate(problem['names']):
            print(f"  {name}: {sobol_indices['ST'][i]:.3f}")
        
        return sobol_indices
    
    # Morris screening
    def morris_screening():
        """Morris method for screening important parameters."""
        from SALib.sample import morris
        from SALib.analyze import morris as morris_analyze
        
        problem = {
            'num_vars': 4,
            'names': ['temperature', 'pressure', 'timesteps', 'ensemble_code'],
            'bounds': [[250, 350], [0.5, 2.0], [5000, 20000], [0, 3]]
        }
        
        # Generate sample
        param_values = morris.sample(problem, N=100)
        
        # Run simulations
        outputs = []
        ensembles = ['NVE', 'NVT', 'NPT', 'NσT']
        
        for params in param_values:
            temp, pressure, timesteps, ensemble_idx = params
            ensemble = ensembles[int(ensemble_idx)]
            
            result = molecular_dynamics_simulation(
                temperature=temp,
                pressure=pressure,
                timesteps=int(timesteps),
                ensemble=ensemble
            )
            outputs.append(result['total_energy'])
        
        # Analyze
        morris_results = morris_analyze.analyze(problem, param_values, np.array(outputs))
        
        print("Morris Screening Results:")
        for i, name in enumerate(problem['names']):
            mu_star = morris_results['mu_star'][i]
            sigma = morris_results['sigma'][i]
            print(f"  {name}: μ* = {mu_star:.3f}, σ = {sigma:.3f}")
        
        return morris_results
    
    # Run both analyses
    print("Performing sensitivity analysis...\n")
    sobol_results = sobol_sensitivity_study()
    print("\n" + "="*50 + "\n")
    morris_results = morris_screening()
    
    return sobol_results, morris_results
```

## Visualization and Analysis

```python
def visualize_parameter_study_results(results):
    """Create comprehensive visualizations of parameter study results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Extract data for analysis
    data = []
    for result in results:
        row = result.parameters.copy()
        row.update(result.outputs['simulation_results'])
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create visualization dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Study Results Dashboard', fontsize=16)
    
    # 1. Parameter correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
    axes[0,0].set_title('Parameter Correlations')
    
    # 2. Energy vs Temperature
    sns.scatterplot(data=df, x='temperature', y='total_energy', 
                   hue='ensemble', size='pressure', ax=axes[0,1])
    axes[0,1].set_title('Energy vs Temperature')
    
    # 3. Energy distribution by ensemble
    sns.boxplot(data=df, x='ensemble', y='total_energy', ax=axes[0,2])
    axes[0,2].set_title('Energy Distribution by Ensemble')
    
    # 4. Convergence vs Timesteps
    sns.scatterplot(data=df, x='timesteps', y='convergence',
                   hue='temperature', ax=axes[1,0])
    axes[1,0].set_title('Convergence vs Timesteps')
    
    # 5. 3D surface plot (Temperature vs Pressure vs Energy)
    from mpl_toolkits.mplot3d import Axes3D
    ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Filter for one ensemble for cleaner plot
    nvt_data = df[df['ensemble'] == 'NVT']
    scatter = ax_3d.scatter(nvt_data['temperature'], nvt_data['pressure'], 
                          nvt_data['total_energy'], c=nvt_data['total_energy'], 
                          cmap='viridis')
    ax_3d.set_xlabel('Temperature (K)')
    ax_3d.set_ylabel('Pressure (atm)')
    ax_3d.set_zlabel('Total Energy')
    ax_3d.set_title('Energy Surface (NVT)')
    
    # 6. Pareto frontier (if multi-objective)
    if 'cost' in df.columns:
        pareto_data = identify_pareto_frontier(df, ['total_energy', 'cost'], [True, True])  # minimize both
        axes[1,2].scatter(df['total_energy'], df['cost'], alpha=0.6, label='All points')
        axes[1,2].scatter(pareto_data['total_energy'], pareto_data['cost'], 
                         color='red', s=50, label='Pareto frontier')
        axes[1,2].set_xlabel('Total Energy')
        axes[1,2].set_ylabel('Cost')
        axes[1,2].set_title('Pareto Frontier')
        axes[1,2].legend()
    else:
        # Alternative: parameter importance
        feature_importance = calculate_feature_importance(df, 'total_energy')
        axes[1,2].bar(feature_importance.index, feature_importance.values)
        axes[1,2].set_title('Parameter Importance')
        axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('parameter_study_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def calculate_feature_importance(df, target_col):
    """Calculate feature importance using random forest."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Fit random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False)

def identify_pareto_frontier(df, objectives, minimize):
    """Identify Pareto frontier points."""
    pareto_mask = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            
            # Check if j dominates i
            dominates = True
            for k, obj in enumerate(objectives):
                if minimize[k]:  # minimize objective
                    if df.iloc[j][obj] > df.iloc[i][obj]:
                        dominates = False
                        break
                else:  # maximize objective
                    if df.iloc[j][obj] < df.iloc[i][obj]:
                        dominates = False
                        break
            
            if dominates:
                # Check for strict domination
                strictly_dominates = False
                for k, obj in enumerate(objectives):
                    if minimize[k]:
                        if df.iloc[j][obj] < df.iloc[i][obj]:
                            strictly_dominates = True
                            break
                    else:
                        if df.iloc[j][obj] > df.iloc[i][obj]:
                            strictly_dominates = True
                            break
                
                if strictly_dominates:
                    pareto_mask[i] = False
                    break
    
    return df[pareto_mask]

# Example usage
if __name__ == "__main__":
    # Run the parameter study
    results = run_parameter_study()
    
    # Visualize results
    df = visualize_parameter_study_results(results)
    
    # Perform sensitivity analysis
    sensitivity_results = sensitivity_analysis()
    
    print("Parameter study completed successfully!")
    print(f"Analyzed {len(results)} parameter combinations")
    print("Results saved to 'parameter_study_dashboard.png'")
```

This comprehensive example demonstrates the full power of MolExp for parameter exploration, from basic grid searches to advanced adaptive optimization and sensitivity analysis.
