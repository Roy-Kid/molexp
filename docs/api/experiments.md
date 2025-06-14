# Experiments

This section provides API documentation for MolExp's experiment management system.

## Experiment

The main class for managing scientific experiments and parameter studies.

::: molexp.experiment.Experiment
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - set_parameter_space
        - set_sampler
        - add_task
        - remove_task
        - configure
        - run
        - pause
        - resume
        - cancel
        - get_results
        - get_status
        - save
        - load
        - export_results

### Basic Experiment Setup

```python
from molexp import Experiment, ParameterSpace, FloatParameter

# Create experiment
experiment = Experiment(
    name="molecular_dynamics_study",
    description="Temperature and pressure effects on MD simulation",
    version="1.0",
    author="Researcher Name"
)

# Define parameter space
param_space = ParameterSpace({
    'temperature': FloatParameter(min=250, max=350, step=25, unit='K'),
    'pressure': FloatParameter(min=0.5, max=2.0, step=0.5, unit='atm')
})

experiment.set_parameter_space(param_space)

# Add computational tasks
from molexp import LocalTask

def md_simulation(temperature, pressure):
    # Simulation logic
    return {'energy': -1000 + temperature*2 + pressure*10}

simulation_task = LocalTask(
    name="md_simulation",
    func=md_simulation,
    outputs=['simulation_results']
)

experiment.add_task(simulation_task)

# Configure and run
experiment.configure(max_parallel=4, save_intermediates=True)
results = experiment.run()
```

### Advanced Configuration

```python
# Advanced experiment configuration
experiment.configure(
    max_parallel=8,                    # Maximum parallel executions
    batch_size=10,                     # Batch size for parameter sets
    timeout=3600,                      # Timeout per parameter set (seconds)
    retry_failed=True,                 # Retry failed parameter sets
    max_retries=3,                     # Maximum retry attempts
    save_intermediates=True,           # Save intermediate results
    output_dir="experiment_results/",  # Output directory
    checkpoint_interval=50,            # Checkpoint every N parameter sets
    resource_limits={                  # Resource limits
        'memory_mb': 4096,
        'cpu_cores': 4
    },
    error_handling='continue',         # Continue on individual failures
    progress_reporting=True,           # Enable progress reporting
    result_caching=True               # Cache results to avoid recomputation
)
```

## Parameter Sampling

### Sampling Strategies

```python
from molexp import GridSampler, RandomSampler, LatinHypercubeSampler

# Grid sampling (full factorial)
grid_sampler = GridSampler()
experiment.set_sampler(grid_sampler)

# Random sampling
random_sampler = RandomSampler(
    n_samples=100,
    seed=42,
    stratified=True
)
experiment.set_sampler(random_sampler)

# Latin Hypercube sampling
lhs_sampler = LatinHypercubeSampler(
    n_samples=64,
    seed=123,
    criterion='maximin'
)
experiment.set_sampler(lhs_sampler)
```

### Custom Sampling

```python
from molexp.sampling import BaseSampler

class CustomSampler(BaseSampler):
    """Custom sampling strategy."""
    
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param
    
    def sample(self, parameter_space, n_samples=None):
        """Generate parameter samples."""
        samples = []
        
        # Custom sampling logic
        for i in range(n_samples or 10):
            sample = {}
            for param_name, param in parameter_space.parameters.items():
                # Custom sampling for each parameter
                sample[param_name] = self._sample_parameter(param)
            samples.append(sample)
        
        return samples
    
    def _sample_parameter(self, parameter):
        """Sample a single parameter."""
        # Implementation depends on parameter type
        if hasattr(parameter, 'sample'):
            return parameter.sample()
        else:
            # Fallback sampling
            return parameter.default_value

# Use custom sampler
experiment.set_sampler(CustomSampler(custom_param="value"))
```

## Result Management

### Result Collection

```python
# Run experiment and collect results
results = experiment.run()

# Access results by parameter set
for result in results:
    print(f"Parameters: {result.parameters}")
    print(f"Outputs: {result.outputs}")
    print(f"Status: {result.status}")
    print(f"Execution time: {result.execution_time}")
    print(f"Error (if any): {result.error}")
    print("---")

# Filter results
successful_results = [r for r in results if r.status == 'completed']
failed_results = [r for r in results if r.status == 'failed']

print(f"Successful: {len(successful_results)}")
print(f"Failed: {len(failed_results)}")
```

### Result Analysis

```python
from molexp.analysis import ResultAnalyzer

# Create analyzer
analyzer = ResultAnalyzer(results)

# Basic statistics
stats = analyzer.compute_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average execution time: {stats['avg_execution_time']:.2f}s")
print(f"Parameter coverage: {stats['parameter_coverage']:.2%}")

# Parameter correlations
correlations = analyzer.compute_correlations()
print("Parameter correlations:")
for (param1, param2), correlation in correlations.items():
    print(f"  {param1} vs {param2}: {correlation:.3f}")

# Sensitivity analysis
sensitivity = analyzer.sensitivity_analysis(target_variable='energy')
print("Sensitivity indices:")
for param, sensitivity_idx in sensitivity.items():
    print(f"  {param}: {sensitivity_idx:.3f}")
```

### Result Export

```python
# Export results in various formats
experiment.export_results(
    filepath="results.csv",
    format="csv",
    include_metadata=True,
    flatten_nested=True
)

experiment.export_results(
    filepath="results.json",
    format="json",
    pretty_print=True,
    include_parameters=True,
    include_outputs=True
)

experiment.export_results(
    filepath="results.xlsx",
    format="excel",
    separate_sheets=True,
    include_charts=True
)

# Export to database
experiment.export_results(
    connection_string="postgresql://user:pass@localhost/experiments",
    format="database",
    table_name="md_study_results",
    create_tables=True
)
```

## Experiment Persistence

### Save and Load

```python
# Save experiment state
experiment.save("experiment_checkpoint.pkl")

# Save experiment configuration only
experiment.save_config("experiment_config.yaml")

# Load complete experiment
loaded_experiment = Experiment.load("experiment_checkpoint.pkl")

# Load experiment from configuration
experiment_from_config = Experiment.from_config("experiment_config.yaml")

# Resume from checkpoint
resumed_experiment = Experiment.load("experiment_checkpoint.pkl")
additional_results = resumed_experiment.run(resume=True)
```

### Version Control

```python
# Experiment versioning
experiment.set_version("2.0")
experiment.add_version_notes("Added new analysis tasks")

# Create experiment branch
branch_experiment = experiment.create_branch("temperature_sensitivity")
branch_experiment.modify_parameter_space(
    {'temperature': FloatParameter(min=200, max=300, step=10)}
)

# Merge results from branch
experiment.merge_branch(branch_experiment)
```

## Multi-Experiment Management

### ExperimentSuite

```python
from molexp import ExperimentSuite

# Create experiment suite
suite = ExperimentSuite(
    name="comprehensive_md_study",
    description="Multi-scale molecular dynamics study"
)

# Add experiments
suite.add_experiment(temperature_study)
suite.add_experiment(pressure_study)
suite.add_experiment(combined_study)

# Configure suite execution
suite.configure(
    execution_order='parallel',  # 'sequential', 'parallel', 'adaptive'
    max_concurrent_experiments=2,
    shared_resources=True,
    result_aggregation=True
)

# Run entire suite
suite_results = suite.run()

# Analyze suite results
suite_analysis = suite.analyze_results()
suite.generate_report("comprehensive_study_report.html")
```

### Experiment Comparison

```python
from molexp import ExperimentComparator

# Compare experiments
comparator = ExperimentComparator([experiment1, experiment2, experiment3])

# Statistical comparison
comparison_stats = comparator.statistical_comparison()
print("Experiment comparison:")
for metric, values in comparison_stats.items():
    print(f"  {metric}: {values}")

# Performance comparison
performance_comparison = comparator.performance_comparison()
print("Performance comparison:")
for exp_name, perf_metrics in performance_comparison.items():
    print(f"  {exp_name}: {perf_metrics}")

# Generate comparison report
comparator.generate_comparison_report("experiment_comparison.html")
```

## Real-time Monitoring

### Progress Tracking

```python
from molexp import ProgressTracker

# Create progress tracker
progress_tracker = ProgressTracker(
    update_interval=5.0,              # Update every 5 seconds
    display_format='detailed',        # 'simple', 'detailed', 'custom'
    enable_email_notifications=True,
    email_config={
        'smtp_server': 'smtp.example.com',
        'recipient': 'researcher@example.com'
    }
)

# Use with experiment
experiment.set_progress_tracker(progress_tracker)

# Run with progress monitoring
results = experiment.run()
```

### Real-time Dashboard

```python
from molexp import ExperimentDashboard

# Create dashboard
dashboard = ExperimentDashboard(
    port=8080,
    update_interval=2.0,
    enable_real_time_plots=True,
    plot_parameters=['temperature', 'pressure'],
    plot_outputs=['energy', 'volume']
)

# Start dashboard
dashboard.start()
print("Dashboard available at http://localhost:8080")

# Run experiment with dashboard monitoring
experiment.set_dashboard(dashboard)
results = experiment.run()

# Stop dashboard
dashboard.stop()
```

## Advanced Features

### Adaptive Experiments

```python
from molexp import AdaptiveExperiment

# Create adaptive experiment
adaptive_exp = AdaptiveExperiment(
    name="adaptive_optimization",
    objective_function=lambda results: results.get('energy', float('inf')),
    optimization_direction='minimize',
    initial_samples=20,
    max_iterations=10,
    convergence_criteria={'tolerance': 1e-3, 'patience': 3}
)

# Set parameter space
adaptive_exp.set_parameter_space(param_space)

# Add acquisition function
from molexp.adaptive import ExpectedImprovement
adaptive_exp.set_acquisition_function(
    ExpectedImprovement(exploration_weight=0.1)
)

# Run adaptive experiment
adaptive_results = adaptive_exp.run()

# Get optimization history
optimization_history = adaptive_exp.get_optimization_history()
best_parameters = adaptive_exp.get_best_parameters()
```

### Multi-Objective Experiments

```python
from molexp import MultiObjectiveExperiment

# Create multi-objective experiment
mo_experiment = MultiObjectiveExperiment(
    name="multi_objective_optimization",
    objectives={
        'energy': {'direction': 'minimize', 'weight': 0.6},
        'stability': {'direction': 'maximize', 'weight': 0.4}
    }
)

# Set parameter space and tasks
mo_experiment.set_parameter_space(param_space)
mo_experiment.add_task(simulation_task)

# Run multi-objective optimization
mo_results = mo_experiment.run()

# Analyze Pareto frontier
pareto_frontier = mo_experiment.get_pareto_frontier()
mo_experiment.plot_pareto_frontier(save_path="pareto_frontier.png")
```

### Experiment Templates

```python
from molexp import ExperimentTemplate

# Create reusable experiment template
md_template = ExperimentTemplate(
    name="molecular_dynamics_template",
    description="Standard MD simulation template",
    parameter_definitions={
        'temperature': {'type': 'float', 'min': 200, 'max': 400, 'unit': 'K'},
        'pressure': {'type': 'float', 'min': 0.1, 'max': 5.0, 'unit': 'atm'},
        'timesteps': {'type': 'int', 'min': 1000, 'max': 100000}
    },
    task_definitions=[
        {
            'name': 'equilibration',
            'type': 'shell',
            'command': 'mdrun -equilibrate -temp {temperature} -press {pressure}'
        },
        {
            'name': 'production',
            'type': 'shell', 
            'command': 'mdrun -production -steps {timesteps}',
            'dependencies': ['equilibration']
        }
    ],
    default_configuration={
        'max_parallel': 4,
        'timeout': 3600,
        'retry_failed': True
    }
)

# Create experiment from template
experiment = md_template.create_experiment(
    name="specific_md_study",
    parameter_overrides={
        'temperature': {'min': 280, 'max': 320}
    },
    config_overrides={
        'max_parallel': 8
    }
)
```

## Integration with External Tools

### Database Integration

```python
from molexp.database import ExperimentDatabase

# Connect to database
db = ExperimentDatabase(
    connection_string="postgresql://user:pass@localhost/experiments",
    schema="molexp"
)

# Save experiment to database
experiment_id = db.save_experiment(experiment)

# Load experiment from database
loaded_experiment = db.load_experiment(experiment_id)

# Query experiments
matching_experiments = db.query_experiments(
    filters={'parameter.temperature': {'$gte': 300}},
    sort_by='creation_date',
    limit=10
)
```

### Cloud Storage Integration

```python
from molexp.cloud import CloudStorage

# Configure cloud storage
cloud_storage = CloudStorage(
    provider='aws',  # 'aws', 'gcp', 'azure'
    bucket='molexp-experiments',
    credentials_file='~/.aws/credentials'
)

# Save experiment to cloud
cloud_storage.save_experiment(experiment, path="experiments/md_study/")

# Load experiment from cloud
loaded_experiment = cloud_storage.load_experiment("experiments/md_study/")

# Sync local and cloud experiments
cloud_storage.sync_experiments(local_dir="./experiments/")
```

This comprehensive documentation covers MolExp's experiment management system, providing tools for designing, executing, and analyzing complex scientific parameter studies.
