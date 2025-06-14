# Shell Tasks

This example demonstrates how to use ShellTask to execute command-line tools and scripts as part of your workflow.

## Overview

Shell tasks are perfect for:
- Integrating existing command-line tools
- Running system utilities
- Executing scripts in different languages
- File manipulation and data processing

## Basic Shell Task Usage

```python
from molexp import ShellTask, TaskGraph, Executor

# Simple command execution
basic_task = ShellTask(
    name="list_files",
    command="ls -la /tmp"
)

# Execute the task
executor = Executor()
result = executor.execute_task(basic_task)
print("Directory listing:", result)
```

## Parameterized Commands

```python
# Using input parameters in commands
analysis_task = ShellTask(
    name="analyze_data",
    command="python analyze.py --input {input_file} --output {output_file} --method {method}",
    inputs={
        'input_file': 'data.csv',
        'output_file': 'results.txt', 
        'method': 'statistical'
    }
)

result = analysis_task.execute()
```

## Environment and Working Directory

```python
# Set environment variables and working directory
simulation_task = ShellTask(
    name="run_simulation",
    command="./simulate --config config.json",
    working_dir="/path/to/simulation",
    env={
        'OMP_NUM_THREADS': '4',
        'CUDA_VISIBLE_DEVICES': '0',
        'SIMULATION_MODE': 'production'
    },
    timeout=3600  # 1 hour timeout
)
```

## Complex Pipeline Example

```python
from molexp import ShellTask, LocalTask, TaskGraph
import os

# Create a data processing pipeline
def setup_pipeline():
    """Create a complete data processing pipeline using shell tasks."""
    
    graph = TaskGraph()
    
    # Step 1: Download data
    download_task = ShellTask(
        name="download_data",
        command="wget https://example.com/dataset.tar.gz -O dataset.tar.gz",
        working_dir="/tmp/pipeline"
    )
    
    # Step 2: Extract data  
    extract_task = ShellTask(
        name="extract_data",
        command="tar -xzf dataset.tar.gz",
        working_dir="/tmp/pipeline"
    )
    
    # Step 3: Preprocess data
    preprocess_task = ShellTask(
        name="preprocess",
        command="python preprocess.py --input dataset/ --output processed/",
        working_dir="/tmp/pipeline"
    )
    
    # Step 4: Run analysis
    analysis_task = ShellTask(
        name="analysis",
        command="Rscript analysis.R processed/ results/",
        working_dir="/tmp/pipeline",
        env={'R_LIBS_USER': '/usr/local/lib/R/site-library'}
    )
    
    # Step 5: Generate report
    report_task = ShellTask(
        name="generate_report",
        command="pandoc results/analysis.md -o report.pdf",
        working_dir="/tmp/pipeline"
    )
    
    # Build pipeline
    graph.add_task(download_task)
    graph.add_task(extract_task, dependencies=[download_task])
    graph.add_task(preprocess_task, dependencies=[extract_task])
    graph.add_task(analysis_task, dependencies=[preprocess_task])
    graph.add_task(report_task, dependencies=[analysis_task])
    
    return graph

# Execute pipeline
pipeline = setup_pipeline()
executor = Executor()
results = executor.execute(pipeline)

for task_name, result in results.items():
    print(f"{task_name} completed successfully")
```

## Error Handling and Validation

```python
def create_robust_shell_task():
    """Create a shell task with comprehensive error handling."""
    
    # Validation function
    def validate_output(output):
        if "ERROR" in output or "FAILED" in output:
            raise ValueError(f"Command failed: {output}")
        return output
    
    # Create task with validation
    robust_task = ShellTask(
        name="robust_computation",
        command="python computation.py --data {data_file}",
        inputs={'data_file': 'input.dat'},
        timeout=300,  # 5 minute timeout
        retry_attempts=2,
        output_validator=validate_output
    )
    
    try:
        result = robust_task.execute()
        print("Computation successful:", result)
    except Exception as e:
        print(f"Task failed after retries: {e}")
        # Handle failure (cleanup, notification, etc.)

create_robust_shell_task()
```

## File Operations

```python
# File manipulation tasks
def file_operations_example():
    """Example of common file operations using shell tasks."""
    
    graph = TaskGraph()
    
    # Create directory structure
    setup_task = ShellTask(
        name="setup_directories", 
        command="mkdir -p data/{raw,processed,results}"
    )
    
    # Copy input files
    copy_task = ShellTask(
        name="copy_files",
        command="cp /source/path/*.txt data/raw/",
    )
    
    # Process files
    process_task = ShellTask(
        name="process_files",
        command="for file in data/raw/*.txt; do processed_file=data/processed/$(basename $file .txt)_processed.txt; sed 's/old/new/g' $file > $processed_file; done"
    )
    
    # Compress results  
    compress_task = ShellTask(
        name="compress_results",
        command="tar -czf results.tar.gz data/processed/"
    )
    
    # Build workflow
    graph.add_task(setup_task)
    graph.add_task(copy_task, dependencies=[setup_task])
    graph.add_task(process_task, dependencies=[copy_task])
    graph.add_task(compress_task, dependencies=[process_task])
    
    return graph

# Execute file operations
file_workflow = file_operations_example()
executor = Executor() 
results = executor.execute(file_workflow)
```

## Integration with Other Task Types

```python
from molexp import LocalTask, ShellTask, TaskGraph

def mixed_workflow():
    """Example combining shell tasks with Python tasks."""
    
    graph = TaskGraph()
    
    # Python task for data generation
    def generate_config(n_samples, method):
        config = {
            'samples': n_samples,
            'method': method,
            'output_dir': 'simulation_output'
        }
        
        import json
        with open('config.json', 'w') as f:
            json.dump(config, f)
        
        return 'config.json'
    
    config_task = LocalTask(
        name="generate_config",
        func=generate_config,
        inputs={'n_samples': 1000, 'method': 'monte_carlo'},
        outputs=['config_file']
    )
    
    # Shell task for simulation
    simulation_task = ShellTask(
        name="run_simulation",
        command="./external_simulator --config {config_file}",
        inputs={},  # config_file will be provided by dependency
        working_dir="/opt/simulator"
    )
    
    # Python task for post-processing
    def analyze_results(simulation_output):
        # Parse simulation results
        import json
        results = json.loads(simulation_output)
        
        # Perform analysis
        mean_value = sum(results['values']) / len(results['values'])
        std_dev = (sum((x - mean_value)**2 for x in results['values']) / len(results['values']))**0.5
        
        return {
            'mean': mean_value,
            'std_dev': std_dev,
            'n_samples': len(results['values'])
        }
    
    analysis_task = LocalTask(
        name="analyze_results",
        func=analyze_results,
        inputs={},  # simulation output will be provided
        outputs=['analysis']
    )
    
    # Build mixed workflow
    graph.add_task(config_task)
    graph.add_task(simulation_task, dependencies=[config_task])
    graph.add_task(analysis_task, dependencies=[simulation_task])
    
    return graph

# Execute mixed workflow
mixed_graph = mixed_workflow()
executor = Executor()
results = executor.execute(mixed_graph)

print("Final analysis:", results['analyze_results'])
```

## Best Practices

### Command Construction
1. **Use Parameter Substitution**: Always use `{parameter}` syntax for inputs
2. **Quote Arguments**: Properly quote arguments that might contain spaces
3. **Validate Commands**: Test commands manually before incorporating

### Error Handling
1. **Set Timeouts**: Always set reasonable timeouts for long-running commands
2. **Check Exit Codes**: Monitor command return codes
3. **Validate Output**: Implement output validation when possible

### Security
1. **Sanitize Inputs**: Validate and sanitize all input parameters
2. **Avoid Shell Injection**: Be careful with user-provided parameters
3. **Use Absolute Paths**: Prefer absolute paths to avoid confusion

### Performance
1. **Parallel Execution**: Shell tasks are perfect for parallel execution
2. **Resource Management**: Consider CPU and memory requirements
3. **Cleanup**: Clean up temporary files and directories

## Advanced Features

### Custom Output Parsing

```python
def parse_simulation_output(output):
    """Custom parser for simulation output."""
    lines = output.strip().split('\n')
    
    results = {}
    for line in lines:
        if line.startswith('ENERGY:'):
            results['energy'] = float(line.split()[1])
        elif line.startswith('TIME:'):
            results['time'] = float(line.split()[1])
    
    return results

advanced_task = ShellTask(
    name="simulation_with_parsing",
    command="./simulate --params {params}",
    inputs={'params': 'simulation.params'},
    output_parser=parse_simulation_output
)
```

### Conditional Execution

```python
def should_run_analysis(context):
    """Determine if analysis should run based on previous results."""
    if 'error_check' in context and context['error_check']['errors'] > 0:
        return False
    return True

conditional_task = ShellTask(
    name="conditional_analysis",
    command="python analysis.py --data {data}",
    condition=should_run_analysis
)
```

This example demonstrates the power and flexibility of shell tasks in MolExp workflows, enabling seamless integration of existing tools and scripts into sophisticated scientific computing pipelines.
