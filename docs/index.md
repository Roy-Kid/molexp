# MolExp

A flexible experiment management and workflow orchestration framework for molecular sciences.

## Overview

MolExp is a Python framework designed to simplify the creation, management, and execution of complex computational workflows in molecular sciences. It provides a clean, modular architecture for defining tasks, managing dependencies, and orchestrating experiment execution.

## Key Features

### 🔧 **Flexible Task Definition**
- Multiple task types: basic tasks, shell commands, local scripts, remote execution
- Support for Hamilton dataflow framework integration
- Easy dependency management between tasks

### 🧪 **Experiment Management**
- Organize tasks into reusable experiments
- YAML serialization for reproducibility
- Comprehensive experiment validation

### 🚀 **Workflow Orchestration**
- Automatic dependency resolution and execution ordering
- Parallel execution opportunity identification
- Real-time status monitoring and progress tracking

### 📊 **Parameter Exploration**
- Built-in support for parameter studies
- Batch processing capabilities
- Result collection and analysis

### 🔍 **Robust Error Handling**
- Comprehensive error detection and reporting
- Graceful failure handling
- Task status tracking and recovery

## Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/MolCrafts/molexp
cd molexp
pip install -e .
```

### Basic Usage

```python
import molexp as mx

# Create tasks
prep_task = mx.Task(
    name="data_preparation",
    readme="Prepare input data",
    args=["--input", "data.txt"]
)

analysis_task = mx.Task(
    name="analysis",
    readme="Analyze prepared data", 
    deps=["data_preparation"]
)

# Create and run workflow
task_pool = mx.TaskPool("my_workflow")
task_pool.add_task(prep_task)
task_pool.add_task(analysis_task)

experiment = mx.Experiment(name="molecular_analysis")
experiment.set_task_pool(task_pool)
executor = mx.ExperimentExecutor(experiment)
results = executor.run()
```

## Use Cases

### Molecular Dynamics Simulations
- System preparation workflows
- Multi-stage simulation pipelines
- Parameter optimization studies

### Computational Chemistry
- High-throughput screening
- Method comparison studies
- Property prediction workflows

### Data Processing Pipelines
- Multi-step data analysis
- Batch processing workflows
- Result aggregation and reporting

### Machine Learning
- Model training pipelines
- Hyperparameter optimization
- Cross-validation workflows

## Architecture

MolExp follows a modular architecture with clear separation of concerns:

- **Tasks**: Basic units of work with dependencies
- **TaskPool**: Collections of related tasks
- **TaskGraph**: Dependency analysis and validation
- **Executor**: Task execution engine
- **Experiment**: High-level workflow organization

## Getting Started

1. **[Installation Guide](getting_started/installation.md)** - Set up MolExp in your environment
2. **[Quick Start Tutorial](getting_started/quick_start.md)** - Create your first workflow
3. **[Basic Concepts](getting_started/concepts.md)** - Understand core concepts
4. **[Examples](examples/basic_usage.md)** - Learn from practical examples

## Community

- **GitHub**: [MolCrafts/molexp](https://github.com/MolCrafts/molexp)
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share workflows

## License

MolExp is released under the MIT License. See [LICENSE](https://github.com/MolCrafts/molexp/blob/main/LICENSE) for details.