# MolExp Examples

This directory contains user-friendly examples that demonstrate the key features and capabilities of MolExp. Each example is self-contained and focuses on specific use cases.

## Overview

The examples are organized from basic to advanced concepts:

1. **Basic Usage** - Getting started with MolExp
2. **Experiment Management** - Managing scientific workflows
3. **Shell Tasks** - Command-line tool integration
4. **Parameter Exploration** - Running parameter studies
5. **Advanced Workflows** - Complex dependency graphs and parallel execution
6. **Hamilton Integration** - Integration with the Hamilton dataflow framework

## Running the Examples

Each example is a standalone Python script that can be run directly:

```bash
# Run from the project root directory
cd /workspaces/molexp

# Basic usage example
python examples/01_basic_usage.py

# Experiment management example
python examples/02_experiment_management.py

# Shell tasks example
python examples/03_shell_tasks.py

# Parameter exploration example
python examples/04_parameter_exploration.py

# Advanced workflow example
python examples/05_advanced_workflow.py

# Hamilton integration example
python examples/06_hamilton_integration.py
```

## Example Details

### 01_basic_usage.py
**Getting Started with MolExp**

This example introduces the fundamental concepts of MolExp:
- Creating tasks with dependencies
- Setting up task pools
- Running experiments with ExperimentExecutor
- Checking execution status and results

Perfect for newcomers to understand the basic workflow.

### 02_experiment_management.py
**Scientific Experiment Organization**

Demonstrates how to use the Experiment class for scientific workflows:
- Creating experiments with metadata
- Adding tasks to experiments
- Experiment validation
- YAML serialization for reproducibility
- Loading and saving experiment configurations

Ideal for researchers who need to organize and document their computational experiments.

### 03_shell_tasks.py
**Command-Line Tool Integration**

Shows how to integrate external command-line tools and scripts:
- ShellTask for general command execution
- LocalTask for local script execution
- RemoteTask for HPC and cluster computing
- File processing pipelines
- Complex command sequences

Great for users who need to integrate existing tools and scripts into MolExp workflows.

### 04_parameter_exploration.py
**Parameter Space Studies**

Demonstrates parameter exploration and optimization workflows:
- Running the same workflow with different parameters
- Parameter space definition
- Result collection and analysis
- Batch processing patterns
- Comparative analysis across conditions

Perfect for users conducting parameter studies, optimization, or sensitivity analysis.

### 05_advanced_workflow.py
**Complex Workflow Orchestration**

Covers advanced workflow features and patterns:
- Complex dependency graphs
- Parallel execution opportunities identification
- Error handling and recovery strategies
- Progress monitoring and status tracking
- Workflow analysis and optimization

For users building sophisticated computational pipelines with complex dependencies.

### 06_hamilton_integration.py
**Hamilton Dataflow Integration**

Shows integration with the Hamilton dataflow framework:
- HamiltonTask creation and configuration
- Module-based data processing
- Type-safe pipeline development
- Combining MolExp workflow orchestration with Hamilton data processing
- Serialization of Hamilton-based tasks

For users who want to combine MolExp's workflow orchestration with Hamilton's type-safe data processing capabilities.

## Key Concepts Covered

### Core Concepts
- **Tasks**: Basic units of work with inputs, outputs, and dependencies
- **TaskPool**: Collections of related tasks
- **TaskGraph**: Dependency analysis and execution ordering
- **ExperimentExecutor**: Experiment execution engine
- **Experiment**: High-level experiment organization

### Task Types
- **Task**: Basic task definition
- **ShellTask**: Command-line tool execution
- **LocalTask**: Local script execution
- **RemoteTask**: Remote computation
- **HamiltonTask**: Hamilton dataflow integration

### Advanced Features
- Dependency validation and cycle detection
- Parallel execution identification
- Error handling and recovery
- Progress monitoring
- YAML serialization for reproducibility
- Parameter exploration patterns

## Best Practices Demonstrated

1. **Clear Task Naming**: Use descriptive names that indicate the task's purpose
2. **Dependency Management**: Properly define task dependencies for correct execution order
3. **Output Specification**: Clearly specify expected outputs for each task
4. **Error Handling**: Design workflows that can handle and recover from failures
5. **Documentation**: Include meaningful descriptions (readme) for all tasks and experiments
6. **Modularity**: Break complex workflows into smaller, reusable components
7. **Validation**: Always validate workflows before execution
8. **Monitoring**: Track execution progress and status

## Next Steps

After running the examples:

1. **Modify Examples**: Try changing parameters, adding tasks, or modifying dependencies
2. **Create Your Own**: Use the examples as templates for your own workflows
3. **Explore Integration**: Integrate your existing tools and scripts using ShellTask
4. **Advanced Features**: Experiment with parallel execution and error handling patterns
5. **Documentation**: Read the full documentation for comprehensive API reference

## Support

If you have questions about the examples or encounter issues:

1. Check the main project documentation
2. Look at the test files for additional usage patterns
3. Create an issue on the project repository

Happy workflow automation with MolExp!
