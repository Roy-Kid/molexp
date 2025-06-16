# molexp: a computational chemistry workflow and experiment management package

`molexp` is a comprehensive computational chemistry workflow and experiment management package, designed to streamline and enhance the efficiency of chemistry-related research and experimentation. At its core, `molexp` leverages the robust capabilities of [Hamilton](https://github.com/DAGWorks-Inc/hamilton), allowing users to define workflows in a manner that is testable, modular, and self-documenting. This ensures that all workflows are easy to understand, maintain, and share, as they are written in pure Python code.

One of the standout features of `molexp` is its advanced experiment management system. This system is meticulously designed to facilitate self-explanatory task organization, making it simple for researchers to manage and document their experiments effectively. Users can perform Create, Read, Update, and Delete (CRUD) operations directly from the command line, using specific variables to tailor the management process to their unique requirements.

## Features

1. **Integration with Hamilton**: By utilizing Hamilton, `molexp` ensures that workflows are not only powerful and flexible but also maintain high standards of code quality and reusability.

2. **Modular Workflow Design**: Workflows in `molexp` are built in a modular fashion, promoting reusability and ease of testing. This modularity allows for individual components to be tested and debugged independently, significantly reducing the time and effort required to ensure the reliability of complex workflows.

3. **Self-Documenting Code**: The use of pure Python code means that all workflows and experiments are inherently self-documenting. This transparency is crucial for collaborative research environments, where clear documentation and easy-to-follow code are essential for effective teamwork.

4. **Comprehensive Experiment Management**: The experiment management system in `molexp` is designed to simplify the organization of tasks. It supports comprehensive CRUD operations via the command line, allowing users to create, view, modify, and delete experiments with ease.

5. **Customizable Variables**: The system’s flexibility is further enhanced by the ability to use specific variables to manage experiments. This feature allows users to tailor their workflow management processes to suit their specific needs and preferences.

6. **User-Friendly Command Line Interface**: The command line interface of `molexp` is intuitive and user-friendly, ensuring that even users with minimal programming experience can effectively manage their computational chemistry workflows and experiments.

7. **Scalability and Flexibility**: `molexp` is designed to handle projects of varying sizes and complexities. Whether you are managing a small-scale experiment or a large, multifaceted research project, `molexp` scales to meet your needs.

8. **Community and Support**: As an open-source package, `molexp` benefits from a growing community of users and contributors. This ensures continuous improvements, updates, and a wealth of shared knowledge and resources.

## Installation

Install molexp from PyPI:

```bash
pip install molexp
```

Or install from source:

```bash
git clone https://github.com/your-repo/molexp.git
cd molexp
pip install -e .
```

## Quick Start

```python
import molexp as mx

# Create a simple shell task
task1 = mx.ShellTask(
    name="setup",
    commands=["echo 'Setting up environment'", "mkdir -p data"]
)

task2 = mx.ShellTask(
    name="process",
    commands=["echo 'Processing data'"],
    deps=["setup"]  # This task depends on 'setup'
)

# Create an experiment
experiment = mx.Experiment(name="my_experiment")
experiment.add_task(task1)
experiment.add_task(task2)

# Run the experiment
executor = mx.ExperimentExecutor(experiment)
results = executor.run()
print(results)
```

## Core Concepts

- **Project**: A standalone research topic containing necessary index and documentation files
- **Experiment**: A scientific trial with specific variables and parameters
- **Task**: Individual workflow components that can be chained together with dependencies
- **TaskPool**: Container for managing multiple tasks
- **Executor**: Engine for executing experiments with proper dependency resolution

## Task Types

- **ShellTask**: Execute shell commands with parameter substitution
- **HamiltonTask**: Integration with Hamilton workflows
- **LocalTask**: Local execution tasks
- **RemoteTask**: Remote execution capabilities

## Examples

See the `examples/` directory for comprehensive usage examples including:
- Basic workflow creation
- Parameter exploration
- Advanced shell task usage
- Hamilton integration
- Complex dependency management

## Documentation

For detailed documentation, visit [our documentation site](docs/) or build locally:

```bash
mkdocs serve
```

## Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests.

## License

This project is licensed under the terms specified in the LICENSE file.