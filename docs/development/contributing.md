# Development Guide

This guide provides information for developers who want to contribute to MolExp or extend its functionality.

## Contributing

We welcome contributions to MolExp! Here's how to get started:

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/MolCrafts/molexp.git
cd molexp
```

2. Create a development environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

### Development Workflow

1. **Create a Branch**: Create a feature branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your feature or fix
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**: Run the test suite
   ```bash
   pytest tests/
   ```

4. **Submit a Pull Request**: Push your branch and create a PR
   - Provide a clear description of your changes
   - Reference any related issues
   - Ensure all CI checks pass

### Coding Standards

MolExp follows these coding standards:

#### Code Style
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all public APIs

#### Documentation
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
- Document all public functions and classes
- Include examples in docstrings when helpful
- Update API documentation for new features

#### Testing
- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Include both unit and integration tests

### Code Review Process

All contributions go through code review:

1. **Automated Checks**: CI runs tests, linting, and type checking
2. **Manual Review**: Core maintainers review code quality and design
3. **Feedback**: Address any feedback from reviewers
4. **Approval**: Once approved, changes are merged

## Architecture

### Core Components

MolExp is built around several key components:

```
src/molexp/
├── task.py          # Task classes and execution
├── pool.py          # Task pool management
├── graph.py         # Task dependency graphs
├── executor.py      # Execution engines
├── workflow.py      # Workflow orchestration
├── experiment.py    # High-level experiment interface
├── param.py         # Parameter space definition
└── dispatch/        # Backend execution dispatch
```

### Design Principles

1. **Modularity**: Components are loosely coupled and easily testable
2. **Extensibility**: Easy to add new task types and execution backends
3. **Flexibility**: Support for various workflow patterns and use cases
4. **Performance**: Efficient execution with minimal overhead
5. **Usability**: Simple APIs for common use cases

### Task System

The task system is the core of MolExp:

```python
# Base task interface
class Task:
    def execute(self, inputs=None, **kwargs):
        """Execute the task with given inputs."""
        pass
    
    def add_dependency(self, task):
        """Add a dependency to this task."""
        pass
```

All specific task types inherit from this base class and implement the `execute` method.

### Execution Model

MolExp uses a graph-based execution model:

1. **Task Graph**: Tasks and their dependencies form a directed acyclic graph
2. **Topological Ordering**: Tasks are executed in dependency order
3. **Parallel Execution**: Independent tasks can run concurrently
4. **Result Propagation**: Task outputs become inputs for dependent tasks

## Adding New Features

### Adding a New Task Type

To add a new task type:

1. **Create the Task Class**:
```python
from molexp.task import Task

class MyCustomTask(Task):
    def __init__(self, name, custom_param, **kwargs):
        super().__init__(name, **kwargs)
        self.custom_param = custom_param
    
    def execute(self, inputs=None, **kwargs):
        # Implementation here
        pass
```

2. **Add Tests**:
```python
# tests/test_custom_task.py
import pytest
from molexp import MyCustomTask

class TestMyCustomTask:
    def test_creation(self):
        task = MyCustomTask("test", custom_param="value")
        assert task.name == "test"
        assert task.custom_param == "value"
    
    def test_execution(self):
        # Test execution logic
        pass
```

3. **Update Exports**:
```python
# src/molexp/__init__.py
from .task import MyCustomTask

__all__ = [..., "MyCustomTask"]
```

4. **Add Documentation**:
   - Update API documentation
   - Add usage examples
   - Update getting started guide if relevant

### Adding a New Execution Backend

To add support for a new execution backend:

1. **Create Backend Class**:
```python
from molexp.dispatch.base import BaseBackend

class MyBackend(BaseBackend):
    def execute_task(self, task, **kwargs):
        # Backend-specific execution logic
        pass
    
    def execute_graph(self, graph, **kwargs):
        # Graph execution logic
        pass
```

2. **Register Backend**:
```python
# In executor.py
BACKENDS = {
    'local': LocalBackend,
    'remote': RemoteBackend,
    'my_backend': MyBackend,  # Add your backend
}
```

3. **Add Configuration Options**:
```python
# Support backend-specific configuration
executor = Executor(backend='my_backend', backend_config={
    'custom_option': 'value'
})
```

### Adding New Parameter Types

To add a new parameter type:

1. **Create Parameter Class**:
```python
from molexp.param import Parameter

class MyParameter(Parameter):
    def __init__(self, name, custom_attr, **kwargs):
        super().__init__(name, **kwargs)
        self.custom_attr = custom_attr
    
    def sample(self, n_samples=1):
        # Sampling logic
        pass
    
    def validate(self, value):
        # Validation logic
        pass
```

2. **Add to Parameter Space**:
```python
# Ensure it works with ParameterSpace
param_space = ParameterSpace({
    'my_param': MyParameter('my_param', custom_attr='value')
})
```

## Testing

### Test Structure

Tests are organized by component:

```
tests/
├── test_task.py          # Task functionality
├── test_pool.py          # Task pool tests
├── test_graph.py         # Task graph tests
├── test_executor.py      # Execution engine tests
├── test_workflow.py      # Workflow tests
├── test_experiment.py    # Experiment tests
├── test_param.py         # Parameter tests
└── integration/          # Integration tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_task.py

# Run with coverage
pytest --cov=molexp tests/

# Run integration tests
pytest tests/integration/
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test execution performance
5. **Regression Tests**: Prevent known bugs from reoccurring

### Writing Good Tests

1. **Use Descriptive Names**: Test names should describe what is being tested
2. **Test Edge Cases**: Include boundary conditions and error cases
3. **Use Fixtures**: Reuse common test setup with pytest fixtures
4. **Mock External Dependencies**: Use mocking for external services
5. **Assert Specific Behaviors**: Test specific outputs, not just "no error"

Example test:

```python
class TestLocalTask:
    def test_execute_simple_function(self):
        """Test execution of a simple Python function."""
        def add(x, y):
            return x + y
        
        task = LocalTask("add_task", func=add)
        result = task.execute(inputs={'x': 2, 'y': 3})
        
        assert result == 5
        assert task.status == TaskStatus.COMPLETED
        assert task.result == 5
    
    def test_execute_with_error(self):
        """Test handling of function execution errors."""
        def failing_function():
            raise ValueError("Test error")
        
        task = LocalTask("failing_task", func=failing_function)
        
        with pytest.raises(TaskExecutionError):
            task.execute()
        
        assert task.status == TaskStatus.FAILED
        assert isinstance(task.error, ValueError)
```

## Performance Considerations

### Optimization Guidelines

1. **Minimize Task Overhead**: Keep task creation and management lightweight
2. **Efficient Serialization**: Use efficient serialization for distributed execution
3. **Memory Management**: Avoid holding large objects in memory unnecessarily
4. **Parallel Execution**: Leverage parallelism where possible
5. **Caching**: Cache expensive computations when appropriate

### Profiling

Use profiling tools to identify performance bottlenecks:

```python
# Profile code execution
import cProfile
cProfile.run('experiment.run()')

# Memory profiling
from memory_profiler import profile

@profile
def test_function():
    # Code to profile
    pass
```

### Benchmarking

Include benchmarks for performance-critical code:

```python
# tests/benchmarks/test_execution_performance.py
import time
import pytest
from molexp import Executor, TaskGraph

class TestExecutionPerformance:
    def test_large_graph_execution(self):
        """Benchmark execution of large task graphs."""
        graph = create_large_task_graph(n_tasks=1000)
        executor = Executor(max_workers=4)
        
        start_time = time.time()
        results = executor.execute(graph)
        execution_time = time.time() - start_time
        
        assert len(results) == 1000
        assert execution_time < 60  # Should complete within 1 minute
```

## Release Process

### Version Management

MolExp uses [semantic versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions  
- PATCH version for backwards-compatible bug fixes

### Release Checklist

1. **Update Version**: Update version in `pyproject.toml`
2. **Update Changelog**: Document all changes in `CHANGELOG.md`
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Ensure docs are up to date
5. **Tag Release**: Create git tag for the version
6. **Build Package**: Build distribution packages
7. **Upload to PyPI**: Upload to Python Package Index
8. **Update Documentation Site**: Deploy updated documentation

### Continuous Integration

MolExp uses GitHub Actions for CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run tests
        run: pytest --cov=molexp tests/
```

## Getting Help

### Documentation

- **API Reference**: Complete API documentation
- **User Guide**: Comprehensive usage examples
- **Examples**: Practical examples for common use cases

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Email**: Contact maintainers directly for sensitive issues

### Contributing

We appreciate all contributions, including:
- Bug reports and fixes
- Feature requests and implementations
- Documentation improvements
- Example scripts and tutorials
- Performance optimizations
- Test coverage improvements

Thank you for contributing to MolExp!
