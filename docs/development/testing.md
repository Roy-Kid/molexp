# Testing

This guide covers the testing framework and practices for MolExp.

## Test Structure

MolExp uses pytest for testing with a comprehensive test suite:

```
tests/
├── test_task.py          # Task functionality tests
├── test_pool.py          # Task pool tests  
├── test_graph.py         # Task graph tests
├── test_executor.py      # Execution engine tests
├── test_workflow.py      # Workflow orchestration tests
├── test_experiment.py    # Experiment management tests
├── test_param.py         # Parameter system tests
├── conftest.py           # Shared test fixtures
└── integration/          # Integration tests
    ├── test_workflows.py
    └── test_experiments.py
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_task.py

# Run specific test class
pytest tests/test_task.py::TestLocalTask

# Run specific test method
pytest tests/test_task.py::TestLocalTask::test_execute_simple_function
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=molexp

# Generate HTML coverage report
pytest --cov=molexp --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"
```

## Writing Tests

### Test Organization

Each source file has a corresponding test file with test classes for each class:

```python
# tests/test_task.py
import pytest
from molexp import Task, LocalTask, ShellTask

class TestTask:
    """Test base Task class functionality."""
    
    def test_task_creation(self):
        task = Task("test_task")
        assert task.name == "test_task"
        assert task.status == TaskStatus.PENDING

class TestLocalTask:
    """Test LocalTask functionality."""
    
    def test_execute_simple_function(self):
        def add(x, y):
            return x + y
        
        task = LocalTask("add_task", func=add)
        result = task.execute(inputs={'x': 2, 'y': 3})
        assert result == 5

class TestShellTask:
    """Test ShellTask functionality."""
    
    def test_execute_command(self):
        task = ShellTask("echo_task", command="echo 'hello'")
        result = task.execute()
        assert "hello" in result
```

### Test Fixtures

Use pytest fixtures for common test setup:

```python
# tests/conftest.py
import pytest
from molexp import LocalTask, TaskGraph, Executor

@pytest.fixture
def simple_task():
    """Create a simple task for testing."""
    def square(x):
        return x ** 2
    
    return LocalTask("square_task", func=square)

@pytest.fixture
def task_graph():
    """Create a task graph for testing."""
    graph = TaskGraph()
    
    task1 = LocalTask("task1", func=lambda: 1)
    task2 = LocalTask("task2", func=lambda x: x + 1)
    task3 = LocalTask("task3", func=lambda x: x * 2)
    
    graph.add_task(task1)
    graph.add_task(task2, dependencies=[task1])
    graph.add_task(task3, dependencies=[task2])
    
    return graph

@pytest.fixture
def executor():
    """Create an executor for testing."""
    return Executor(max_workers=2)
```

### Parametrized Tests

Use parametrization for testing multiple scenarios:

```python
@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (0, 5, 5),
    (-1, 1, 0),
    (10, -5, 5),
])
def test_addition(x, y, expected):
    def add(a, b):
        return a + b
    
    task = LocalTask("add_task", func=add)
    result = task.execute(inputs={'a': x, 'b': y})
    assert result == expected
```

### Testing Error Conditions

Test error handling and edge cases:

```python
def test_task_execution_error():
    """Test task execution with function that raises exception."""
    def failing_function():
        raise ValueError("Test error")
    
    task = LocalTask("failing_task", func=failing_function)
    
    with pytest.raises(TaskExecutionError):
        task.execute()
    
    assert task.status == TaskStatus.FAILED
    assert isinstance(task.error, ValueError)

def test_dependency_cycle_detection():
    """Test detection of circular dependencies."""
    graph = TaskGraph()
    task1 = LocalTask("task1", func=lambda: 1)
    task2 = LocalTask("task2", func=lambda: 2)
    
    graph.add_task(task1)
    graph.add_task(task2, dependencies=[task1])
    
    with pytest.raises(CyclicDependencyError):
        graph.add_dependency(task1, task2)  # Creates cycle
```

### Mocking and Patching

Use mocking for external dependencies:

```python
from unittest.mock import Mock, patch

def test_remote_task_execution():
    """Test remote task execution with mocked SSH connection."""
    with patch('paramiko.SSHClient') as mock_ssh:
        # Configure mock
        mock_client = Mock()
        mock_ssh.return_value = mock_client
        mock_client.exec_command.return_value = (None, Mock(), Mock())
        
        # Test remote task
        task = RemoteTask(
            "remote_task",
            func=lambda: "result",
            host="test.host",
            user="testuser"
        )
        
        result = task.execute()
        
        # Verify SSH connection was attempted
        mock_ssh.assert_called_once()
        mock_client.connect.assert_called_once()
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_workflows.py
class TestWorkflowIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment execution."""
        # Create experiment with parameter space
        experiment = Experiment("integration_test")
        
        param_space = ParameterSpace({
            'x': FloatParameter(min=1, max=5, step=1),
            'y': FloatParameter(min=1, max=3, step=1)
        })
        experiment.set_parameter_space(param_space)
        
        # Add computation task
        def compute(x, y):
            return x * y + x + y
        
        task = LocalTask("compute", func=compute)
        experiment.add_task(task)
        
        # Execute experiment
        results = experiment.run()
        
        # Verify results
        assert len(results) == 15  # 5 * 3 parameter combinations
        assert all(r.status == TaskStatus.COMPLETED for r in results)
        assert all('compute' in r.outputs for r in results)
```

## Test Data and Resources

### Test Data Organization

```
tests/
├── data/
│   ├── input_files/
│   ├── expected_outputs/
│   └── fixtures/
├── resources/
│   ├── test_molecules.xyz
│   └── reference_results.json
└── conftest.py
```

### Loading Test Data

```python
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_molecule(test_data_dir):
    """Load sample molecule data."""
    molecule_file = test_data_dir / "input_files" / "water.xyz"
    return load_molecule(molecule_file)

def test_molecule_processing(sample_molecule):
    """Test molecule processing with real data."""
    result = process_molecule(sample_molecule)
    assert result is not None
```

## Performance Testing

### Benchmarking

```python
import time
import pytest

@pytest.mark.performance
def test_large_graph_execution_performance():
    """Benchmark execution of large task graphs."""
    n_tasks = 1000
    graph = create_large_task_graph(n_tasks)
    executor = Executor(max_workers=4)
    
    start_time = time.time()
    results = executor.execute(graph)
    execution_time = time.time() - start_time
    
    assert len(results) == n_tasks
    assert execution_time < 60  # Should complete within 1 minute
    
    print(f"Executed {n_tasks} tasks in {execution_time:.2f} seconds")
    print(f"Throughput: {n_tasks/execution_time:.1f} tasks/second")

@pytest.mark.performance
def test_memory_usage():
    """Test memory usage for large experiments."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large experiment
    experiment = create_large_experiment(n_parameters=10000)
    results = experiment.run()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 500  # Should not use more than 500MB
    print(f"Memory usage: {memory_increase:.1f} MB")
```

### Load Testing

```python
@pytest.mark.load
def test_concurrent_experiments():
    """Test running multiple experiments concurrently."""
    import concurrent.futures
    
    def run_experiment(experiment_id):
        experiment = create_test_experiment(f"exp_{experiment_id}")
        return experiment.run()
    
    # Run 10 experiments concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_experiment, i) for i in range(10)]
        results = [future.result() for future in futures]
    
    assert len(results) == 10
    assert all(len(r) > 0 for r in results)
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest --cov=molexp --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Quality Gates

```yaml
    - name: Check test coverage
      run: |
        pytest --cov=molexp --cov-fail-under=90
    
    - name: Run linting
      run: |
        flake8 src/molexp/
        black --check src/molexp/
        isort --check-only src/molexp/
    
    - name: Type checking
      run: |
        mypy src/molexp/
```

## Best Practices

### Test Design

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should clearly describe what is being tested
3. **Follow AAA Pattern**: Arrange, Act, Assert
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Keep Tests Independent**: Tests should not depend on each other

### Performance

1. **Use Fixtures**: Reuse expensive setup with fixtures
2. **Mock External Dependencies**: Don't test external services
3. **Parallel Execution**: Use pytest-xdist for parallel test execution
4. **Skip Slow Tests**: Mark slow tests and skip them during development

### Maintenance

1. **Update Tests with Code**: Keep tests synchronized with code changes
2. **Regular Test Review**: Periodically review and refactor tests
3. **Test Coverage Goals**: Maintain high test coverage (>90%)
4. **Documentation**: Document complex test scenarios

### Example Test Session

```bash
# Development testing workflow
pytest -x --lf                    # Stop on first failure, run last failed
pytest -k "test_task"             # Run tests matching pattern
pytest --collect-only             # Show what tests would run
pytest -v --tb=short             # Verbose output with short tracebacks
pytest --durations=10            # Show 10 slowest tests
```

This comprehensive testing approach ensures MolExp maintains high quality and reliability across all components.
