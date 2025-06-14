# Installation

## Requirements

MolExp requires Python 3.8 or higher and has the following dependencies:

- `pydantic` >= 2.0 - Data validation and serialization
- `pyyaml` - YAML file handling
- `pytest` - Testing framework (development)

## Installation Methods

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/MolCrafts/molexp.git
cd molexp

# Install in development mode
pip install -e .
```

### For Development

If you plan to contribute to MolExp or run the test suite:

```bash
# Clone the repository
git clone https://github.com/MolCrafts/molexp.git
cd molexp

# Install with development dependencies
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/
```

## Verification

To verify your installation:

```python
import molexp as mx

# Create a simple task
task = mx.Task(name="test_task", readme="Test installation")
print(f"MolExp installed successfully! Created task: {task.name}")
```

## Optional Dependencies

### For Hamilton Integration

If you plan to use HamiltonTask for dataflow workflows:

```bash
pip install hamilton-sdk
```

### For Enhanced Documentation

For building documentation locally:

```bash
pip install mkdocs-material mkdocstrings[python]
```

## Troubleshooting

### Common Issues

**Import Error**: If you encounter import errors, ensure you've installed all dependencies:
```bash
pip install pydantic pyyaml
```

**Python Version**: MolExp requires Python 3.8+. Check your version:
```bash
python --version
```

**Path Issues**: If running from source, ensure the package is in your Python path or use development installation (`pip install -e .`).

### Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/MolCrafts/molexp/issues) for similar problems
2. Create a new issue with your system details and error messages
3. Include Python version, operating system, and full error traceback
