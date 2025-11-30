# Quick Start Guide: Project-Experiment-Run in molexp

## Installation

```bash
cd /workspaces/molexp
pip install -e .
```

## Basic Workflow

### 1. Initialize Workspace

```bash
# Option 1: Use current directory
molexp init

# Option 2: Set environment variable
export MOLEXP_WORKSPACE=/path/to/workspace
molexp init
```

### 2. Create Project

```bash
molexp project create my-project \
  --name "My Research Project" \
  --desc "Description of the project" \
  --owner "researcher-name" \
  --tags "tag1,tag2"
```

### 3. Create Experiment

```bash
molexp experiment create my-project exp-1 \
  --name "Parameter Sweep" \
  --workflow "path/to/workflow.py" \
  --desc "Scanning parameter space" \
  --params '{"param1": [1, 2, 3]}'
```

### 4. Run Workflow with Asset Tracking

```python
from molexp.workspace import Workspace
from molexp.context import RunContext, use_run_context
from molexp.assets import AssetRepo, register_asset
from pathlib import Path

# Load workspace
workspace = Workspace.from_env()

# Create run
run = workspace.create_run(
    project_id="my-project",
    experiment_id="exp-1",
    parameters={"param1": 1.0},
    workflow_file="workflow.py",
)

# Set up context
ctx = RunContext(
    asset_repo=AssetRepo(),
    run_metadata=run,
    workspace=workspace,
)

# Execute workflow
with use_run_context(ctx):
    # Your workflow code here
    output_file = Path("output.txt")
    output_file.write_text("results")
    
    # Register output (automatically tracked and deduplicated)
    register_asset(output_file, label="results")

# Update run status
from molexp.models import RunStatus
from datetime import datetime

run.status = RunStatus.SUCCEEDED
run.finished_at = datetime.now()
workspace.update_run(run)
```

### 5. Query and Explore

```bash
# List all projects
molexp project list

# List experiments in a project
molexp experiment list my-project

# List runs in an experiment
molexp run list my-project exp-1

# View run details
molexp run info my-project exp-1 <run_id>

# List all assets
molexp asset list

# View asset details
molexp asset info <asset_id>
```

## Python API Examples

### Create Complete Workflow

```python
from molexp.workspace import Workspace

# Initialize
workspace = Workspace.from_path("./workspace")

# Create project
project = workspace.create_project(
    project_id="md-simulations",
    name="Molecular Dynamics",
    owner="researcher",
)

# Create experiment
experiment = workspace.create_experiment(
    project_id="md-simulations",
    experiment_id="density-scan",
    name="Density Parameter Scan",
    workflow_source="workflows/md.py",
    parameter_space={"density": [0.8, 0.9, 1.0, 1.1, 1.2]},
)

# Create multiple runs
for density in [0.8, 0.9, 1.0]:
    run = workspace.create_run(
        project_id="md-simulations",
        experiment_id="density-scan",
        parameters={"density": density},
        workflow_file="workflows/md.py",
    )
    print(f"Created run: {run.run_id}")
```

### Query Assets

```python
# Find all assets
assets = workspace.list_assets()

for asset in assets:
    print(f"Asset: {asset.asset_id}")
    print(f"  Type: {asset.type.value}")
    print(f"  Size: {asset.size_bytes} bytes")
    print(f"  Producer: {asset.producer_run_id}")
```

### Check for Duplicate Assets

```python
from molexp.id_utils import compute_content_hash
from pathlib import Path

# Compute hash of a file
file_path = Path("data.txt")
content_hash = compute_content_hash(file_path)

# Check if asset already exists
existing_id = workspace.find_asset_by_hash(content_hash)

if existing_id:
    print(f"Asset already exists: {existing_id}")
else:
    print("New asset, will be stored")
```

## Directory Structure

After running the above commands, your workspace will look like:

```
workspace/
├── projects/
│   └── my-project/
│       ├── project.yaml
│       └── experiments/
│           └── exp-1/
│               ├── experiment.yaml
│               └── runs/
│                   └── 20251129_183000_a3b2/
│                       ├── run.json
│                       ├── context.json
│                       ├── asset_refs.json
│                       ├── logs/
│                       └── artifacts/
└── assets/
    └── <asset_id>/
        ├── meta.yaml
        └── data/
            └── output.txt
```

## Key Features

### Automatic Asset Deduplication

When you register an asset with `register_asset()`:
1. Content hash is computed (SHA256)
2. System checks if asset with same hash exists
3. If exists: reuses existing asset_id
4. If new: creates new asset and stores data
5. AssetRef is created linking run to asset

**Result**: Identical files are stored only once!

### Complete Reproducibility

Each run captures:
- Workflow snapshot (file, git commit)
- All parameters
- Environment variables
- Dependency versions
- Hardware info
- Input/output asset references

### Backward Compatibility

Existing molexp code works without changes:

```python
from molexp.task_base import Task, EmptyConfig
from molexp.engine import TaskEngine

# This still works exactly as before
task = MyTask(name="task")
engine = TaskEngine()
result = engine.run(task)
```

## Common Patterns

### Pattern 1: Parameter Sweep

```python
workspace = Workspace.from_env()

for param_value in [0.1, 0.2, 0.3, 0.4, 0.5]:
    run = workspace.create_run(
        project_id="my-project",
        experiment_id="sweep",
        parameters={"param": param_value},
        workflow_file="sweep.py",
    )
    
    # Execute workflow...
    # Results automatically tracked
```

### Pattern 2: Reusing Assets Across Runs

```python
# First run produces an asset
run1 = workspace.create_run(...)
# ... execute and register asset with id: asset_123

# Second run uses that asset as input
from molexp.models import AssetRef
from datetime import datetime

run2 = workspace.create_run(...)
refs = workspace.get_asset_refs(
    run2.project_id, run2.experiment_id, run2.run_id
)

# Add input reference
refs.inputs.append(AssetRef(
    asset_id="asset_123",
    role="input_structure",
    accessed_at=datetime.now(),
))

workspace.save_asset_refs(
    run2.project_id, run2.experiment_id, run2.run_id, refs
)
```

### Pattern 3: Querying Run History

```python
# Get all runs for an experiment
runs = workspace.list_runs("my-project", "exp-1")

# Filter by status
from molexp.models import RunStatus

successful_runs = [r for r in runs if r.status == RunStatus.SUCCEEDED]
failed_runs = [r for r in runs if r.status == RunStatus.FAILED]

# Find runs with specific parameter
runs_with_param = [
    r for r in runs 
    if r.parameters.get("density") == 1.0
]
```

## Troubleshooting

### Workspace not found

```bash
# Set workspace explicitly
export MOLEXP_WORKSPACE=/path/to/workspace
# Or use --path flag
molexp project list --path /path/to/workspace
```

### Asset not deduplicating

Check that:
1. File content is truly identical (byte-for-byte)
2. You're using `register_asset()` within a RunContext
3. Workspace is properly initialized

### Run metadata not captured

Ensure you:
1. Create RunContext with `run_metadata` and `workspace`
2. Use `with use_run_context(ctx):` block
3. Call `register_asset()` inside the context

## Next Steps

- See [examples/project_experiment_run_example.py](file:///workspaces/molexp/examples/project_experiment_run_example.py) for complete example
- Read [docs/architecture.md](file:///workspaces/molexp/docs/architecture.md) for design details
- Run tests: `pytest tests/test_models.py tests/test_id_utils.py tests/test_integration.py`
