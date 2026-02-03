# Workspace API

The Workspace class provides a unified interface for managing projects, experiments, runs, and assets. This document covers the complete API reference.

## What the Workspace API Is

The `Workspace` class is the central access point for all repository operations in MolExp. It provides methods for creating, reading, updating, and deleting projects, experiments, runs, and assets. The workspace manages the underlying file system structure and ensures data consistency.

The workspace uses a repository pattern, with separate repositories for each entity type (projects, experiments, runs, assets). This design allows for easy extension and testing, as repositories can be swapped or mocked.

## Why This Design

The workspace API design provides several benefits. First, it offers a single, consistent interface for all workspace operations, reducing the cognitive load of learning multiple APIs. Second, the repository pattern allows for different storage backends (file system, database, cloud storage) without changing the API.

Third, the workspace automatically manages the file system structure, creating necessary directories and maintaining consistency. This reduces boilerplate code and prevents common errors. Finally, the API is designed to be intuitive and Pythonic, making it easy to use in interactive sessions and scripts.

## How to Use

### Workspace Initialization

Create a workspace from a path or environment variable:

```python
from molexp.workspace.core import Workspace
from pathlib import Path

# From explicit path
workspace = Workspace.from_path("./my_workspace")

# From environment variable (defaults to current directory)
workspace = Workspace.from_env("MOLEXP_WORKSPACE")

# Direct initialization
workspace = Workspace(Path("./workspace"))
```

### Project Operations

Projects are top-level containers:

```python
# Create project
project = workspace.create_project(
    id="my_project",
    name="My Project",
    description="Project description",
    owner="alice",
    tags=["tag1", "tag2"],
    config={"custom": "value"},
)

# Get project
project = workspace.get_project("my_project")

# List all projects
projects = workspace.list_projects()

# Delete project
workspace.delete_project("my_project")
```

### Experiment Operations

Experiments define workflow templates:

```python
# Create experiment
experiment = workspace.create_experiment(
    id="my_project",
    id="exp_1",
    name="Experiment 1",
    workflow_source="workflow.py",
    description="Experiment description",
    workflow_type="taskgraph_v1",
    git_commit="abc123",
    parameter_space={"param": [1, 2, 3]},
    default_inputs=[],  # List of AssetRef
)

# Get experiment
experiment = workspace.get_experiment("my_project", "exp_1")

# List experiments in project
experiments = workspace.list_experiments("my_project")

# Delete experiment
workspace.delete_experiment("my_project", "exp_1")
```

### Run Operations

Runs are execution instances:

```python
# Create run
run = workspace.create_run(
    id="my_project",
    id="exp_1",
    parameters={"param": 1},
    workflow_file="workflow.py",
    git_commit="abc123",
    id=None,  # Auto-generated if None
)

# Get run
run = workspace.get_run("my_project", "exp_1", "run_001")

# Update run
run.status = RunStatus.SUCCEEDED
workspace.update_run(run)

# List runs in experiment
runs = workspace.list_runs("my_project", "exp_1")

# Delete run
workspace.delete_run("my_project", "exp_1", "run_001")
```

### Run Context Operations

Save and retrieve run context snapshots:

```python
from molexp.models import RunContextSnapshot

# Save context
context = RunContextSnapshot(
    id=run.id,
    status=run.status,
    # ... other context data
)
workspace.save_run_context("my_project", "exp_1", "run_001", context)

# Get context
context = workspace.get_run_context("my_project", "exp_1", "run_001")
```

### Asset Reference Operations

Manage asset references for runs:

```python
from molexp.models import AssetRefsCollection, AssetRef

# Save asset references
refs = AssetRefsCollection(
    inputs=[],
    outputs=[
        AssetRef(asset_id="asset_001", role="result", producer_id="run_001"),
    ],
)
workspace.save_asset_refs("my_project", "exp_1", "run_001", refs)

# Get asset references
refs = workspace.get_asset_refs("my_project", "exp_1", "run_001")
```

### Asset Operations

Manage assets in the workspace:

```python
from molexp.models import Asset, AssetType
from pathlib import Path

# Store asset
asset = Asset(
    asset_id="asset_001",
    type=AssetType.STRUCTURE,
    format="pdb",
    created_at=datetime.now(),
    size_bytes=1024,
    content_hash="abc123...",
    # ... other fields
)
asset_id = workspace.store_asset(asset, Path("structure.pdb"))

# Get asset metadata
asset = workspace.get_asset("asset_001")

# Retrieve asset data
workspace.retrieve_asset("asset_001", Path("downloaded.pdb"))

# Find asset by content hash
asset_id = workspace.find_asset_by_hash("abc123...")

# List all assets
assets = workspace.list_assets()

# Delete asset
workspace.delete_asset("asset_001")
```

### Complete Example

Here's a complete example using the workspace API:

```python
from molexp.workspace.core import Workspace
from molexp.models import RunStatus, AssetType, Asset
from datetime import datetime
from pathlib import Path

def main():
    # Initialize workspace
    workspace = Workspace.from_path("./workspace")
    
    # Create project
    project = workspace.create_project(
        id="demo",
        name="Demo Project",
    )
    
    # Create experiment
    experiment = workspace.create_experiment(
        id="demo",
        id="demo_exp",
        name="Demo Experiment",
        workflow_source="demo.py",
    )
    
    # Create run
    run = workspace.create_run(
        id="demo",
        id="demo_exp",
        parameters={"value": 42},
        workflow_file="demo.py",
    )
    
    # Store an asset
    asset = Asset(
        asset_id="demo_asset",
        type=AssetType.OTHER,
        format="txt",
        created_at=datetime.now(),
        size_bytes=100,
        content_hash="hash123",
        files=[],
    )
    workspace.store_asset(asset, Path("demo.txt"))
    
    # Update run status
    run.status = RunStatus.SUCCEEDED
    workspace.update_run(run)
    
    # Query
    runs = workspace.list_runs("demo", "demo_exp")
    print(f"Total runs: {len(runs)}")

if __name__ == "__main__":
    main()
```

The Workspace API provides a complete interface for managing all aspects of your scientific computing workflows, from project organization to asset tracking.



