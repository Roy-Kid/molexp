# Asset Management

Asset management in MolExp provides content-addressable storage for workflow outputs with automatic deduplication. Assets are tracked and linked to runs, enabling full traceability of produced artifacts.

## What Asset Management Is

Assets are reusable digital artifacts produced by workflows—files, data structures, or any persistent resources. MolExp's asset system stores assets with content-based addressing (using content hashes), automatically deduplicates identical content, and links assets to the runs that produced them.

The asset system provides a complete lifecycle: creation, storage, retrieval, and tracking. Assets are stored in a content-addressable manner, meaning identical content is stored only once, saving space and enabling efficient sharing between runs.

## Why This Design

Content-addressable storage has several advantages. First, automatic deduplication saves storage space—if multiple runs produce identical results, only one copy is stored. Second, it enables efficient sharing—runs can reference the same asset without duplication.

Third, content hashing provides integrity verification—you can verify that an asset hasn't been corrupted by comparing its hash. Fourth, the system provides full traceability—every asset is linked to the run that produced it, making it easy to understand provenance.

Finally, the design integrates seamlessly with the workspace architecture—assets are automatically tracked when registered during workflow execution, requiring minimal manual intervention.

## How to Use

### Registering Assets

The simplest way to register assets is using `register_asset()` during task execution:

```python
from molexp.assets import register_asset
from molexp.workflow.node import Task
from pydantic import BaseModel

class SaveTask(Task[BaseModel, str]):
    config_type = BaseModel
    
    def execute(self, data: list) -> str:
        # Save data to file
        output_path = "results.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(map(str, data)))
        
        # Register asset (requires run context)
        register_asset(
            output_path,
            label="results",
            meta={"row_count": len(data)},
        )
        
        return output_path
```

When executed within a run context, `register_asset()` automatically:
1. Computes content hash for deduplication
2. Stores asset in workspace repository
3. Creates asset reference linking to the run
4. Updates run's asset references

### Manual Asset Management

You can also manage assets directly through the workspace:

```python
from molexp.workspace.core import Workspace
from molexp.models import Asset, AssetType, AssetFile
from datetime import datetime
from pathlib import Path

# Create workspace
workspace = Workspace.from_path("./workspace")

# Create asset metadata
asset = Asset(
    asset_id="asset_001",
    type=AssetType.STRUCTURE,
    format="pdb",
    created_at=datetime.now(),
    size_bytes=1024,
    content_hash="abc123...",
    files=[
        AssetFile(
            path="structure.pdb",
            size=1024,
            hash="abc123...",
        ),
    ],
    metadata={"description": "Protein structure"},
)

# Store asset
asset_id = workspace.store_asset(asset, Path("structure.pdb"))

# Retrieve asset
retrieved_asset = workspace.get_asset(asset_id)

# Download asset data
workspace.retrieve_asset(asset_id, Path("downloaded.pdb"))
```

### Finding Assets by Content

You can find existing assets by content hash:

```python
# Compute hash of a file
from molexp.utils.id import compute_content_hash

content_hash = compute_content_hash(Path("my_file.txt"))

# Check if asset exists
existing_asset_id = workspace.find_asset_by_hash(content_hash)

if existing_asset_id:
    print(f"Asset already exists: {existing_asset_id}")
    # Reuse existing asset instead of creating new one
else:
    # Create new asset
    asset_id = workspace.store_asset(new_asset, Path("my_file.txt"))
```

This enables efficient asset reuse—if you produce the same content in different runs, you can reuse the existing asset instead of duplicating storage.

### Accessing Run Assets

You can retrieve all assets produced by a run:

```python
# Get asset references for a run
asset_refs = workspace.get_asset_refs(
    id="my_project",
    id="exp_1",
    id="run_001",
)

if asset_refs:
    print(f"Run produced {len(asset_refs.outputs)} assets")
    for ref in asset_refs.outputs:
        asset = workspace.get_asset(ref.asset_id)
        print(f"Asset: {asset.asset_id} ({asset.type})")
        print(f"Role: {ref.role}")
```

### Asset Types

MolExp recognizes different asset types for better organization:

```python
from molexp.models import AssetType

# Structure files
asset = Asset(
    asset_id="struct_001",
    type=AssetType.STRUCTURE,  # .pdb, .xyz, .mol2
    format="pdb",
    # ...
)

# Trajectory files
asset = Asset(
    asset_id="traj_001",
    type=AssetType.TRAJECTORY,  # .xtc, .dcd, .trr
    format="xtc",
    # ...
)

# Images
asset = Asset(
    asset_id="img_001",
    type=AssetType.IMAGE,  # .png, .jpg
    format="png",
    # ...
)

# Tables/data
asset = Asset(
    asset_id="data_001",
    type=AssetType.TABLE,  # .csv, .json
    format="csv",
    # ...
)
```

Asset types help organize and filter assets, making it easier to find specific types of results.

### Complete Example

Here's a complete example showing asset lifecycle:

```python
from molexp.workspace.core import Workspace
from molexp.workflow.context import RunContext, use_run_context
from molexp.assets import AssetRepo, register_asset
from molexp.models import AssetType
from pathlib import Path

def main():
    # Set up workspace
    workspace = Workspace.from_path("./workspace")
    
    # Create project, experiment, run
    project = workspace.create_project("demo", name="Demo")
    experiment = workspace.create_experiment(
        id="demo",
        id="demo_exp",
        name="Demo Experiment",
        workflow_source="demo.py",
    )
    run = workspace.create_run(
        id="demo",
        id="demo_exp",
        parameters={},
        workflow_file="demo.py",
    )
    
    # Create run context
    ctx = RunContext(
        asset_repo=AssetRepo(),
        id=run.id,
        run_metadata=run,
        workspace=workspace,
    )
    
    # Execute workflow with asset registration
    with use_run_context(ctx):
        # Simulate workflow producing assets
        output1 = Path("result1.txt")
        output1.write_text("Result 1")
        register_asset(output1, label="primary_result")
        
        output2 = Path("result2.txt")
        output2.write_text("Result 2")
        register_asset(output2, label="secondary_result")
    
    # Retrieve assets
    asset_refs = workspace.get_asset_refs("demo", "demo_exp", run.id)
    if asset_refs:
        print(f"Run produced {len(asset_refs.outputs)} assets:")
        for ref in asset_refs.outputs:
            asset = workspace.get_asset(ref.asset_id)
            print(f"  - {ref.role}: {asset.asset_id} ({asset.type})")

if __name__ == "__main__":
    main()
```

Asset management provides a complete solution for tracking, storing, and retrieving workflow outputs, ensuring full traceability and efficient storage through content-addressable deduplication.



