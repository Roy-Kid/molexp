"""Asset management with hierarchical libraries and workflows.

This module provides Asset, AssetLibrary, and AssetWorkflow classes for managing
assets at different scopes (workspace, project, experiment, run) with support for
automated workflows.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .utils import generate_asset_id


class Asset(BaseModel):
    """Asset with managed payload storage.
    
    Assets are created through AssetLibrary and have auto-generated IDs and timestamps.
    Each asset has a managed payload directory containing the actual content.
    
    Args:
        asset_id: Auto-generated unique identifier.
        name: User-provided label.
        library_root: Root directory of the asset library.
        created_at: Auto-generated creation timestamp.
        metadata: Additional metadata.
    
    Example:
        >>> library = AssetLibrary(Path("/workspace/assets"))
        >>> asset = library.import_asset("dataset", "/data/qm9.tar.bz2")
        >>> print(asset.uri)  # "asset://abc123def456"
        >>> print(asset.path)  # /workspace/assets/abc123def456/payload
    """
    
    asset_id: str = Field(..., description="Auto-generated unique identifier")
    name: str = Field(..., description="User-provided label")
    library_root: Path = Field(..., description="Asset library root directory")
    created_at: datetime = Field(..., description="Auto-generated creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def uri(self) -> str:
        """Get asset URI.
        
        Returns:
            URI in format 'asset://{asset_id}'.
        """
        return f"asset://{self.asset_id}"
    
    @property
    def id(self) -> str:
        """Entity ID for indexed folder system.
        
        Returns:
            Asset ID.
        """
        return self.asset_id
    
    @property
    def path(self) -> Path:
        """Get filesystem path for the asset payload.

        Returns:
            Path to asset payload directory.
        """
        return self.library_root / self.asset_id / "payload"
    
    def update_metadata(self, **metadata_updates: Any) -> None:
        """Update asset metadata and save to disk.

        Args:
            **metadata_updates: Metadata fields to update as keyword arguments.
        """
        # Update metadata in asset object
        self.metadata.update(metadata_updates)

        # Save updated metadata to disk
        asset_dir = self.library_root / self.id
        metadata_file = asset_dir / "asset.json"

        with open(metadata_file, 'w') as f:
            json.dump({
                "asset_id": self.asset_id,
                "name": self.name,
                "created_at": self.created_at.isoformat(),
                "metadata": self.metadata
            }, f, indent=2)

    def is_ready(self) -> bool:
        """Check if asset is ready for use.

        Returns:
            True if asset status is 'ready', False otherwise.
        """
        return self.metadata.get("status") == "ready"

    def mark_ready(self) -> None:
        """Mark asset as ready and save to disk."""
        self.update_metadata(status="ready")


class AssetWorkflow:
    """Workflow for downloading and processing assets.
    
    Workflows consist of multiple steps that execute sequentially, passing
    results between steps via kwargs.
    
    Example:
        >>> def download_step(**kwargs):
        ...     download_file(kwargs['url'], "/tmp/data.tar")
        ...     return {"asset_path": "/tmp/data.tar"}
        >>> 
        >>> def extract_step(**kwargs):
        ...     extract_tarball(kwargs['asset_path'], "/data/extracted")
        ...     return {"asset_path": "/data/extracted", "asset_name": "dataset"}
        >>> 
        >>> workflow = AssetWorkflow("download_dataset", [download_step, extract_step])
        >>> asset = library.run_workflow("download_dataset", url="https://example.com/data.tar")
    """
    
    def __init__(self, name: str, steps: list[Callable]):
        """Initialize workflow.
        
        Args:
            name: Workflow name
            steps: List of callable steps to execute sequentially
        """
        self.name = name
        self.steps = steps
    
    def execute(self, library: AssetLibrary, **kwargs) -> Asset:
        """Execute workflow steps to create asset.
        
        Each step receives kwargs and returns a dict that updates kwargs for the next step.
        The final step must return 'asset_name' and 'asset_path' in kwargs.
        
        Args:
            library: AssetLibrary to create the final asset in
            **kwargs: Initial parameters for the workflow
            
        Returns:
            Created Asset
            
        Raises:
            ValueError: If final step doesn't return required keys
        """
        for step in self.steps:
            result = step(**kwargs)
            if result:
                kwargs.update(result)
        
        # Final step must provide asset_name and asset_path
        if 'asset_name' not in kwargs or 'asset_path' not in kwargs:
            raise ValueError(
                f"Workflow '{self.name}' must return 'asset_name' and 'asset_path' "
                f"in final step. Got: {list(kwargs.keys())}"
            )
        
        return library.create_asset(
            name=kwargs['asset_name'],
            path=kwargs['asset_path']
        )


class AssetLibrary:
    """Asset library for a specific scope (workspace/project/experiment/run).
    
    Each entity (workspace, project, experiment, run) has its own AssetLibrary.
    Libraries are isolated - assets in one library don't appear in parent/child libraries.
    
    Example:
        >>> # Workspace-level library
        >>> workspace_lib = AssetLibrary(Path("/workspace/assets"))
        >>> asset = workspace_lib.create_asset("bert", "/models/bert.pt")
        >>> 
        >>> # Project-level library
        >>> project_lib = AssetLibrary(Path("/workspace/projects/qm9/assets"))
        >>> dataset = project_lib.create_asset("qm9", "/data/qm9.tar.bz2")
        >>> 
        >>> # Libraries are isolated
        >>> workspace_lib.list_assets()  # Only contains "bert"
        >>> project_lib.list_assets()  # Only contains "qm9"
    """
    
    def __init__(self, root: Path):
        """Initialize asset library.
        
        Args:
            root: Root directory for this library's assets
        """
        self.root = root
        self._index_file = self.root / "index.json"
        self._workflows: dict[str, AssetWorkflow] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load asset index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                self._index: dict[str, str] = json.load(f)
        else:
            self._index: dict[str, str] = {}
    
    def _save_index(self) -> None:
        """Save asset index to disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def import_asset(
        self,
        name: str,
        src: str | Path,
        action: Literal["copy", "move", "symlink", "hardlink"] = "copy",
        meta: dict[str, Any] | None = None,
    ) -> Asset:
        """Import asset with managed payload storage.
        
        Registers an asset and materializes its content into a managed location
        under the asset store with support for different import actions.
        
        Storage layout:
            {library_root}/{asset_id}/
                asset.json      - metadata file
                payload/        - actual content (file or directory)
        
        Args:
            name: Human-readable label for the asset.
            src: Local path (file or directory) to import.
            action: How to bring content into store:
                - "copy": Copy source into store (default)
                - "move": Move source into store
                - "symlink": Create symlink in store pointing to source
                - "hardlink": Create hardlinks (best-effort, fallback to copy)
            meta: Optional metadata to persist with asset.
            
        Returns:
            Asset object representing the imported asset.
            
        Raises:
            ValueError: If asset with this name already exists.
            FileNotFoundError: If source path doesn't exist.
        """
        if name in self._index:
            raise ValueError(f"Asset '{name}' already exists in this library")
        
        source_path = Path(src).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {src}")
        
        # Auto-generate asset ID and timestamp
        asset_id = generate_asset_id()
        created_at = datetime.now()
        
        # Create asset directory structure
        asset_dir = self.root / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = asset_dir / "payload"
        
        # Materialize content based on action
        if action == "copy":
            self._import_copy(source_path, payload_dir)
        elif action == "move":
            self._import_move(source_path, payload_dir)
        elif action == "symlink":
            self._import_symlink(source_path, payload_dir)
        elif action == "hardlink":
            self._import_hardlink(source_path, payload_dir)
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Prepare metadata
        metadata = meta.copy() if meta else {}
        metadata["import_action"] = action
        metadata["source_path"] = str(source_path)
        
        # Create asset object
        asset = Asset(
            asset_id=asset_id,
            name=name,
            library_root=self.root,
            created_at=created_at,
            metadata=metadata,
        )
        
        # Save asset metadata
        metadata_file = asset_dir / "asset.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "asset_id": asset_id,
                "name": name,
                "created_at": created_at.isoformat(),
                "metadata": metadata,
            }, f, indent=2)
        
        # Update index
        self._index[name] = asset_id
        self._save_index()
        
        return asset
    
    def _import_copy(self, source: Path, dest: Path) -> None:
        """Copy source to destination.
        
        Args:
            source: Source path (file or directory).
            dest: Destination path.
        """
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        else:
            shutil.copytree(source, dest, dirs_exist_ok=True)
    
    def _import_move(self, source: Path, dest: Path) -> None:
        """Move source to destination.
        
        Args:
            source: Source path (file or directory).
            dest: Destination path.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    
    def _import_symlink(self, source: Path, dest: Path) -> None:
        """Create symlink at destination pointing to source.
        
        Args:
            source: Source path (file or directory).
            dest: Destination symlink path.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)
    
    def _import_hardlink(self, source: Path, dest: Path) -> None:
        """Create hardlinks for files (best-effort, fallback to copy).
        
        For directories, recursively hardlink all files.
        Falls back to copy if hardlink is not supported.
        
        Args:
            source: Source path (file or directory).
            dest: Destination path.
        """
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, dest)
            except (OSError, NotImplementedError):
                # Fallback to copy if hardlink not supported
                shutil.copy2(source, dest)
        else:
            # For directories, recursively hardlink files
            dest.mkdir(parents=True, exist_ok=True)
            for item in source.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(source)
                    dest_file = dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(item, dest_file)
                    except (OSError, NotImplementedError):
                        shutil.copy2(item, dest_file)
    
    def get_asset(self, name: str) -> Asset | None:
        """Get asset by name.
        
        Args:
            name: Asset name
            
        Returns:
            Asset object if found, None otherwise
        """
        if name not in self._index:
            return None
        
        asset_id = self._index[name]
        metadata_file = self.root / asset_id / "asset.json"
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return Asset(
            asset_id=data['asset_id'],
            name=data['name'],
            library_root=self.root,
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )
    
    def list_assets(self) -> list[Asset]:
        """List all assets in this library.
        
        Returns:
            List of Asset objects
        """
        assets = []
        for name in self._index.keys():
            asset = self.get_asset(name)
            if asset is not None:
                assets.append(asset)
        return assets
    
    def update_metadata(self, asset: Asset, **metadata_updates: Any) -> Asset:
        """Update asset metadata and save to disk.
        
        Args:
            asset: Asset object to update
            **metadata_updates: Metadata fields to update as keyword arguments
            
        Returns:
            Updated Asset object
        """
        # Update metadata in asset object
        asset.metadata.update(metadata_updates)
        
        # Save updated metadata to disk
        asset_dir = self.root / asset.id
        metadata_file = asset_dir / "asset.json"
        
        with open(metadata_file, 'w') as f:
            json.dump({
                "asset_id": asset.asset_id,
                "name": asset.name,
                "created_at": asset.created_at.isoformat(),
                "metadata": asset.metadata
            }, f, indent=2)
        
        return asset
    
    def add_workflow(self, name: str, workflow: AssetWorkflow) -> None:
        """Add workflow for downloading/processing assets.
        
        Args:
            name: Workflow name
            workflow: AssetWorkflow instance
        """
        self._workflows[name] = workflow
    
    def run_workflow(self, name: str, **kwargs) -> Asset:
        """Execute workflow to create asset.
        
        Args:
            name: Workflow name
            **kwargs: Parameters to pass to workflow
            
        Returns:
            Created Asset
            
        Raises:
            KeyError: If workflow not found
        """
        if name not in self._workflows:
            raise KeyError(f"Workflow '{name}' not found in this library")
        
        return self._workflows[name].execute(self, **kwargs)
