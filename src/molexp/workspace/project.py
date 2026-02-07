"""Project entity with experiment management.

A Project represents a research project with runtime behavior and references.
Construction auto-generates metadata; persistence happens via materialize().
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workspace import Workspace

from .base import _save_metadata, _load_metadata, _reconstruct, _list_children
from .metadata import ProjectMetadata, ExperimentMetadata
from .experiment import Experiment
from .utils import slugify


class Project:
    """Research project container with runtime behavior.
    
    Construction is side-effect free and auto-generates metadata.
    Filesystem operations happen explicitly via materialize().
    
    Example:
        >>> # User provides only what they care about
        >>> project = Project(
        ...     name="QM9 Energy Prediction",
        ...     workspace=workspace
        ... )
        >>> 
        >>> # System fields auto-generated
        >>> assert project.id  # Slugified from name
        >>> assert project.metadata.created_at  # Timestamp auto-generated
        >>> 
        >>> # No filesystem side effects yet
        >>> # Explicitly materialize to disk
        >>> project.materialize()  # NOW directories/files are created
    """
    
    def __init__(
        self,
        name: str,
        workspace: Workspace,
    ):
        """Initialize project with user inputs and auto-generate metadata.
        
        Args:
            name: Human-readable project name (user input)
            workspace: Root workspace (runtime dependency)
        """
        self.workspace = workspace
        
        # Auto-generate metadata with system fields
        self.metadata = ProjectMetadata(
            id=slugify(name),  # Auto-generated from name
            name=name,
            created_at=datetime.now(),  # Auto-generated timestamp
            updated_at=datetime.now(),
            description="",
            owner="",
            tags=[],
            config={}
        )
        
        # Runtime-only cache
        self._assets_lib = None
    
    # Property proxies for convenient access to metadata fields
    
    @property
    def id(self) -> str:
        """Project identifier."""
        return self.metadata.id
    
    @property
    def name(self) -> str:
        """Human-readable project name."""
        return self.metadata.name
    
    @property
    def created_at(self):
        """Creation timestamp."""
        return self.metadata.created_at
    
    @property
    def updated_at(self):
        """Last update timestamp."""
        return self.metadata.updated_at
    
    @property
    def description(self) -> str:
        """Project description."""
        return self.metadata.description

    @property
    def owner(self) -> str:
        """Project owner."""
        return self.metadata.owner

    @property
    def tags(self) -> list[str]:
        """Project tags."""
        return self.metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        """Project configuration."""
        return self.metadata.config
    
    @property
    def description(self) -> str:
        """Project description."""
        return self.metadata.description

    @property
    def owner(self) -> str:
        """Project owner."""
        return self.metadata.owner

    @property
    def tags(self) -> list[str]:
        """Project tags."""
        return self.metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        """Project configuration."""
        return self.metadata.config
    
    @property
    def assets(self):
        """Project-level asset library."""
        from .asset import AssetLibrary

        assets_dir = self.workspace.root / "projects" / self.id / "assets"
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(assets_dir)
        return self._assets_lib

    def import_asset(self, name: str, src: str | Path, action: str = "copy", meta: dict[str, Any] | None = None):
        """Import asset and update project metadata.

        Args:
            name: Asset name
            src: Source path
            action: Import action ("copy", "move", "symlink", "hardlink")
            meta: Optional metadata

        Returns:
            Asset object
        """
        asset = self.assets.import_asset(name, src, action, meta)

        # Register asset in project metadata
        if name not in self.metadata.assets:
            self.metadata.assets.append(name)
            self.save()

        return asset

    def create_asset(self, name: str, meta: dict[str, Any] | None = None):
        """Create a new empty asset and update project metadata.

        Args:
            name: Asset name
            meta: Optional metadata

        Returns:
            Asset object with empty payload directory
        """
        from .asset import Asset
        from .utils import generate_asset_id
        from datetime import datetime
        import json

        # Check if asset already exists
        existing_asset = self.assets.get_asset(name)
        if existing_asset:
            return existing_asset

        # Generate asset ID and create directory structure
        asset_id = generate_asset_id()
        created_at = datetime.now()

        asset_dir = self.assets.root / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        payload_dir = asset_dir / "payload"
        payload_dir.mkdir(exist_ok=True)

        # Prepare metadata
        metadata = meta.copy() if meta else {}

        # Create asset object
        asset = Asset(
            asset_id=asset_id,
            name=name,
            library_root=self.assets.root,
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
        self.assets._index[name] = asset_id
        self.assets._save_index()

        # Register asset in project metadata
        if name not in self.metadata.assets:
            self.metadata.assets.append(name)
            self.save()

        return asset

    def materialize(self) -> None:
        """Create filesystem structure and persist metadata.

        This is the explicit side-effect method that:
        - Creates project directory
        - Writes metadata JSON file
        - Initializes subdirectories (assets, experiments)
        """
        project_dir = self.workspace.root / "projects" / self.id
        project_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, project_dir / "project.json")
    
    def save(self) -> None:
        """Save updated metadata to disk."""
        project_dir = self.workspace.root / "projects" / self.id
        self.metadata.updated_at = datetime.now()
        _save_metadata(self.metadata, project_dir / "project.json")
    
    def create_experiment(self, name: str, id: str | None = None, exist_ok: bool = False) -> Experiment:
        """Create experiment in this project.

        Args:
            name: Experiment name (user input)
            id: Optional custom ID (if None, auto-generates UUID)
            exist_ok: If True, return existing experiment if found by name (when id=None)
                     or by id (when id is provided)

        Returns:
            Created Experiment (already materialized)

        Raises:
            ValueError: If experiment with this ID already exists and exist_ok=False
        """
        # If exist_ok=True and no custom id, try to find existing experiment by name
        if exist_ok and id is None:
            experiments_dir = self.workspace.root / "projects" / self.id / "experiments"
            if experiments_dir.exists():
                from .base import _load_metadata, _reconstruct
                from .metadata import ExperimentMetadata

                for exp_dir in experiments_dir.iterdir():
                    if exp_dir.is_dir():
                        metadata_file = exp_dir / "experiment.json"
                        if metadata_file.exists():
                            metadata = _load_metadata(ExperimentMetadata, metadata_file)
                            if metadata.name == name:
                                # Found existing experiment with same name
                                attrs = {
                                    'metadata': metadata,
                                    'project': self,
                                    '_assets_lib': None,
                                }
                                return _reconstruct(Experiment, attrs)

        # Construct experiment (no side effects)
        experiment = Experiment(
            name=name,
            project=self,
            id=id,
        )

        # Check if experiment already exists by ID
        experiment_dir = self.workspace.root / "projects" / self.id / "experiments" / experiment.id
        if experiment_dir.exists():
            if exist_ok:
                # Load existing experiment
                metadata_file = experiment_dir / "experiment.json"
                if metadata_file.exists():
                    from .base import _load_metadata, _reconstruct
                    from .metadata import ExperimentMetadata
                    metadata = _load_metadata(ExperimentMetadata, metadata_file)
                    attrs = {
                        'metadata': metadata,
                        'project': self,
                        '_assets_lib': None,
                    }
                    return _reconstruct(Experiment, attrs)
            raise ValueError(f"Experiment '{experiment.id}' already exists")

        # Explicitly materialize
        experiment.materialize()

        # Register experiment in project metadata
        if experiment.id not in self.metadata.experiments:
            self.metadata.experiments.append(experiment.id)
            self.save()

        return experiment
    
    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID (load from disk).

        Args:
            experiment_id: Experiment UUID

        Returns:
            Experiment object if found, None otherwise
        """
        experiment_dir = (
            self.workspace.root / "projects" / self.id / "experiments" / experiment_id
        )

        if not experiment_dir.exists():
            return None

        metadata = _load_metadata(ExperimentMetadata, experiment_dir / "experiment.json")
        return _reconstruct(Experiment, {
            "project": self,
            "metadata": metadata,
            "_assets_lib": None,
        })
    
    def list_experiments(self) -> list[Experiment]:
        """List experiments in this project.

        Returns:
            List of Experiment objects
        """
        return _list_children(
            children_dir=self.workspace.root / "projects" / self.id / "experiments",
            metadata_filename="experiment.json",
            metadata_cls=ExperimentMetadata,
            child_cls=Experiment,
            attrs_factory=lambda m: {"project": self, "metadata": m, "_assets_lib": None},
        )
