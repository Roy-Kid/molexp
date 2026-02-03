"""Experiment entity with run management.

An Experiment represents a research experiment with runtime behavior and references.
Construction auto-generates metadata; persistence happens via materialize().
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project
    from .workspace import Workspace

from .base import _save_metadata, _load_metadata, _reconstruct, _list_children
from .metadata import ExperimentMetadata, RunMetadata
from .run import Run
from .utils import generate_id


class Experiment:
    """Research experiment container with runtime behavior.
    
    Construction is side-effect free and auto-generates metadata.
    Filesystem operations happen explicitly via materialize().
    
    Example:
        >>> # User provides only what they care about
        >>> experiment = Experiment(
        ...     name="Hyperparameter Search",
        ...     project=project
        ... )
        >>> 
        >>> # System fields auto-generated
        >>> assert experiment.id  # UUID auto-generated
        >>> assert experiment.metadata.created_at  # Timestamp auto-generated
        >>> 
        >>> # Workspace accessed via project (respects hierarchy)
        >>> assert experiment.workspace is project.workspace
        >>> 
        >>> # No filesystem side effects yet
        >>> # Explicitly materialize to disk
        >>> experiment.materialize()  # NOW directories/files are created
    """
    
    def __init__(
        self,
        name: str,
        project: Project,
        id: str | None = None,
    ):
        """Initialize experiment with user inputs and auto-generate metadata.

        Args:
            name: Human-readable experiment name (user input)
            project: Parent project (runtime dependency)
            id: Optional custom ID (if None, auto-generates UUID)
        """
        self.project = project

        # Auto-generate metadata with system fields
        self.metadata = ExperimentMetadata(
            id=id if id is not None else generate_id(),
            name=name,
            created_at=datetime.now(),  # Auto-generated timestamp
            updated_at=datetime.now(),
            description="",
            tags=[],
            config={},
        )
        
        # Runtime-only cache
        self._assets_lib = None
    
    # Property proxies for convenient access to metadata fields
    
    @property
    def id(self) -> str:
        """Experiment identifier."""
        return self.metadata.id
    
    @property
    def name(self) -> str:
        """Human-readable experiment name."""
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
        """Experiment description."""
        return self.metadata.description

    @property
    def tags(self) -> list[str]:
        """Experiment tags."""
        return self.metadata.tags

    @property
    def config(self) -> dict[str, Any]:
        """Experiment configuration."""
        return self.metadata.config
    
    @property
    def workspace(self) -> Workspace:
        """Access workspace through project (respects hierarchy)."""
        return self.project.workspace
    
    @property
    def assets(self):
        """Experiment-level asset library."""
        from .asset import AssetLibrary
        
        assets_dir = (
            self.workspace.root / "projects" / self.project.id / 
            "experiments" / self.id / "assets"
        )
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(assets_dir)
        return self._assets_lib
    
    def materialize(self) -> None:
        """Create filesystem structure and persist metadata.

        This is the explicit side-effect method that:
        - Creates experiment directory
        - Writes metadata JSON file
        - Initializes subdirectories (assets, runs)
        """
        experiment_dir = (
            self.workspace.root / "projects" / self.project.id /
            "experiments" / self.id
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, experiment_dir / "experiment.json")
    
    def save(self) -> None:
        """Save updated metadata to disk."""
        experiment_dir = (
            self.workspace.root / "projects" / self.project.id /
            "experiments" / self.id
        )
        self.metadata.updated_at = datetime.now()
        _save_metadata(self.metadata, experiment_dir / "experiment.json")
    
    def create_run(self, parameters: dict[str, Any] | None = None) -> Run:
        """Create run in this experiment.
        
        Args:
            parameters: Run parameters (user input)
            
        Returns:
            Created Run (not yet materialized)
        """
        # Construct run (no side effects)
        run = Run(
            experiment=self,
            parameters=parameters,
        )
        
        # Explicitly materialize
        run.materialize()
        
        return run
    
    def get_run(self, run_id: str) -> Run | None:
        """Get run by ID (load from disk).

        Args:
            run_id: Run UUID

        Returns:
            Run object if found, None otherwise
        """
        run_dir = (
            self.workspace.root / "projects" / self.project.id / "experiments" /
            self.id / "runs" / run_id
        )

        if not run_dir.exists():
            return None

        metadata = _load_metadata(RunMetadata, run_dir / "run.json")
        return _reconstruct(Run, {
            "experiment": self,
            "metadata": metadata,
        })
    
    def list_runs(self) -> list[Run]:
        """List runs in this experiment.

        Returns:
            List of Run objects
        """
        return _list_children(
            children_dir=(
                self.workspace.root / "projects" / self.project.id / "experiments" /
                self.id / "runs"
            ),
            metadata_filename="run.json",
            metadata_cls=RunMetadata,
            child_cls=Run,
            attrs_factory=lambda m: {"experiment": self, "metadata": m},
        )
