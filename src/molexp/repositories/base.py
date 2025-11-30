"""Abstract base classes for repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Asset, Experiment, Project, Run


class AssetRepository(ABC):
    """Abstract interface for asset storage."""

    @abstractmethod
    def store(self, asset: Asset, source_path: Path) -> str:
        """Store asset data and metadata.
        
        Args:
            asset: Asset metadata
            source_path: Path to source file/directory
            
        Returns:
            asset_id
        """
        ...

    @abstractmethod
    def retrieve(self, asset_id: str, dest_path: Path) -> None:
        """Retrieve asset data to destination.
        
        Args:
            asset_id: Asset identifier
            dest_path: Destination path
        """
        ...

    @abstractmethod
    def get_meta(self, asset_id: str) -> Asset | None:
        """Get asset metadata.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Asset metadata or None if not found
        """
        ...

    @abstractmethod
    def exists(self, content_hash: str) -> str | None:
        """Check if asset with given hash exists.
        
        Args:
            content_hash: Content hash to search for
            
        Returns:
            asset_id if found, None otherwise
        """
        ...

    @abstractmethod
    def delete(self, asset_id: str) -> None:
        """Delete asset.
        
        Args:
            asset_id: Asset identifier
        """
        ...

    @abstractmethod
    def list_all(self) -> list[Asset]:
        """List all assets.
        
        Returns:
            List of all assets
        """
        ...


class ProjectRepository(ABC):
    """Abstract interface for project storage."""

    @abstractmethod
    def create(self, project: Project) -> Project:
        """Create a new project."""
        ...

    @abstractmethod
    def get(self, project_id: str) -> Project | None:
        """Get project by ID."""
        ...

    @abstractmethod
    def update(self, project: Project) -> Project:
        """Update existing project."""
        ...

    @abstractmethod
    def delete(self, project_id: str) -> None:
        """Delete project."""
        ...

    @abstractmethod
    def list_all(self) -> list[Project]:
        """List all projects."""
        ...


class ExperimentRepository(ABC):
    """Abstract interface for experiment storage."""

    @abstractmethod
    def create(self, experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        ...

    @abstractmethod
    def get(self, project_id: str, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        ...

    @abstractmethod
    def update(self, experiment: Experiment) -> Experiment:
        """Update existing experiment."""
        ...

    @abstractmethod
    def delete(self, project_id: str, experiment_id: str) -> None:
        """Delete experiment."""
        ...

    @abstractmethod
    def list_by_project(self, project_id: str) -> list[Experiment]:
        """List all experiments in a project."""
        ...


class RunRepository(ABC):
    """Abstract interface for run storage."""

    @abstractmethod
    def create(self, run: Run) -> Run:
        """Create a new run."""
        ...

    @abstractmethod
    def get(self, project_id: str, experiment_id: str, run_id: str) -> Run | None:
        """Get run by ID."""
        ...

    @abstractmethod
    def update(self, run: Run) -> Run:
        """Update existing run."""
        ...

    @abstractmethod
    def delete(self, project_id: str, experiment_id: str, run_id: str) -> None:
        """Delete run."""
        ...

    @abstractmethod
    def list_by_experiment(self, project_id: str, experiment_id: str) -> list[Run]:
        """List all runs in an experiment."""
        ...
