"""Workspace with project management and workspace-level asset library.

A Workspace represents the top-level container with runtime behavior.
Construction auto-generates metadata; persistence happens via materialize().
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .asset import AssetLibrary
from .base import _save_metadata, _load_metadata, _reconstruct, _list_children
from .metadata import WorkspaceMetadata, ProjectMetadata
from .project import Project
from .utils import slugify


class Workspace:
    """Top-level workspace with project management and global asset library.
    
    Construction is side-effect free and auto-generates metadata.
    Filesystem operations happen explicitly via materialize().
    
    Each workspace has its own asset library for workspace-level (global) assets.
    
    Example:
        >>> # User provides only what they care about
        >>> workspace = Workspace(root="./my_workspace", name="My Workspace")
        >>> 
        >>> # System fields auto-generated
        >>> assert workspace.id  # Slugified from name
        >>> assert workspace.metadata.created_at  # Timestamp auto-generated
        >>> 
        >>> # No filesystem side effects yet
        >>> # Explicitly materialize to disk
        >>> workspace.materialize()  # NOW directories/files are created
        >>> 
        >>> # Create project
        >>> project = workspace.create_project(name="QM9 Energy Prediction")
    """
    
    def __init__(self, root: Path | str, name: str | None = None):
        """Initialize workspace with user inputs and auto-generate metadata.
        
        Args:
            root: Root directory for workspace (user input)
            name: Human-readable workspace name (user input, defaults to root basename)
        """
        self.root = Path(root).resolve()
        
        # Default name to root directory basename if not provided
        if name is None:
            name = self.root.name
        
        # Auto-generate metadata with system fields
        self.metadata = WorkspaceMetadata(
            id=slugify(name),  # Auto-generated from name
            name=name,
            created_at=datetime.now(),  # Auto-generated timestamp
            updated_at=datetime.now(),
        )
        
        # Runtime-only state
        self._assets_lib = None
    
    # Property proxies for convenient access to metadata fields
    
    @property
    def id(self) -> str:
        """Workspace identifier."""
        return self.metadata.id
    
    @property
    def name(self) -> str:
        """Human-readable workspace name."""
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
    def assets(self) -> AssetLibrary:
        """Workspace-level asset library."""
        if self._assets_lib is None:
            self._assets_lib = AssetLibrary(self.root / "assets")
        return self._assets_lib
    
    def materialize(self) -> None:
        """Create filesystem structure and persist metadata.

        This is the explicit side-effect method that:
        - Creates workspace root directory
        - Writes metadata JSON file
        - Initializes subdirectories (assets, projects)
        """
        self.root.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, self.root / "workspace.json")

    def save(self) -> None:
        """Save updated metadata to disk."""
        self.metadata.updated_at = datetime.now()
        _save_metadata(self.metadata, self.root / "workspace.json")
    
    @classmethod
    def from_env(cls, env_var: str = "MOLEXP_WORKSPACE") -> Workspace:
        """Create workspace from environment variable or current directory.
        
        Args:
            env_var: Environment variable name (default: MOLEXP_WORKSPACE)
            
        Returns:
            Workspace instance loaded from disk
        """
        workspace_path = os.environ.get(env_var)
        if workspace_path:
            return cls.load(Path(workspace_path))
        return cls.load(Path.cwd())
    
    @classmethod
    def load(cls, root: Path | str) -> Workspace:
        """Load workspace from disk.
        
        Args:
            root: Path to workspace root
            
        Returns:
            Workspace instance loaded from metadata
        """
        root = Path(root).resolve()
        metadata_file = root / "workspace.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Workspace metadata not found at {metadata_file}")

        workspace_metadata = _load_metadata(WorkspaceMetadata, metadata_file)
        return _reconstruct(cls, {
            "root": root,
            "metadata": workspace_metadata,
            "_assets_lib": None,
        })

    @classmethod
    def from_path(cls, root: Path | str) -> Workspace:
        """Load an existing workspace or initialize a new one at path.

        If workspace metadata exists, it is loaded. Otherwise a new workspace is
        created and materialized on disk.
        """
        root = Path(root).resolve()
        metadata_file = root / "workspace.json"
        if metadata_file.exists():
            return cls.load(root)

        workspace = cls(root=root, name=root.name)
        workspace.materialize()
        return workspace
    
    # ============ Project Operations ============
    
    def create_project(self, name: str) -> Project:
        """Create a new project.
        
        Args:
            name: Human-readable project name (user input)
            
        Returns:
            Created Project (already materialized)
            
        Raises:
            ValueError: If project with this ID already exists
        """
        # Construct project (no side effects)
        project = Project(name=name, workspace=self)
        
        # Check if project already exists
        project_dir = self.root / "projects" / project.id
        if project_dir.exists():
            raise ValueError(f"Project '{project.id}' already exists")
        
        # Explicitly materialize
        project.materialize()

        # Register project in workspace metadata
        if project.id not in self.metadata.projects:
            self.metadata.projects.append(project.id)
            self.save()

        return project
    
    def get_project(self, name_or_id: str) -> Project | None:
        """Get project by name or ID (load from disk).
        
        Args:
            name_or_id: Project name or id
            
        Returns:
            Project object if found, None otherwise
        """
        # Try as ID first
        id = name_or_id
        project_dir = self.root / "projects" / id

        # If not found, try as name (slugified)
        if not project_dir.exists():
            id = slugify(name_or_id)
            project_dir = self.root / "projects" / id

        if not project_dir.exists():
            return None

        metadata_file = project_dir / "project.json"
        project_metadata = _load_metadata(ProjectMetadata, metadata_file)
        return _reconstruct(Project, {
            "workspace": self,
            "metadata": project_metadata,
            "_assets_lib": None,
        })
    
    def list_projects(self) -> list[Project]:
        """List all projects.

        Returns:
            List of Project objects
        """
        return _list_children(
            children_dir=self.root / "projects",
            metadata_filename="project.json",
            metadata_cls=ProjectMetadata,
            child_cls=Project,
            attrs_factory=lambda m: {"workspace": self, "metadata": m, "_assets_lib": None},
        )
    
    def delete_project(self, id: str) -> None:
        """Delete project.
        
        Args:
            id: Project identifier
            
        Raises:
            KeyError: If project not found
        """
        project_dir = self.root / "projects" / id
        if not project_dir.exists():
            raise KeyError(f"Project '{id}' not found")
        
        import shutil
        shutil.rmtree(project_dir)
