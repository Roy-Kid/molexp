"""Workspace management for Project-Experiment-Run architecture."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.id import generate_run_id
from ..models import (
    Asset,
    AssetRef,
    AssetRefsCollection,
    AssetType,
    Experiment,
    Project,
    Run,
    RunContextSnapshot,
    RunStatus,
    WorkflowSnapshot,
    WorkflowTemplate,
)
from ..repositories import (
    FileSystemAssetRepo,
    FileSystemExperimentRepo,
    FileSystemProjectRepo,
    FileSystemRunRepo,
)


class Workspace:
    """Central access point to all repositories and workspace operations."""

    def __init__(self, root: Path) -> None:
        """Initialize workspace.
        
        Args:
            root: Root directory for workspace
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Initialize repositories
        self.projects = FileSystemProjectRepo(self.root / "projects")
        self.experiments = FileSystemExperimentRepo(self.root / "projects")
        self.runs = FileSystemRunRepo(self.root / "projects")
        self.assets = FileSystemAssetRepo(self.root / "assets")

    @classmethod
    def from_env(cls, env_var: str = "MOLEXP_WORKSPACE") -> Workspace:
        """Create workspace from environment variable or current directory.
        
        Args:
            env_var: Environment variable name (default: MOLEXP_WORKSPACE)
            
        Returns:
            Workspace instance
        """
        workspace_path = os.environ.get(env_var)
        if workspace_path:
            return cls(Path(workspace_path))
        # Default to current directory
        return cls(Path.cwd())

    @classmethod
    def from_path(cls, path: str | Path) -> Workspace:
        """Create workspace from explicit path.
        
        Args:
            path: Path to workspace root
            
        Returns:
            Workspace instance
        """
        return cls(Path(path))

    # ============ Project Operations ============

    def create_project(
        self,
        project_id: str,
        name: str,
        description: str = "",
        owner: str = "",
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Project:
        """Create a new project.
        
        Args:
            project_id: Unique project identifier (slug)
            name: Human-readable name
            description: Project description
            owner: Project owner
            tags: List of tags
            config: Project configuration
            
        Returns:
            Created Project
        """
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            owner=owner,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config=config or {},
        )
        return self.projects.create(project)

    def get_project(self, project_id: str) -> Project | None:
        """Get project by ID."""
        return self.projects.get(project_id)

    def list_projects(self) -> list[Project]:
        """List all projects."""
        return self.projects.list_all()

    def delete_project(self, project_id: str) -> None:
        """Delete project."""
        self.projects.delete(project_id)

    # ============ Experiment Operations ============

    def create_experiment(
        self,
        project_id: str,
        experiment_id: str,
        name: str,
        workflow_source: str,
        description: str = "",
        workflow_type: str = "taskgraph_v1",
        git_commit: str | None = None,
        parameter_space: dict[str, Any] | None = None,
        default_inputs: list[AssetRef] | None = None,
    ) -> Experiment:
        """Create a new experiment.
        
        Args:
            project_id: Parent project ID
            experiment_id: Unique experiment identifier (slug)
            name: Human-readable name
            workflow_source: Path to workflow file
            description: Experiment description
            workflow_type: Workflow type
            git_commit: Git commit hash
            parameter_space: Parameter space definition
            default_inputs: Default input assets
            
        Returns:
            Created Experiment
        """
        experiment = Experiment(
            experiment_id=experiment_id,
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            workflow_template=WorkflowTemplate(
                type=workflow_type,
                source=workflow_source,
                git_commit=git_commit,
            ),
            parameter_space=parameter_space or {},
            default_inputs=default_inputs or [],
        )
        return self.experiments.create(experiment)

    def get_experiment(self, project_id: str, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self.experiments.get(project_id, experiment_id)

    def list_experiments(self, project_id: str) -> list[Experiment]:
        """List all experiments in a project."""
        return self.experiments.list_by_project(project_id)

    def delete_experiment(self, project_id: str, experiment_id: str) -> None:
        """Delete experiment."""
        self.experiments.delete(project_id, experiment_id)

    # ============ Run Operations ============

    def create_run(
        self,
        project_id: str,
        experiment_id: str,
        parameters: dict[str, Any],
        workflow_file: str,
        git_commit: str | None = None,
        run_id: str | None = None,
    ) -> Run:
        """Create a new run.
        
        Args:
            project_id: Parent project ID
            experiment_id: Parent experiment ID
            parameters: Run parameters
            workflow_file: Workflow file path
            git_commit: Git commit hash
            run_id: Optional run ID (auto-generated if not provided)
            
        Returns:
            Created Run
        """
        if run_id is None:
            run_id = generate_run_id()
        
        run = Run(
            run_id=run_id,
            project_id=project_id,
            experiment_id=experiment_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=RunStatus.PENDING,
            parameters=parameters,
            workflow_snapshot=WorkflowSnapshot(
                git_commit=git_commit,
                workflow_file=workflow_file,
            ),
            executor_info={},
            working_dir=f"projects/{project_id}/experiments/{experiment_id}/runs/{run_id}",
        )
        return self.runs.create(run)

    def get_run(self, project_id: str, experiment_id: str, run_id: str) -> Run | None:
        """Get run by ID."""
        return self.runs.get(project_id, experiment_id, run_id)

    def update_run(self, run: Run) -> Run:
        """Update run."""
        return self.runs.update(run)

    def list_runs(self, project_id: str, experiment_id: str) -> list[Run]:
        """List all runs in an experiment."""
        return self.runs.list_by_experiment(project_id, experiment_id)

    def delete_run(self, project_id: str, experiment_id: str, run_id: str) -> None:
        """Delete run."""
        self.runs.delete(project_id, experiment_id, run_id)

    # ============ Run Context Operations ============

    def save_run_context(
        self,
        project_id: str,
        experiment_id: str,
        run_id: str,
        context: RunContextSnapshot,
    ) -> None:
        """Save run context snapshot."""
        self.runs.save_context(project_id, experiment_id, run_id, context)

    def get_run_context(
        self, project_id: str, experiment_id: str, run_id: str
    ) -> RunContextSnapshot | None:
        """Get run context snapshot."""
        return self.runs.get_context(project_id, experiment_id, run_id)

    # ============ Asset Reference Operations ============

    def save_asset_refs(
        self,
        project_id: str,
        experiment_id: str,
        run_id: str,
        refs: AssetRefsCollection,
    ) -> None:
        """Save asset references."""
        self.runs.save_asset_refs(project_id, experiment_id, run_id, refs)

    def get_asset_refs(
        self, project_id: str, experiment_id: str, run_id: str
    ) -> AssetRefsCollection | None:
        """Get asset references."""
        return self.runs.get_asset_refs(project_id, experiment_id, run_id)

    # ============ Asset Operations ============

    def store_asset(self, asset: Asset, source_path: Path) -> str:
        """Store asset in repository."""
        return self.assets.store(asset, source_path)

    def get_asset(self, asset_id: str) -> Asset | None:
        """Get asset metadata."""
        return self.assets.get_meta(asset_id)

    def retrieve_asset(self, asset_id: str, dest_path: Path) -> None:
        """Retrieve asset data."""
        self.assets.retrieve(asset_id, dest_path)

    def find_asset_by_hash(self, content_hash: str) -> str | None:
        """Find asset by content hash."""
        return self.assets.exists(content_hash)

    def list_assets(self) -> list[Asset]:
        """List all assets."""
        return self.assets.list_all()

    def delete_asset(self, asset_id: str) -> None:
        """Delete asset."""
        self.assets.delete(asset_id)
