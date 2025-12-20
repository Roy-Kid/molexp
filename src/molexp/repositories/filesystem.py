"""File system implementations of repositories."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import yaml

from ..utils.id import compute_content_hash
from .base import (AssetRepository, DuplicateEntityError, EntityNotFoundError,
                   ExperimentRepository, ProjectRepository, RepositoryIOError,
                   RunRepository)
from .indexed import IndexFileManager

if TYPE_CHECKING:
    from ..models import (Asset, AssetRefsCollection, Experiment, Project, Run,
                          RunContextSnapshot)


logger = logging.getLogger(__name__)


# ============================================================================
# Atomic File Operations
# ============================================================================


@contextmanager
def atomic_write(target_path: Path, mode: str = "w") -> Iterator[Path]:
    """Context manager for atomic file writes.

    Writes to a temporary file first, then atomically renames to target.
    This prevents data corruption if the write is interrupted.

    Args:
        target_path: Final destination path
        mode: File mode ("w" for text, "wb" for binary)

    Yields:
        Temporary file path to write to

    Raises:
        RepositoryIOError: If the write operation fails
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_path)

    try:
        os.close(fd)  # We'll reopen with proper mode
        yield tmp_path

        # Atomic rename (works on POSIX, best-effort on Windows)
        tmp_path.replace(target_path)
        logger.debug(f"Atomically wrote {target_path}")

    except Exception as e:
        # Clean up temp file on failure
        tmp_path.unlink(missing_ok=True)
        raise RepositoryIOError("write", str(target_path), e) from e


def safe_yaml_dump(data: dict, path: Path) -> None:
    """Safely dump YAML data to a file using atomic write."""
    with atomic_write(path) as tmp_path:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def safe_json_dump(data: dict, path: Path) -> None:
    """Safely dump JSON data to a file using atomic write."""
    with atomic_write(path) as tmp_path:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)


class FileSystemAssetRepo(AssetRepository):
    """File system-based asset repository with content-addressable storage."""

    def __init__(self, root: Path) -> None:
        """Initialize repository.

        Args:
            root: Root directory for assets (e.g., workspace/assets)
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Build hash index for fast lookups
        self._hash_index: dict[str, str] = {}
        self._build_hash_index()

    def _build_hash_index(self) -> None:
        """Build index of content_hash -> asset_id."""
        self._hash_index.clear()
        for asset_dir in self.root.iterdir():
            if not asset_dir.is_dir():
                continue
            meta_path = asset_dir / "meta.yaml"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = yaml.safe_load(f)
                    if "content_hash" in meta and "asset_id" in meta:
                        self._hash_index[meta["content_hash"]] = meta["asset_id"]
                except Exception:
                    continue

    def store(self, asset: Asset, source_path: Path) -> str:
        """Store asset data and metadata."""
        asset_dir = self.root / asset.asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory
        data_dir = asset_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Copy source file(s) to data directory
        source_path = Path(source_path)
        if source_path.is_file():
            dest = data_dir / source_path.name
            shutil.copy2(source_path, dest)
        elif source_path.is_dir():
            shutil.copytree(source_path, data_dir, dirs_exist_ok=True)

        # Write metadata
        meta_path = asset_dir / "meta.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump(asset.model_dump(mode="json"), f, sort_keys=False)

        # Update hash index
        self._hash_index[asset.content_hash] = asset.asset_id

        return asset.asset_id

    def retrieve(self, asset_id: str, dest_path: Path) -> None:
        """Retrieve asset data to destination."""
        asset_dir = self.root / asset_id
        data_dir = asset_dir / "data"

        if not data_dir.exists():
            raise FileNotFoundError(f"Asset {asset_id} data not found")

        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy all files from data directory
        if data_dir.is_dir():
            files = list(data_dir.iterdir())
            if len(files) == 1 and files[0].is_file():
                # Single file: copy directly to dest_path
                shutil.copy2(files[0], dest_path)
            else:
                # Multiple files or directories: copy entire data dir
                shutil.copytree(data_dir, dest_path, dirs_exist_ok=True)

    def get_meta(self, asset_id: str) -> Asset | None:
        """Get asset metadata."""
        from ..models import Asset

        meta_path = self.root / asset_id / "meta.yaml"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            data = yaml.safe_load(f)

        return Asset.model_validate(data)

    def exists(self, content_hash: str) -> str | None:
        """Check if asset with given hash exists."""
        return self._hash_index.get(content_hash)

    def delete(self, asset_id: str) -> None:
        """Delete asset."""
        asset_dir = self.root / asset_id
        if asset_dir.exists():
            # Remove from hash index
            meta = self.get_meta(asset_id)
            if meta and meta.content_hash in self._hash_index:
                del self._hash_index[meta.content_hash]
            # Delete directory
            shutil.rmtree(asset_dir)

    def list_all(self) -> list[Asset]:
        """List all assets."""
        assets = []
        for asset_dir in self.root.iterdir():
            if asset_dir.is_dir():
                meta = self.get_meta(asset_dir.name)
                if meta:
                    assets.append(meta)
        return assets


class FileSystemProjectRepo(ProjectRepository):
    """File system-based project repository."""

    def __init__(self, root: Path) -> None:
        """Initialize repository.

        Args:
            root: Root directory for projects (e.g., workspace/projects)
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, project: Project) -> Project:
        """Create a new project."""
        from ..models import Project

        project_dir = self.root / project.project_id
        if project_dir.exists():
            raise ValueError(f"Project {project.project_id} already exists")

        project_dir.mkdir(parents=True)
        (project_dir / "experiments").mkdir()

        # Write project metadata using IndexFileManager
        IndexFileManager.write_index(project_dir, project)

        return project

    def get(self, project_id: str) -> Project | None:
        """Get project by ID."""
        from ..models import Project

        project_dir = self.root / project_id
        if not project_dir.exists():
            return None

        return IndexFileManager.read_index(project_dir, "project", Project)

    def update(self, project: Project) -> Project:
        """Update existing project."""
        from datetime import datetime

        project_dir = self.root / project.project_id
        if not project_dir.exists():
            raise ValueError(f"Project {project.project_id} not found")

        # Update timestamp
        project.updated_at = datetime.now()

        IndexFileManager.write_index(project_dir, project)

        return project

    def delete(self, project_id: str) -> None:
        """Delete project."""
        project_dir = self.root / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)

    def list_all(self) -> list[Project]:
        """List all projects."""
        from ..models import Project

        projects = []
        for project_dir in self.root.iterdir():
            if project_dir.is_dir():
                project = self.get(project_dir.name)
                if project:
                    projects.append(project)
        return projects


class FileSystemExperimentRepo(ExperimentRepository):
    """File system-based experiment repository."""

    def __init__(self, root: Path) -> None:
        """Initialize repository.

        Args:
            root: Root directory for projects (e.g., workspace/projects)
        """
        self.root = Path(root)

    def create(self, experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        from ..models import Experiment

        exp_dir = (
            self.root / experiment.project_id / "experiments" / experiment.experiment_id
        )
        if exp_dir.exists():
            raise ValueError(f"Experiment {experiment.experiment_id} already exists")

        exp_dir.mkdir(parents=True)
        (exp_dir / "runs").mkdir()

        # Write experiment metadata using IndexFileManager
        IndexFileManager.write_index(exp_dir, experiment)

        return experiment

    def get(self, project_id: str, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        from ..models import Experiment

        exp_dir = self.root / project_id / "experiments" / experiment_id
        if not exp_dir.exists():
            return None

        return IndexFileManager.read_index(exp_dir, "experiment", Experiment)

    def update(self, experiment: Experiment) -> Experiment:
        """Update existing experiment."""
        from datetime import datetime

        exp_dir = (
            self.root / experiment.project_id / "experiments" / experiment.experiment_id
        )
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment.experiment_id} not found")

        # Update timestamp
        experiment.updated_at = datetime.now()

        IndexFileManager.write_index(exp_dir, experiment)

        return experiment

    def delete(self, project_id: str, experiment_id: str) -> None:
        """Delete experiment."""
        exp_dir = self.root / project_id / "experiments" / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

    def list_by_project(self, project_id: str) -> list[Experiment]:
        """List all experiments in a project."""
        from ..models import Experiment

        experiments = []
        exp_root = self.root / project_id / "experiments"
        if not exp_root.exists():
            return experiments

        for exp_dir in exp_root.iterdir():
            if exp_dir.is_dir():
                exp = self.get(project_id, exp_dir.name)
                if exp:
                    experiments.append(exp)
        return experiments


class FileSystemRunRepo(RunRepository):
    """File system-based run repository."""

    def __init__(self, root: Path) -> None:
        """Initialize repository.

        Args:
            root: Root directory for projects (e.g., workspace/projects)
        """
        self.root = Path(root)

    def create(self, run: Run) -> Run:
        """Create a new run."""
        from ..models import AssetRefsCollection, Run

        run_dir = (
            self.root
            / run.project_id
            / "experiments"
            / run.experiment_id
            / "runs"
            / run.run_id
        )
        if run_dir.exists():
            raise ValueError(f"Run {run.run_id} already exists")

        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "artifacts").mkdir()

        # Write run metadata using IndexFileManager
        IndexFileManager.write_index(run_dir, run)

        # Initialize empty asset_refs
        asset_refs_path = run_dir / "asset_refs.json"
        empty_refs = AssetRefsCollection()
        with open(asset_refs_path, "w") as f:
            json.dump(empty_refs.model_dump(mode="json"), f, indent=2)

        return run

    def get(self, project_id: str, experiment_id: str, run_id: str) -> Run | None:
        """Get run by ID."""
        from ..models import Run

        run_dir = (
            self.root / project_id / "experiments" / experiment_id / "runs" / run_id
        )
        if not run_dir.exists():
            return None

        return IndexFileManager.read_index(run_dir, "run", Run)

    def update(self, run: Run) -> Run:
        """Update existing run."""
        from datetime import datetime

        run_dir = (
            self.root
            / run.project_id
            / "experiments"
            / run.experiment_id
            / "runs"
            / run.run_id
        )
        if not run_dir.exists():
            raise ValueError(f"Run {run.run_id} not found")

        # Update timestamp
        run.updated_at = datetime.now()

        IndexFileManager.write_index(run_dir, run)

        return run

    def delete(self, project_id: str, experiment_id: str, run_id: str) -> None:
        """Delete run."""
        run_dir = (
            self.root / project_id / "experiments" / experiment_id / "runs" / run_id
        )
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def list_by_experiment(self, project_id: str, experiment_id: str) -> list[Run]:
        """List all runs in an experiment."""
        from ..models import Run

        runs = []
        run_root = self.root / project_id / "experiments" / experiment_id / "runs"
        if not run_root.exists():
            return runs

        for run_dir in run_root.iterdir():
            if run_dir.is_dir():
                run = self.get(project_id, experiment_id, run_dir.name)
                if run:
                    runs.append(run)
        return runs

    def save_context(
        self,
        project_id: str,
        experiment_id: str,
        run_id: str,
        context: RunContextSnapshot,
    ) -> None:
        """Save run context snapshot."""
        context_path = (
            self.root
            / project_id
            / "experiments"
            / experiment_id
            / "runs"
            / run_id
            / "context.json"
        )
        with open(context_path, "w") as f:
            json.dump(context.model_dump(mode="json"), f, indent=2)

    def get_context(
        self, project_id: str, experiment_id: str, run_id: str
    ) -> RunContextSnapshot | None:
        """Get run context snapshot."""
        from ..models import RunContextSnapshot

        context_path = (
            self.root
            / project_id
            / "experiments"
            / experiment_id
            / "runs"
            / run_id
            / "context.json"
        )
        if not context_path.exists():
            return None

        with open(context_path) as f:
            data = json.load(f)

        return RunContextSnapshot.model_validate(data)

    def save_asset_refs(
        self,
        project_id: str,
        experiment_id: str,
        run_id: str,
        refs: AssetRefsCollection,
    ) -> None:
        """Save asset references."""
        refs_path = (
            self.root
            / project_id
            / "experiments"
            / experiment_id
            / "runs"
            / run_id
            / "asset_refs.json"
        )
        with open(refs_path, "w") as f:
            json.dump(refs.model_dump(mode="json"), f, indent=2)

    def get_asset_refs(
        self, project_id: str, experiment_id: str, run_id: str
    ) -> AssetRefsCollection | None:
        """Get asset references."""
        from ..models import AssetRefsCollection

        refs_path = (
            self.root
            / project_id
            / "experiments"
            / experiment_id
            / "runs"
            / run_id
            / "asset_refs.json"
        )
        if not refs_path.exists():
            return None

        with open(refs_path) as f:
            data = json.load(f)

        return AssetRefsCollection.model_validate(data)
